# Código completo SimCLR para copiar para o notebook
# Este arquivo contém todo o código necessário organizado em seções

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

# ===== PROJECTION HEAD E NT-XENT LOSS =====
class ProjectionHead(nn.Module):
    """MLP projection head para SimCLR"""
    def __init__(self, embedding_dim=128, projection_dim=64):
        super(ProjectionHead, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, projection_dim)
        )

    def forward(self, x):
        return self.projection(x)


class NTXentLoss(nn.Module):
    """NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss"""
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        z_i, z_j: embeddings de duas views aumentadas
        Shape: (batch_size, projection_dim)
        """
        batch_size = z_i.shape[0]

        # Normalizar L2
        z_i = F.normalize(z_i, p=2, dim=1)
        z_j = F.normalize(z_j, p=2, dim=1)

        # Concatenar
        z = torch.cat([z_i, z_j], dim=0)  # (2*batch_size, projection_dim)

        # Calcular similaridade
        sim_matrix = torch.mm(z, z.T) / self.temperature  # (2N, 2N)

        # Criar labels: positivos são (i, i+N) e (i+N, i)
        labels = torch.cat([torch.arange(batch_size) + batch_size,
                           torch.arange(batch_size)]).to(z_i.device)

        # Mascarar diagonal (não comparar consigo mesmo)
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z_i.device)
        sim_matrix = sim_matrix.masked_fill(mask, -9e15)

        # Cross entropy
        loss = F.cross_entropy(sim_matrix, labels)

        return loss


# ===== SIMCLR MODEL =====
class SimCLR(nn.Module):
    """Modelo SimCLR completo"""
    def __init__(self, encoder, projection_head):
        super(SimCLR, self).__init__()
        self.encoder = encoder
        self.projection_head = projection_head

    def forward(self, x_i, x_j):
        # Encode ambas as views
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        # Project
        z_i = self.projection_head(h_i)
        z_j = self.projection_head(h_j)

        return z_i, z_j


# ===== DATASET COM AUGMENTAÇÕES =====
class IMUContrastiveDataset(Dataset):
    """
    Dataset que retorna duas views aumentadas de cada janela
    """
    def __init__(self, windows, n_augs=2):
        """
        windows: numpy array (n_windows, time_steps, channels)
        """
        self.windows = windows
        self.n_augs = n_augs

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]  # (T, C)

        # Gerar duas views aumentadas
        view1 = IMUAugmentations.apply_random_augmentations(window, self.n_augs)
        view2 = IMUAugmentations.apply_random_augmentations(window, self.n_augs)

        # Converter para torch e transpor para (C, T)
        view1 = torch.FloatTensor(view1).T  # (C, T)
        view2 = torch.FloatTensor(view2).T  # (C, T)

        return view1, view2


# ===== FUNÇÃO DE CRIAÇÃO DE JANELAS =====
def create_windows(df, window_size=75, overlap=0.5, feature_columns=['accX', 'accY', 'accZ']):
    """
    Cria janelas deslizantes de IMU

    Args:
        df: DataFrame com dados
        window_size: tamanho da janela em samples
        overlap: porcentagem de overlap (0.0 a 1.0)
        feature_columns: colunas de features

    Returns:
        windows: numpy array (n_windows, window_size, n_channels)
        labels: numpy array (n_windows,) - label majoritário na janela
        calves: numpy array (n_windows,) - ID do bezerro
    """
    stride = int(window_size * (1 - overlap))

    windows = []
    labels = []
    calves = []

    # Agrupar por bezerro e comportamento
    for calf_id in df['calfId'].unique():
        df_calf = df[df['calfId'] == calf_id].copy()

        # Iterar por cada comportamento ORIGINAL (não usar 'label')
        for behavior_name in df_calf['behaviour'].unique():
            df_behavior = df_calf[df_calf['behaviour'] == behavior_name].copy()
            # Resetar índice (não ordenar por timestamp que pode não existir)
            df_behavior = df_behavior.reset_index(drop=True)

            # Extrair features do acelerômetro
            features = df_behavior[feature_columns].values

            # Criar janelas com stride
            for i in range(0, len(features) - window_size + 1, stride):
                window = features[i:i+window_size]

                if window.shape[0] == window_size:
                    windows.append(window)
                    labels.append(behavior_name)  # String (nome do comportamento)
                    calves.append(calf_id)

    return np.array(windows), np.array(labels), np.array(calves)


# ===== VARIÁVEIS GLOBAIS =====
# Configurações
NUM_CLASSES = 8
N_AUGMENTATIONS = 2  # Número de augmentações por view
label_encoder = LabelEncoder()

# Inicializar encoder e projection head
encoder = ResNet1DEncoder(in_channels=3, embedding_dim=128)
projection_head = ProjectionHead(embedding_dim=128, projection_dim=64)

# ===== FUNÇÃO DE PRÉ-TREINO =====
def pretrain_simclr(model, train_loader, val_loader, criterion, optimizer, scheduler,
                     device, num_epochs=50, patience=10, save_path='simclr_encoder_best.pth'):
    """
    Pré-treino contrastivo com SimCLR
    """
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(num_epochs):
        # Treino
        model.train()
        train_loss = 0.0
        for (view1, view2) in train_loader:
            view1, view2 = view1.to(device), view2.to(device)

            optimizer.zero_grad()
            z_i, z_j = model(view1, view2)
            loss = criterion(z_i, z_j)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validação
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (view1, view2) in val_loader:
                view1, view2 = view1.to(device), view2.to(device)
                z_i, z_j = model(view1, view2)
                loss = criterion(z_i, z_j)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f"Época [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Atualizar scheduler
        if scheduler is not None:
            scheduler.step()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.encoder.state_dict(), save_path)
            print(f"✅ Modelo salvo em '{save_path}'")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⚠️ Early stopping na época {epoch+1}")
                break

    return history


# ===== CLASSIFICADOR PARA FINE-TUNING =====
class BehaviorClassifier(nn.Module):
    """Classificador que usa o encoder pré-treinado"""
    def __init__(self, encoder, embedding_dim=128, num_classes=8):
        super(BehaviorClassifier, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.BatchNorm1d(64),  # Adicionar BatchNorm
            nn.ReLU(),
            nn.Dropout(0.5),  # Aumentar dropout de 0.3 para 0.5
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (batch, time_steps, channels) -> (batch, channels, time_steps)
        # Debug: print shape antes e depois
        # print(f"DEBUG: Input shape: {x.shape}")
        x = x.transpose(1, 2)
        # print(f"DEBUG: After transpose: {x.shape}")
        h = self.encoder(x)
        return self.classifier(h)

    def unfreeze_encoder(self):
        """Descongela encoder para fine-tuning completo"""
        for param in self.encoder.parameters():
            param.requires_grad = True


print("✅ Código SimCLR completo pronto para uso!")
