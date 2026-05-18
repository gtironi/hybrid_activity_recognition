"""Training loops: Stage 1 (class weights), Stage 2 (finetune), FixMatch + eval."""
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


def train_stage1(model_cls, train_dl, val_dl, num_classes, n_feats, device,
                 epochs=50, checkpoint_path="stage1.pth"):
    print(f"[Stage 1] {model_cls.__name__} | device: {device}")
    model = model_cls(num_classes, n_feats).to(device)

    all_labels = train_dl.dataset.labels.cpu().numpy()
    classes_unique = np.unique(all_labels)
    weights = compute_class_weight('balanced', classes=classes_unique, y=all_labels)
    weights_tensor = torch.FloatTensor(weights).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.3)

    best_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    patience = 25
    counter = 0

    for epoch in range(epochs):
        model.train()
        tr_loss, correct, total = 0.0, 0, 0
        for x_sig, x_feat, y in train_dl:
            x_sig, x_feat, y = x_sig.to(device), x_feat.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x_sig, x_feat)
            loss = criterion(out, y)
            if torch.isnan(loss):
                print("Loss NaN. Abortando.")
                return model
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * x_sig.size(0)
            _, pred = out.max(1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
        tr_loss /= len(train_dl.dataset)
        tr_acc = 100 * correct / total

        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for x_sig, x_feat, y in val_dl:
                x_sig, x_feat, y = x_sig.to(device), x_feat.to(device), y.to(device)
                out = model(x_sig, x_feat)
                v_loss += criterion(out, y).item() * x_sig.size(0)
                _, pred = out.max(1)
                v_correct += pred.eq(y).sum().item()
                v_total += y.size(0)
        v_loss /= len(val_dl.dataset)
        v_acc = 100 * v_correct / v_total
        print(f"Ep {epoch+1:03d}/{epochs} | Train {tr_loss:.4f}/{tr_acc:.2f}% | Val {v_loss:.4f}/{v_acc:.2f}%")
        scheduler.step(v_loss)

        if v_acc > best_acc:
            best_acc = v_acc
            best_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), checkpoint_path)
            counter = 0
        else:
            counter += 1
        if counter >= patience:
            print(f"Early stopping (sem melhoria por {patience} epocas).")
            break

    print(f"[Stage 1] Best Val Acc: {best_acc:.2f}%")
    model.load_state_dict(best_wts)
    return model


def train_stage2(model_cls, train_dl, val_dl, num_classes, n_feats, device,
                 epochs=20, checkpoint_path="stage1.pth", finetune_path="finetune.pth"):
    print(f"[Stage 2 - Finetune] {model_cls.__name__}")
    model = model_cls(num_classes, n_feats).to(device)
    if not os.path.exists(checkpoint_path):
        print(f"Stage 1 checkpoint nao encontrado: {checkpoint_path}")
        return None
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        tr_loss, correct, total = 0.0, 0, 0
        for x_sig, x_feat, y in train_dl:
            x_sig, x_feat, y = x_sig.to(device), x_feat.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x_sig, x_feat)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * x_sig.size(0)
            _, pred = out.max(1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
        tr_loss /= len(train_dl.dataset)
        tr_acc = 100 * correct / total

        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for x_sig, x_feat, y in val_dl:
                x_sig, x_feat, y = x_sig.to(device), x_feat.to(device), y.to(device)
                out = model(x_sig, x_feat)
                v_loss += criterion(out, y).item() * x_sig.size(0)
                _, pred = out.max(1)
                v_correct += pred.eq(y).sum().item()
                v_total += y.size(0)
        v_loss /= len(val_dl.dataset)
        v_acc = 100 * v_correct / v_total
        print(f"FT Ep {epoch+1:02d}/{epochs} | Train {tr_loss:.4f}/{tr_acc:.2f}% | Val {v_loss:.4f}/{v_acc:.2f}%")
        scheduler.step(v_loss)

        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), finetune_path)

    print(f"[Stage 2] Best Val Acc: {best_acc:.2f}%")
    return model


class SensorFixMatchAugment:
    def __init__(self, device):
        self.device = device

    def weak_aug(self, x_signal, x_features):
        B = x_signal.shape[0]
        noise = torch.randn_like(x_signal) * 0.05
        scale = torch.rand(B, 1, 1, device=self.device) * 0.2 + 0.9
        aug_signal = (x_signal * scale) + noise
        aug_features = x_features + torch.randn_like(x_features) * 0.02
        return aug_signal, aug_features

    def strong_aug(self, x_signal, x_features):
        x_s, x_f = self.weak_aug(x_signal, x_features)
        B, C, T = x_s.shape
        x_aug = x_s.clone()
        if np.random.rand() > 0.5 and T > 10:
            num_seg = np.random.randint(2, 5)
            seg_len = T // num_seg
            for i in range(B):
                perm = torch.randperm(num_seg)
                temp = [x_s[i, :, p * seg_len:(p + 1) * seg_len] for p in perm]
                shuffled = torch.cat(temp, dim=1)
                if shuffled.shape[1] < T:
                    pad = torch.zeros(C, T - shuffled.shape[1], device=self.device)
                    shuffled = torch.cat([shuffled, pad], dim=1)
                elif shuffled.shape[1] > T:
                    shuffled = shuffled[:, :T]
                x_aug[i] = shuffled
        else:
            mask_len = int(T * 0.3)
            for i in range(B):
                start = np.random.randint(0, T - mask_len)
                x_aug[i, :, start:start + mask_len] = 0.0
        f_aug = x_f + torch.randn_like(x_f) * 0.05
        return x_aug, f_aug


def train_fixmatch(model_cls, labeled_dl, unlabeled_dl, val_dl, num_classes, n_feats, device,
                   epochs=20, checkpoint_path="finetune.pth", save_path="fixmatch.pth",
                   threshold=0.7, lambda_u=1.0):
    print(f"[FixMatch] {model_cls.__name__} | threshold={threshold} lambda_u={lambda_u}")
    model = model_cls(num_classes, n_feats).to(device)
    if os.path.exists(checkpoint_path):
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print(f"Pesos carregados: {checkpoint_path}")
        except Exception as e:
            print(f"Aviso: nao foi possivel carregar ({e}). Comecando do zero.")
    else:
        print("Sem checkpoint. Comecando do zero.")

    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    augmenter = SensorFixMatchAugment(device)

    def cycle(it):
        while True:
            for x in it:
                yield x
    unlab_iter = cycle(unlabeled_dl)

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss, loss_s_acc, loss_u_acc, mask_acc, steps = 0, 0, 0, 0, 0
        for x_s_lab, x_f_lab, y_lab in labeled_dl:
            x_s_lab, x_f_lab, y_lab = x_s_lab.to(device), x_f_lab.to(device), y_lab.to(device)
            x_s_u, x_f_u = next(unlab_iter)
            x_s_u, x_f_u = x_s_u.to(device), x_f_u.to(device)
            bs = x_s_lab.size(0)
            x_s_u, x_f_u = x_s_u[:bs], x_f_u[:bs]

            x_s_lab_w, x_f_lab_w = augmenter.weak_aug(x_s_lab, x_f_lab)
            logits_lab = model(x_s_lab_w, x_f_lab_w)
            loss_s = F.cross_entropy(logits_lab, y_lab)

            with torch.no_grad():
                x_s_w, x_f_w = augmenter.weak_aug(x_s_u, x_f_u)
                logits_w = model(x_s_w, x_f_w)
                probs = torch.softmax(logits_w, dim=1)
                max_probs, pseudo = torch.max(probs, dim=1)
                mask = max_probs.ge(threshold).float()
                mask_acc += mask.mean().item()

            x_s_st, x_f_st = augmenter.strong_aug(x_s_u, x_f_u)
            logits_st = model(x_s_st, x_f_st)
            loss_u_un = F.cross_entropy(logits_st, pseudo, reduction='none')
            loss_u = (loss_u_un * mask).mean()

            loss = loss_s + lambda_u * loss_u
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            loss_s_acc += loss_s.item()
            loss_u_acc += loss_u.item()
            steps += 1

        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for x_s, x_f, y in val_dl:
                x_s, x_f, y = x_s.to(device), x_f.to(device), y.to(device)
                out = model(x_s, x_f)
                _, pred = out.max(1)
                v_correct += pred.eq(y).sum().item()
                v_total += y.size(0)
        v_acc = 100 * v_correct / v_total
        print(f"FM Ep {epoch+1:02d}/{epochs} | Loss {total_loss/steps:.4f} "
              f"(Sup {loss_s_acc/steps:.3f} Unsup {loss_u_acc/steps:.3f}) | "
              f"MaskRate {mask_acc/steps:.1%} | Val Acc {v_acc:.2f}%")
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), save_path)
    print(f"[FixMatch] Best Val Acc: {best_acc:.2f}%")
    return model


def evaluate(model_cls, test_dl, num_classes, n_feats, device, checkpoint_path, class_names):
    print(f"[Eval] {model_cls.__name__} | {checkpoint_path}")
    model = model_cls(num_classes, n_feats).to(device)
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint nao encontrado: {checkpoint_path}")
        return None
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss, y_true, y_pred = 0.0, [], []
    with torch.no_grad():
        for x_sig, x_feat, y in test_dl:
            x_sig, x_feat, y = x_sig.to(device), x_feat.to(device), y.to(device)
            out = model(x_sig, x_feat)
            test_loss += criterion(out, y).item() * x_sig.size(0)
            _, pred = out.max(1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1w = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    labels = list(range(num_classes))
    report = classification_report(y_true, y_pred, labels=labels,
                                   target_names=list(class_names), zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    print(f"Acc: {acc:.4f} | F1-macro: {f1m:.4f} | F1-weighted: {f1w:.4f}")
    return {
        "accuracy": acc, "f1_macro": f1m, "f1_weighted": f1w,
        "loss": test_loss / len(test_dl.dataset),
        "y_true": np.array(y_true), "y_pred": np.array(y_pred),
        "classification_report": report, "confusion_matrix": cm,
    }
