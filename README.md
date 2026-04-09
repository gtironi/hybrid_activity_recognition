# DeepCalf-SSL: Accelerometer-Based Behavior Analysis

-----

**Participants:**

  - João Gabriel Machado - FGV (EMAp)
  - Gustavo Tironi - FGV (EMAp)

**Subject:**

- Deep Learning (Master) - Dário

-----

This repository contains a PyTorch implementation for classifying pre-weaned calf behaviors using high-frequency accelerometer data. It leverages the [**AcTBeCalf dataset**](https://zenodo.org/records/13259482) and implements **Semi-Supervised Learning (SSL)** techniques—specifically **FixMatch**—to improve classification performance on rare behaviors by utilizing large-scale unlabeled sensor data.

## Key Features:

  - **Data Processing:** Optimized pipelines for handling large-scale parquet files using Polars and PyArrow.
  - **Feature Engineering:** Hybrid approach using raw signals (225 datapoints) + Top 75 statistical features extracted via **TSFEL**.
  - **Modeling:** A custom Hybrid **CNN-LSTM** architecture tailored for time-series, outperforming standard ResNet-1D baselines.
  - **Methodology:** Subject-independent splitting, Time-Windowing (3s), and Semi-Supervised training loops.

-----

# Repository Structure

## 1\. Data Engineering & Pipeline

Scripts responsible for transforming raw CSVs into optimized Parquet files and generating the "Windowed" datasets used in the final model.

  * **`parquet_converter.py`**: Converts the massive raw CSV datasets into Parquet format, significantly reducing storage size and improving I/O speed for training.
  * **`window_creator_labeled.py` & `window_creator_unlabeled.py`**: The core preprocessing engine. These scripts:
    1.  Slice raw data into **3-second windows** (75 datapoints at 25Hz).
    2.  Extract hundreds of statistical features using **TSFEL**.
    3.  Select the **Top 75 features** (based on Feature Importance analysis).
    4.  Save the result as `WindowedCalf.parquet`.
  * **`window_test.py`**: Quality control script to verify data integrity, array shapes, and class distribution in the generated parquets.

## 2\. Exploratory Analysis

  * **`analise_exploratoria_rotulados.ipynb`**: Initial EDA. Analyzes class imbalance (e.g., "Lying" dominates with \~48%, while "Fall" is \<0.1%) and visualizes accelerometer signal patterns.

## 3\. Experimental History (Chronological)

Scripts representing the evolution of our approach, from baselines to advanced SSL.

  * **`Neural_Network.ipynb`**: *Early Experiment.* A simple Dense Neural Network. Failed to capture temporal dependencies, resulting in poor accuracy (\~30%) and majority-class bias.
  * **`Random_forest.ipynb`**: *Early Experiment.* Random Forest applied to raw data (no TSFEL). Showed high instability (Test 68% vs Validation 13%), indicating severe overfitting to specific subjects.
  * **`modelos_comparacao.ipynb`**: *Temporal Analysis.* Benchmark of 12 architecture variations (LSTM, CNN1D, CNN2D, Hybrid).
      * **Key Insight:** "Forward-looking" models performed poorly (\~20%). "Backward-looking" (causal) models performed best (\~58%), proving behavior classification relies on past context.
  * **`actbecalf-dl.ipynb`**: *Transformer & Baseline.* Implementation of **TimeMAE** (Masked Autoencoders) vs. **ResNet-1D Baseline**.
      * Result: Transformer (54%) failed to beat the CNN Baseline (56%) due to the low entropy of the calf movement data.
  * **`simCLR.ipynb` & `simclr_comparison.ipynb`**: *Contrastive Learning.* First successful attempt at SSL using SimCLR.
      * Result: Achieved \~82% accuracy, but was limited to a simplified subset of **8 classes** (vs the full 19).

## 4\. Final Model (The Solution)

  * **`actbecalf-windowed.ipynb`**: **State-of-the-Art Model.**
      * Implements the **Hybrid CNN-LSTM** architecture.
      * **Inputs:** Concatenates Raw Signal (via CNN-LSTM) + Statistical Features (via MLP).
      * **Training:** Uses **FixMatch** (Semi-Supervised Learning) to leverage unlabelled data.
      * **Result:** Achieved **83-84% Accuracy** on the full 19-class taxonomy, surpassing the Random Forest benchmark (79%) and all previous Deep Learning attempts.

-----

# Model Evolution & Methodology

Our approach evolved through three distinct phases, driven by error analysis and the specific challenges of the AcTBeCalf dataset.

## Phase 1: The Baseline Struggle

We started with standard Deep Learning architectures (CNN1D, LSTM).

  * **Challenge:** The models struggled to generalize across subjects.
  * **Experiment:** We tested specific temporal directions in `modelos_comparacao.ipynb`.
  * **Finding:** Pure LSTMs were too slow. Pure CNNs lacked long-term context. "Looking ahead" (Forward window) destroyed accuracy.
  * **Result:** Accuracy plateaued at **\~56%**.

## Phase 2: The Self-Supervised Pivot (SimCLR & TimeMAE)

Hypothesizing that labeled data was insufficient (only 27 hours vs 2000+ hours available), we turned to SSL.

  * **TimeMAE:** Tried to reconstruct masked signals. Failed because calf data has low entropy (lots of lying down), making reconstruction "too easy" and uninformative.
  * **SimCLR:** Contrastive learning showed great promise (**\~82%**), but we were simplifying the problem to only 8 classes. We needed a solution for the full 19-behavior taxonomy.

## Phase 3: Hybrid Architecture + FixMatch (Final Solution)

To beat the strong Random Forest baseline (which uses statistical features from [TSFEL](https://tsfel.readthedocs.io/en/latest/)), we designed a **Hybrid Model**.

1. **Data Engineering:** We processed the data into **3-second windows** (75hz of x, y and z sensors).
2. **Feature Fusion:**
- **Branch A (Deep Learning):** A **CNN-1D** extracts spatial features, fed into a **Bidirectional LSTM** to capture temporal context;
- **Branch B (Feature Engineering):** An MLP processes the **Top 75 TSFEL features** (Entropy, FFT, etc.);
- **Concatenation:** Both branches merge for the final classification.
3. **FixMatch Training:** We used the unlabelled data to enforce consistency. If the model predicts "Walking" for a window, it is forced to predict "Walking" for a slightly augmented version of that same window.

### Training Strategy: The Three-Stage Pipeline

To overcome the challenges of extreme class imbalance (e.g., "Lying" is >48% of the data, while "Fall" is <0.1%) and the scarcity of labeled data, we implemented a progressive training pipeline. Instead of training end-to-end in one go, we found that guiding the model through distinct learning phases yielded significantly better convergence.

#### **Step 1: Supervised Learning with Class Balancing**
* **Goal:** Force the model to recognize rare and critical behaviors (like *Cough* or *Fall*) that would otherwise be ignored by the loss function. We noticed that, in some of the latter models, the model just learned to predict the same 1, 2 classes all the time, like prediciting *lying* every time, for each window.
* **Technique:** We used **Weighted Cross-Entropy Loss**, inversely proportional to class frequency.
* **Result:** The model learned feature diversity but struggled with overall accuracy due to the artificial penalty on common classes.
* **Accuracy:** **~76%**

#### **Step 2: Fine-Tuning (Re-balancing)**
* **Goal:** Re-calibrate the model's decision boundaries to reflect the **real-world probability distribution** of behaviors.
* **Technique:** We removed the class weights and fine-tuned the model with a lower Learning Rate (`1e-4`). This allows the classifier to maximize global accuracy without "forgetting" the rare features learned in Phase 1.
* **Result:** A significant boost in general performance.
* **Accuracy:** **~80%**

#### **Step 3: Semi-Supervised Learning (FixMatch)**
* **Goal:** Generalize the model's understanding by exposing it to the massive variability found in the **2000+ hours of unlabeled data**.
* **Technique:** We initialized the FixMatch loop using the weights from Phase 2 (Warm Start). The model generated pseudo-labels for unlabeled windows (Teacher) and learned to predict them consistently under strong augmentations (Student).
* **Result:** The model became robust to noise and sensor variations, achieving our state-of-the-art result.
* **Accuracy:** **~83.4%**

### Results Summary

| Model Architecture | Input Type | Strategy | Accuracy (Test) |
| :--- | :--- | :--- | :--- |
| **Dense NN** | Raw | Supervised | 28% (+-4%) |
| **ResNet-1D** | Raw | Supervised | 56% (+- 2%) |
| **CNN2D** | Raw | Supervised | 52% (+- 1%) |
| **LSTM** | Raw | Supervised | 54% (+- 1%) |
| **CNN1D** | Raw | Supervised | 55% (+- 1%) |
| **CNN-LSTM Initial** | Raw | Supervised | 58% (+- 1%) |
| **TimeMAE** | Raw | Self-Supervised | 54% (+- 2%) |
| **Random Forest** | Raw | Supervised | 13% to 68% (incosistent) |
| **Random Forest** | TSFEL Features | Supervised | 78.5% (+- 1.5%) |
| **SimCLR Initial** | Raw | Contrastive (8 Classes) | 64% (+- 3%) |
| **SimCLR** | Raw | Contrastive (8 Classes) | 82% (+- 1%) |
| **Hybrid CNN-LSTM** | Raw + TSFEL | Supervised | 80% (+- 2%) |
| **Hybrid CNN-LSTM (FixMatch)** | **Raw + TSFEL** | **FixMatch (SSL)** | **83.4% (+- 1%)** |

-----

## Usage

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Prepare Data:**
    - For the labeled data, run `parquet_converter.py` with `AcTBeCalf.csv` followed by `window_creator_labeled.py`.
    - For the unlabeled data, run `parquet_converter.py` with `Time_Adj_Raw_Data.csv` followed by `window_creator_unlabeled.py`.
3.  **Train:**
    Open `actbecalf-windowed.ipynb` and run the training loop.
