# DeepPrint-Inspired Fingerprint Recognition Using a Pretrained CNN

An implementation of a fixed-length fingerprint representation learning and recognition framework inspired by the IEEE TPAMI (2021) paper:

> **"Learning a Fixed-Length Fingerprint Representation"**  
> *Joshua J. Engelsma, Kai Cao, and Anil K. Jain*  
> IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), Vol. 43, No. 6, June 2021.

---

## 1. Project Overview

Prevailing Automated Fingerprint Identification Systems (AFIS) rely on variable-length minutiae point sets ($m = (x, y, \theta)$). While highly discriminative, minutiae sets suffer from:
1. **Computational Complexity**: Requires expensive graph matching / point-set alignment ($O(n^3)$ or $O(n^2)$), causing identification across large galleries to take seconds or minutes.
2. **Poor Quality Degradation**: Ridge distortion, skin dryness, or partial impressions lead to spurious or missing minutiae.
3. **Template Privacy Challenges**: Variable-length graphs cannot be directly matched in the encrypted domain using homomorphic encryption or fast spatial indexing (e.g., Product Quantization).

**DeepPrint** addresses this by learning a **fixed-length 192-dimensional compact representation** that compresses down to **200 bytes** and enables $O(d)$ constant-time authentication via simple cosine similarity ($s = R_p^T R_g$).

This project implements a practical, reproducible adaptation of DeepPrint:
- Uses a **pretrained CNN backbone** (ResNet-50 / MobileNetV3-Large with ImageNet weights) fine-tuned on fingerprint images.
- Applies **Contrast Limited Adaptive Histogram Equalization (CLAHE)** and orientation-preserving geometric preprocessing.
- Enforces strict **subject-independent splitting (Open-Set evaluation)** to prevent data leakage.
- Projects features to a **fixed-length 192-dimensional unit-norm vector**.
- Implements the exact **200-byte min-max template compression** from Section 3.5 of the original paper.
- Computes all paper evaluation metrics: **Rank-1, Hit Rate @ Penetration Rate 1%, TAR @ FAR=0.1% / 0.01%, EER, Genuine/Impostor score distributions, and ROC curves**.

---

## 2. Directory Structure

```text
fingerprint_deepprint/
│
├── dataset/                     # Local symlinks or cached dataset splits
├── notebooks/
│   └── fingerprint_recognition.ipynb  # End-to-end interactive Jupyter notebook
│
├── src/
│   ├── __init__.py              # Package init
│   ├── dataset.py               # Parsing SOCOFing & FVC, subject-independent splitting
│   ├── preprocessing.py         # CLAHE, resizing, 3-channel grayscale formatting
│   ├── model.py                 # ResNet-50 & MobileNetV3 architectures with 192-d projection
│   ├── train.py                 # Two-stage fine-tuning, loss, scheduler, checkpointing
│   ├── embeddings.py            # Feature extraction, L2 normalization, 200-byte compression
│   ├── matching.py              # Cosine similarity 1:1 verification & 1:N identification
│   └── evaluation.py            # ROC, EER, TAR@FAR, Rank-1, CMC, HR@PR=1% metrics & plots
│
├── checkpoints/                 # Saved model weights (best_model.pth)
├── results/                     # Generated ROC, CMC, and score distribution figures
├── main.py                      # Complete command-line driver script
├── requirements.txt             # Python dependencies
└── README.md                    # Detailed documentation and research analysis
```

---

## 3. Environment & Setup

Activate your existing virtual environment or create a new one:

```bash
# Activate existing virtual environment
source ~/venv/bin/activate

# Or create a new environment if needed:
# python3 -m venv ~/venv
# source ~/venv/bin/activate

# Navigate to project directory
cd fingerprint_deepprint

# Install dependencies
pip install -r requirements.txt
```

---

## 4. Running the System

### Option A: Complete Pipeline via Command Line (`main.py`)

Run two-stage training and evaluation on the local SOCOFing dataset:

```bash
# Run with ResNet-50 backbone (subsampled to 50 subjects for rapid execution on CPU)
python main.py --data_dir ../data/archive/SOCOFing --model_arch resnet50 --max_subjects 50 --epochs_s1 2 --epochs_s2 3

# Run with lightweight MobileNetV3-Large backbone
python main.py --data_dir ../data/archive/SOCOFing --model_arch mobilenet_v3 --max_subjects 100 --epochs_s1 3 --epochs_s2 5

# Evaluate directly from a saved checkpoint without re-training
python main.py --data_dir ../data/archive/SOCOFing --eval_only --checkpoint checkpoints/best_model.pth
```

### Option B: Interactive Jupyter Notebook

Launch Jupyter Notebook:

```bash
jupyter notebook notebooks/fingerprint_recognition.ipynb
```

Follow the numbered cells sequentially to visualize:
1. Raw fingerprint impressions (Pristine vs Altered vs FVC).
2. CLAHE enhanced ridge patterns.
3. Training loss and validation accuracy curves.
4. Genuine vs Impostor cosine similarity histograms.
5. Semi-log Receiver Operating Characteristic (ROC) curve.
6. Cumulative Match Characteristic (CMC) curve.

---

## 5. Architectural Comparison: DeepPrint vs Modified Implementation

| Component / Property | Original DeepPrint (IEEE TPAMI 2021) | Our Modified Implementation |
|---|---|---|
| **Backbone Network** | Inception-v4 Stem + Custom Dual Inception Branches | Pretrained ResNet-50 (or MobileNetV3-Large) |
| **Pretrained Weights** | None (Trained from scratch on 455k prints) | Pretrained on ImageNet-1K (`DEFAULT` weights) |
| **Alignment Module** | End-to-end Spatial Transformer Network (STN) | CLAHE + Canonical Spatial Centering & Scaling |
| **Texture Branch** | Inception Modules A, B, C $\to$ GAP $\to$ 96-dim vector | ResNet-50 Residual Blocks $\to$ GAP $\to$ Linear Projection |
| **Minutiae Branch** | Shared parameters with 6-channel minutiae heatmap | Omitted (Minutiae ground-truth maps unavailable) |
| **Supervision / Loss** | Multi-task: Cross-Entropy + Center Loss + Minutiae MSE | Two-stage Cross-Entropy fine-tuning + L2 embedding |
| **Embedding Size** | 192 dimensions ($96\text{ texture} + 96\text{ minutiae}$) | 192 dimensions (L2-normalized unit vector) |
| **Template Size** | 768 bytes uncompressed $\to$ **200 bytes** compressed | 768 bytes uncompressed $\to$ **200 bytes** compressed |
| **Compression Method** | Min-max 8-bit quantization + 2 float32 scalars | Exact Section 3.5 min-max quantization reproduction |
| **Comparison Metric** | Cosine similarity $s = R_p^T R_g$ ($O(d)$ complexity) | Cosine similarity $s = R_p^T R_g$ ($O(d)$ complexity) |
| **Training Dataset** | 455k longitudinal rolled prints (38,291 fingers, private) | SOCOFing (600 subjects, 6,000 fingers, Real + Altered) |
| **Evaluation Protocol**| NIST SD4, NIST SD14, FVC2004 DB1 A, 1.1M background | Held-out SOCOFing subjects (Open-Set) & FVC2004 DB1_B |

---

## 6. Data Leakage Prevention (Subject-Independent Split)

In biometric verification and identification, the real-world operational requirement is **Open-Set Generalization**: matching impressions from fingers that were **never seen during model training**.

1. **Subject Isolation**: Subjects 1 to 420 (70%) are used exclusively for training. All fingers and impressions of these subjects are isolated to the training set.
2. **Validation Isolation**: Subjects 421 to 510 (15%) monitor validation loss and early stopping.
3. **Testing Isolation**: Subjects 511 to 600 (15%) are held out strictly for final evaluation.
4. **Gallery vs Probe Construction**:
   - For each held-out finger, the single pristine impression in `Real/` is enrolled as the **Gallery**.
   - The corresponding altered impressions in `Altered-Easy/` act as unknown **Probes**.
   - This models the real-world AFIS scenario: searching distorted/altered query prints against an enrolled database.

---

## 7. Mathematical Formulations of Benchmark Metrics

### A. Cosine Similarity Matching
$$\forall R_p, R_g \in \mathbb{S}^{191}, \quad s(R_p, R_g) = R_p^T R_g = \sum_{i=1}^{192} R_p[i] \cdot R_g[i]$$

### B. Template Compression (Section 3.5)
$$R'_i = \text{round}\left( 255 \times \frac{R_i - \min(R)}{\max(R) - \min(R)} \right) \in [0, 255]$$
$$\text{Storage} = 192 \text{ bytes (uint8 payload)} + 8 \text{ bytes (min, max float32)} = 200 \text{ bytes}$$

### C. False Accept Rate (FAR) and False Reject Rate (FRR)
$$\text{FAR}(\tau) = \frac{|\{s \in \mathcal{S}_{\text{imp}} : s \ge \tau\}|}{|\mathcal{S}_{\text{imp}}|}, \quad \text{FRR}(\tau) = \frac{|\{s \in \mathcal{S}_{\text{gen}} : s < \tau\}|}{|\mathcal{S}_{\text{gen}}|}$$
$$\text{TAR}(\tau) = 1 - \text{FRR}(\tau)$$

### D. Equal Error Rate (EER)
$$\tau^* = \arg\min_{\tau} |\text{FAR}(\tau) - \text{FRR}(\tau)|, \quad \text{EER} = \frac{\text{FAR}(\tau^*) + \text{FRR}(\tau^*)}{2}$$

### E. Hit Rate at Penetration Rate $p$ ($HR @ PR=p$)
For gallery size $N$, candidate retrieval threshold is $k = \lceil p \cdot N \rceil$. For $PR = 1.0\%$, the top $1\%$ most similar enrolled candidates are returned. $HR$ is the percentage of probes whose genuine enrolled template is within that top $1\%$ candidate list.
