# DeepPrint Biometric AFIS Dashboard (Frontend)

Interactive local dashboard for evaluating and comparing deep fixed-length fingerprint representations based on the IEEE TPAMI 2021 research paper:
> **"Learning a Fixed-Length Fingerprint Representation"**  
> *Joshua J. Engelsma, Kai Cao, Anil K. Jain (IEEE Transactions on Pattern Analysis and Machine Intelligence, Vol. 43, No. 6, June 2021)*

---

## Quick Start: How to Open & View the Dashboard

You can view the dashboard in any web browser without needing any backend server:

### Option 1: Direct File Opening (Fastest)
Simply open `index.html` in your browser:
```bash
xdg-open frontend/index.html
# Or on Windows/macOS: double-click frontend/index.html
```

### Option 2: Local HTTP Server (Recommended)
```bash
cd frontend
python3 -m http.server 8000
```
Then visit: **`http://localhost:8000`** in your browser.

---

## Dashboard Features

1. **Interactive Model Switcher**:
   - `[★ Paper Dual-Branch]`: Spatial Transformer Network (STN) + Dual-Branch (96-d Texture + 96-d Minutiae) with 6-Channel Minutiae Orientation Heatmap Guidance.
   - `[🔹 ResNet-50 Baseline]`: Standard CNN feature extraction with Global Average Pooling (GAP) and Linear Projection.
   - `[⚔️ Head-to-Head Comparison]`: Side-by-side performance deltas, comparative bar charts, and overlaid ROC curves.

2. **Benchmarked Conditions (Open-Set Evaluation)**:
   - **SOCOFing Altered-Easy**: Mild distortions (CR, Obl, Zcut).
   - **SOCOFing Altered-Medium**: Moderate distortions.
   - **SOCOFing Altered-Hard**: Severe scarring and core obliteration.
   - **SOCOFing Combined**: Unified cross-alteration test set.
   - **Section 3.5 200-Byte Template**: Min-max scalar quantization ($74\%$ storage reduction with $>0.9999$ correlation).

3. **Distortion Robustness Analysis**:
   - Central Rotation (CR)
   - Obliteration (Obl)
   - Z-Cut Scars (Zcut)

4. **Live Match Inspector**:
   - Real visual examples of Genuine and Impostor fingerprint pairs.
   - Side-by-side similarity scores and matching decisions ($\tau^* = 0.4500$).

5. **Section 3 Architecture & Mathematical Explainer**:
   - Details STN affine alignment ($\theta \in \mathbb{R}^{2\times 3}$).
   - Minutiae orientation heatmap formula ($H \in \mathbb{R}^{6 \times 64 \times 64}$).
   - Multi-task joint loss ($\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}}^{\text{tex}} + \mathcal{L}_{\text{CE}}^{\text{min}} + \lambda \mathcal{L}_{\text{MSE}}$).
   - Unit hypersphere projection ($R \in S^{191}$).

---

## Directory Contents

- `index.html`: Main standalone single-page dashboard.
- `test_results.json`: Complete JSON benchmark export for both models.
- `model_comparison_*.png`: Comparative head-to-head charts.
- `dual_branch/`: Individual ROC, CMC, score distributions, and distortion charts for the Paper Dual-Branch model.
- `baseline/`: Individual ROC, CMC, score distributions, and distortion charts for the ResNet-50 Baseline model.
- `samples/`: Rendered probe and gallery BMP/PNG pairs for the Live Match Inspector.
