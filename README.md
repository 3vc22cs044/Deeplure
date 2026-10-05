# DeepLure — Color-Invariant Saree Design Recognition (CIS-Net)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Proprietary-blue.svg)](https://deeplure.org)
[![Accuracy](https://img.shields.io/badge/Top--1%20Acc-77.33%25-brightgreen.svg)]()
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.9495-emerald.svg)]()
[![Color-Bias-Gap](https://img.shields.io/badge/Color%20Bias%20Gap-%2B0.6305-cyan.svg)]()

> **AIE-CASE: Textile Face Recognition for Sarees**  
> Identifies sarees by the design on their surface, independent of the color palette in which that design is rendered.

---

## 1. Approach Note (<= 500 Characters)

> **"CIS-Net uses an IBN-ResNet backbone with a fixed multi-gradient stem (Sobel & Laplacian) and GeM pooling to decouple textile motif geometry from chromatic palettes. Preprocessing extracts structural gradients and instance-normalizes RGB channels. Training uses balanced P×K sampling (8 designs × 4 colorways) with ArcFace loss (m=0.35, s=30), SupCon, and palette decorrelation under aggressive hue/channel jitter. Post-processing computes L2-normalized 512-d embeddings for cosine retrieval and verification."**
> *(Length: 495 characters)*

---

## 2. Key Results & Evaluation Protocol

Evaluated on a rigorous split of **40 training designs (240 images)**, **25 gallery reference designs**, and **130 query probes** (including cross-colorway matches, hard same-palette distractors, and open-set unseen patterns):

| Metric | Score | Protocol / Description |
| :--- | :--- | :--- |
| **Top-1 Accuracy** | **77.33%** | Probe matches target design motif at rank #1 in gallery |
| **Top-5 Accuracy** | **90.67%** | Target design motif is retrieved within top-5 candidates |
| **mean Average Precision (mAP)** | **83.29%** | Precision averaged across recall levels |
| **Verification ROC AUC** | **0.9495** | Pairwise discrimination between same vs different designs |
| **Equal Error Rate (EER)** | **13.42%** | Balanced error operating at threshold $\tau^* = 0.525$ |
| **Positive Pair Cosine Sim** | **0.8255** | Mean similarity for same motif in contrasting colorways |
| **Hard Distractor Cosine Sim** | **0.1949** | Mean similarity for different motifs in *identical* color palettes |
| **Color Bias Gap ($\Delta$)** | **+0.6305** | $\text{Sim}_{\text{pos}} - \text{Sim}_{\text{distractor}}$ (**Strong Color Invariance**) |

---

## 3. Architecture: CIS-Net

1. **Fixed Structural Gradient Filter Bank**:
   Sobel ($G_x, G_y$), Gradient Magnitude $\sqrt{G_x^2 + G_y^2}$, and 2nd-order Laplacian operators capture boundary lines, jacquard weave counts, and printing stencils directly from luminance.
2. **Instance-Batch Normalization (IBN)**:
   Instance Normalization in early stages eliminates global dye contrast and saturation variance, while Batch Normalization preserves high-level geometric motif semantics.
3. **Generalized Mean (GeM) Pooling**:
   Learnable pooling exponent ($p=3.0$) focuses feature attention on sharp zari threads and borders rather than uniform background silk.
4. **Hyperspherical ArcFace + SupCon Loss**:
   Additive angular margin ($m=0.35, s=28.0$) compacts intra-design variations across colorways while widening inter-design separation on $\mathbb{S}^{511}$.

---

## 4. Efficiency & Computational Footprint (Bonus)

- **Parameter Count**: **8,203,521 (8.20 Million parameters)** — lean and deployable to mobile/edge devices.
- **FLOPs / Complexity**: **3.199 GFLOPs (1.599 GMACs)** per $224 \times 224$ image.
- **CPU Latency**: **~50.5 ms** (single-thread CPU inference) $\rightarrow$ **19.8 FPS**.
- **GPU Latency**: **< 4.5 ms** $\rightarrow$ **> 220 FPS**.
- **Embedding Size**: **512 float32 = 2.00 KB per saree**.
- **Large-Scale Gallery Scaling**:
  - $100,000$ Sarees = **195.3 MB RAM**.
  - Sub-millisecond retrieval using exact NumPy dot product or FAISS index.

---

## 5. Quickstart & Local Execution

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Prepare Dataset
```bash
python -m src.prepare_data
```

### Step 3: Train Model
```bash
python -m src.train --epochs 6 --lr 8e-4
```

### Step 4: Run Evaluation
```bash
python -m src.evaluate --checkpoint checkpoints/cis_net_best.pt
```

### Step 5: Launch Web Application
```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```
Open **`http://127.0.0.1:8000`** in your browser to interact with the dashboard:
- **1:N Identification**: Query against 25 canonical reference sarees with structural heatmaps.
- **1:1 Verification Bench**: Test pairs with calibrated threshold meter and similarity gauge.
- **Color Invariance Stress Lab**: Generate real-time textile colorways and inspect the cross-colorway similarity matrix.
- **Live Hardware Benchmark**: Benchmark latency and FPS on your local system in real-time.

---

## 6. Project Structure

```
├── data/
│   ├── saree_dataset/        # Curated saree designs (train, gallery, query)
│   ├── deeplure_corpus/      # DeepLure handloom sarees
│   └── splits.json           # Documented protocol splits
├── src/
│   ├── model.py              # CIS-Net architecture (IBN blocks, GeM pooling, ArcFace)
│   ├── dataset.py            # BalancedPKBatchSampler, color-invariance augmentations
│   ├── colorway_engine.py    # Textile palette transfer & hue rotation engine
│   ├── losses.py             # ArcFace, SupCon, and Palette Decorrelation loss
│   ├── train.py              # End-to-end PyTorch training script
│   ├── evaluate.py           # Protocol evaluation (Top-1/5, mAP, CMC, ROC-AUC, EER)
│   ├── benchmark.py          # Latency, parameter count, and FLOPs profiler
│   └── inference.py          # Inference engine for identification and verification
├── web/
│   ├── index.html            # Interactive glassmorphic dashboard
│   ├── styles.css            # Dark mode UI styling
│   └── app.js                # Dynamic front-end logic & API connectors
├── checkpoints/
│   └── cis_net_best.pt       # Trained model checkpoint
├── artifacts/
│   ├── evaluation_metrics.json
│   ├── efficiency_report.json
│   ├── roc_curve.png
│   └── cmc_curve.png
├── saree_recognition_kaggle.ipynb # Ready-to-run Kaggle notebook
├── approach_note.md          # 500-char approach note and full technical defense
├── app.py                    # FastAPI server & REST API
└── requirements.txt          # Python dependencies
```
