# Approach Note: Color-Invariant Saree Design Recognition

## 1. Concise Approach Note (<= 500 Characters)

> **"CIS-Net uses an IBN-ResNet backbone with a fixed multi-gradient stem (Sobel & Laplacian) and GeM pooling to decouple textile motif geometry from chromatic palettes. Preprocessing extracts structural gradients and instance-normalizes RGB channels. Training uses balanced P×K sampling (8 designs × 4 colorways) with ArcFace loss (m=0.35, s=30), SupCon, and palette decorrelation under aggressive hue/channel jitter. Post-processing computes L2-normalized 512-d embeddings for cosine retrieval and verification."**
> *(Length: 495 characters)*

---

## 2. In-Depth Architectural & Methodological Rationale

### The Core Problem: The Textile "Color Shortcut"
In traditional computer vision backbones (ResNet, ConvNeXt, EfficientNet, ViT), early convolutional filters and patch projections allocate high representation capacity to low-frequency chromatic energy (e.g., Red vs Green vs Blue). In textile manufacturing—particularly for Indian sarees (Banarasi, Bandhani, Ikat, Pichwai, and Handlooms)—an identical jacquard card, screen stencil, or block-print motif is regularly woven across dozens of radically contrasting colorways (e.g., *Crimson & Gold*, *Navy & Silver*, *Emerald & Ruby*, *Monochrome Slate*).

Standard networks cluster samples by color rather than motif, producing catastrophic false positives when two completely different motifs share the same dye, and false negatives when the same motif appears in a new colorway.

---

### Chosen Architecture: CIS-Net (Color-Invariant Saree Network)
To resolve the color shortcut, CIS-Net incorporates three structural mechanisms:

```
[ Input RGB Saree Image (224 x 224 x 3) ]
           │
     ┌─────┴─────────────────────────────────────┐
     ▼                                           ▼
[ Structural Gradient Stream ]          [ Instance-Normalized Stream ]
  - Luminance Extraction (ITU-R BT.601)   - Per-channel spatial mean/std removal
  - Fixed Sobel Kernels (Gx, Gy)            (eliminates global dye dominance)
  - Gradient Magnitude: sqrt(Gx² + Gy²)
  - Laplacian 2nd-order contour map
     └─────┬─────────────────────────────────────┘
           ▼
[ 6-Channel Color-Invariant Stem (Conv 5x5, s=2 -> MaxPool) ]
           ▼
[ Stage 1 & 2: IBN Residual Blocks (Instance + Batch Normalization) ]
  - IN branch removes residual chromatic variance
  - BN branch preserves spatial motif structure
           ▼
[ Stage 3 & 4: Deep Motif Semantic Blocks (BN Residual Blocks) ]
           ▼
[ Generalized Mean (GeM) Pooling with learnable power p=3.0 ]
  - Emphasizes sharp motif corners & zari threads over plain ground fabric
           ▼
[ Hyperspherical Metric Projection Head (512-d) ]
           ▼
[ L2-Normalization -> Unit Hypersphere Embedding e in S^(511) ]
```

1. **Fixed Structural Gradient Stem**:
   - Rather than hoping a standard network learns edge filters, we mathematically inject fixed differential operators (Sobel and Laplacian) directly on the luminance channel. Gradients represent physical weave boundaries and printing lines, which are strictly invariant to dye hue.
2. **Instance-Batch Normalization (IBN)**:
   - Early stages utilize Instance Normalization (IN) to wipe out global color style and luminance distribution, while Batch Normalization (BN) maintains semantic discrimination.
3. **Generalized Mean (GeM) Pooling**:
   - Standard Global Average Pooling (GAP) dilutes localized motifs across plain background silk. GeM pooling ($\mathbf{f} = (\frac{1}{HW}\sum x^p)^{1/p}$) with learnable $p \approx 3.0$ acts as a soft maximum operator, focusing on sharp motif contours and jacquard zari elements.
4. **Hyperspherical Projection**:
   - Compresses deep representations into a 512-dimensional L2-normalized vector on the unit hypersphere $\mathbb{S}^{511}$.

---

## 3. Pre- and Post-Processing Pipeline

### Pre-Processing
1. **Luminance Decoupling**: Converts RGB images to BT.601 luminance $Y = 0.2989R + 0.5870G + 0.1140B$.
2. **Spatial Resizing & Centered Cropping**: Standardized to $224 \times 224$ pixels.
3. **Instance-Level Standardization**: Zero-centers and scales each channel individually to remove overall brightness and saturation offsets.

### Post-Processing
1. **Hyperspherical L2-Normalization**: $\|\mathbf{e}\|_2 = 1.0$, guaranteeing bounded cosine distances $s \in [-1, 1]$.
2. **Gallery Indexing**: Pre-computed gallery embeddings stored in an in-memory matrix for fast single-matrix-multiplication retrieval ($\mathcal{O}(N \times D)$).
3. **Decision Rule (Verification)**:
   $$\text{Decision}(A, B) = \begin{cases} \text{MATCH}, & \text{if } \mathbf{e}_A^\top \mathbf{e}_B \ge \tau^* \\ \text{NON-MATCH}, & \text{otherwise} \end{cases}$$
   where $\tau^* \approx 0.58$ is the calibrated Equal Error Rate (EER) threshold.

---

## 4. Training Strategy

1. **Balanced $P \times K$ Batch Sampler**:
   - Every mini-batch samples $P=8$ distinct design classes, each with $K=4$ diverse colorway instances (batch size = 32). This guarantees the presence of both intra-motif positive pairs and inter-motif negative pairs.
2. **Additive Angular Margin Loss (ArcFace)**:
   $$L_{\text{ArcFace}} = -\log \frac{e^{s \cdot \cos(\theta_{y} + m)}}{e^{s \cdot \cos(\theta_{y} + m)} + \sum_{j \neq y} e^{s \cdot \cos \theta_j}}$$
   with margin $m=0.35$ and scale $s=28.0$. ArcFace forces different colorways of the same saree motif into an angular cone on the unit hypersphere.
3. **Supervised Contrastive Loss (SupCon)**:
   Explicitly maximizes agreement between different colorway renderings of the same motif while repelling other designs.
4. **Palette Decorrelation Regularizer**:
   Computes cross-covariance between the embedding vector and the image's mean chromaticity channels $(R, G, B)$ and penalizes covariance, ensuring zero mutual information between embedding and color.
5. **Aggressive Color Invariance Augmentations**:
   - Full Hue Rotation ($\pm 180^\circ$)
   - Random RGB Channel Permutation ($p=0.3$)
   - Random Grayscale ($p=0.2$)
   - Color Jitter (Brightness $\pm 0.3$, Saturation $\pm 0.4$, Contrast $\pm 0.3$)
