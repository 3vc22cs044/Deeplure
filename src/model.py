"""
CIS-Net: Color-Invariant Saree Network for Textile Pattern Recognition.
Incorporates:
1. Fixed & Learnable Structural Gradient Filter Bank (Sobel & Laplacian)
2. Instance-Batch Normalization (IBN) for color style decoupling
3. Generalized Mean (GeM) Pooling with learnable exponent
4. Hyperspherical Metric Embedding Head with ArcFace Loss
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientStructureExtractor(nn.Module):
    """
    Fixed differential filter bank capturing structural motif contours,
    edges, and weaves independent of chromatic tone.
    Outputs: [Sobel_X, Sobel_Y, Gradient_Magnitude, Laplacian]
    """
    def __init__(self):
        super().__init__()
        # Sobel X
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0],
                                [-2.0, 0.0, 2.0],
                                [-1.0, 0.0, 1.0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        # Sobel Y
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0],
                                [ 0.0,  0.0,  0.0],
                                [ 1.0,  2.0,  1.0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        # Laplacian
        laplacian = torch.tensor([[ 0.0,  1.0,  0.0],
                                  [ 1.0, -4.0,  1.0],
                                  [ 0.0,  1.0,  0.0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        self.register_buffer("laplacian", laplacian)

    def forward(self, gray: torch.Tensor) -> torch.Tensor:
        """
        Input: gray tensor [B, 1, H, W] in [0, 1]
        Output: structural feature maps [B, 3, H, W]
        """
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
        lap = F.conv2d(gray, self.laplacian, padding=1)
        return torch.cat([mag, lap, gray], dim=1)


class IBNConv(nn.Module):
    """
    Instance-Batch Normalization (IBN-a):
    Splits channels into two halves:
    - First half uses Instance Normalization (removes global color/contrast variance)
    - Second half uses Batch Normalization (preserves discriminative motif identity)
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        split_c = out_channels // 2
        self.split_c = split_c
        self.in_norm = nn.InstanceNorm2d(split_c, affine=True)
        self.bn_norm = nn.BatchNorm2d(out_channels - split_c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x_in = self.in_norm(x[:, :self.split_c, :, :])
        x_bn = self.bn_norm(x[:, self.split_c:, :, :])
        out = torch.cat([x_in, x_bn], dim=1)
        return self.act(out)


class ResidualIBNBlock(nn.Module):
    """Residual Block with IBN normalization in early convs."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, use_ibn: bool = True):
        super().__init__()
        self.use_ibn = use_ibn
        self.stride = stride

        if use_ibn:
            self.conv1 = IBNConv(in_channels, out_channels, stride=stride)
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True)
            )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.act(out + res)
        return out


class GeMPooling(nn.Module):
    """
    Generalized Mean Pooling (GeM) with learnable exponent p.
    f = ( (1 / |X|) * sum(x^p) )^(1/p)
    Emphasizes salient jacquard and border motif edges over uniform fabric background.
    """
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p_clamped = self.p.clamp(min=1.0, max=10.0)
        x_clamped = x.clamp(min=self.eps)
        out = F.avg_pool2d(x_clamped.pow(p_clamped), (x.size(-2), x.size(-1))).pow(1.0 / p_clamped)
        return out.squeeze(-1).squeeze(-1)


class CISNet(nn.Module):
    """
    Color-Invariant Saree Network (CIS-Net).
    A lean, high-accuracy network tailored for textile motif recognition across arbitrary colorways.
    """
    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Structural Edge & Gradient Extractor
        self.gradient_extractor = GradientStructureExtractor()

        # Learnable Color-Invariant Stem
        # Input: 3 channels structural (mag, lap, gray) + 3 channels instance-normalized RGB = 6 channels
        self.stem = nn.Sequential(
            nn.Conv2d(6, 48, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Stage 1: 48 -> 64 (uses IBN for color style invariant features)
        self.stage1 = nn.Sequential(
            ResidualIBNBlock(48, 64, stride=1, use_ibn=True),
            ResidualIBNBlock(64, 64, stride=1, use_ibn=True)
        )

        # Stage 2: 64 -> 128 (uses IBN)
        self.stage2 = nn.Sequential(
            ResidualIBNBlock(64, 128, stride=2, use_ibn=True),
            ResidualIBNBlock(128, 128, stride=1, use_ibn=True)
        )

        # Stage 3: 128 -> 256 (standard BN for high-level semantic motif layout)
        self.stage3 = nn.Sequential(
            ResidualIBNBlock(128, 256, stride=2, use_ibn=False),
            ResidualIBNBlock(256, 256, stride=1, use_ibn=False)
        )

        # Stage 4: 256 -> 384 (deep motif features)
        self.stage4 = nn.Sequential(
            ResidualIBNBlock(256, 384, stride=2, use_ibn=False),
            ResidualIBNBlock(384, 384, stride=1, use_ibn=False)
        )

        # Salient Motif Pooling
        self.gem_pool = GeMPooling(p=3.0)

        # Metric Projection Head (maps to unit hypersphere)
        self.head = nn.Sequential(
            nn.Linear(384, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
            nn.SiLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim)
        )

    def extract_structure_inputs(self, x_rgb: torch.Tensor) -> torch.Tensor:
        """
        Converts RGB [B, 3, H, W] to 6-channel representation:
        - 3 channels: fixed structural gradients [gradient magnitude, laplacian, luminance]
        - 3 channels: instance-normalized RGB (strips mean chromatic bias)
        """
        # Convert RGB to luminance [B, 1, H, W] using ITU-R BT.601 coefficients
        gray = 0.2989 * x_rgb[:, 0:1, :, :] + 0.5870 * x_rgb[:, 1:2, :, :] + 0.1140 * x_rgb[:, 2:3, :, :]
        struct_feats = self.gradient_extractor(gray)

        # Instance-normalized RGB
        in_rgb = F.instance_norm(x_rgb, eps=1e-5)

        return torch.cat([struct_feats, in_rgb], dim=1)

    def forward(self, x: torch.Tensor, return_embedding_only: bool = True) -> torch.Tensor:
        """
        Forward pass.
        Returns L2-normalized 512-dimensional embedding on unit hypersphere.
        """
        inputs_6ch = self.extract_structure_inputs(x)
        feat = self.stem(inputs_6ch)
        feat = self.stage1(feat)
        feat = self.stage2(feat)
        feat = self.stage3(feat)
        feat = self.stage4(feat)

        pooled = self.gem_pool(feat)
        proj = self.head(pooled)

        # L2-normalization
        embedding = F.normalize(proj, p=2, dim=1)
        return embedding


class ArcFaceHead(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss (Deng et al., CVPR 2019).
    Enforces geodesic angular margin on the unit hypersphere:
    L = -log( e^(s * cos(theta_y + m)) / (e^(s * cos(theta_y + m)) + sum_{j != y} e^(s * cos(theta_j))) )
    """
    def __init__(self, embedding_dim: int, num_classes: int, scale: float = 32.0, margin: float = 0.35):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        embeddings: [B, D] L2-normalized
        labels: [B] class indices
        """
        # Normalize weights to unit hypersphere
        norm_weights = F.normalize(self.weight, p=2, dim=1)
        # Cosine similarity: [B, C]
        cosine = F.linear(embeddings, norm_weights).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2)).clamp(min=1e-7)

        # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        return output
