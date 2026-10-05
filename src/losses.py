"""
Loss functions and metric learning objectives for Color-Invariant Saree Recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss (Khosla et al., NeurIPS 2020).
    Explicitly groups all colorway instances of the same saree design together in cosine space,
    while separating different design motifs.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        features: [B, D] L2-normalized embeddings
        labels: [B] class IDs
        """
        device = features.device
        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Compute cosine similarity matrix
        sim_matrix = torch.matmul(features, features.T) / self.temperature

        # For numerical stability
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()

        # Mask out self-contrast
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Compute log-probs
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-7)

        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-7)

        loss = -mean_log_prob_pos.mean()
        return loss


class PaletteDecorrelationLoss(nn.Module):
    """
    Palette Decorrelation Regularizer.
    Penalizes covariance between the learned 512-d motif embedding and the global chromaticity
    vector (Mean RGB / Chroma values), explicitly discouraging color shortcuts.
    """
    def __init__(self):
        super().__init__()

    def forward(self, embeddings: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        """
        embeddings: [B, D]
        images: [B, 3, H, W]
        """
        # Extract mean chromaticity per image: [B, 3]
        chroma = images.mean(dim=[-2, -1])  # [B, 3]
        chroma_centered = chroma - chroma.mean(dim=0, keepdim=True)
        emb_centered = embeddings - embeddings.mean(dim=0, keepdim=True)

        # Compute cross-covariance matrix: [D, 3]
        cov = torch.matmul(emb_centered.T, chroma_centered) / (embeddings.size(0) - 1 + 1e-6)
        return torch.mean(cov ** 2)
