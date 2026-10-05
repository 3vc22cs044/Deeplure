"""
End-to-End PyTorch Training Pipeline for Color-Invariant Saree Recognition.
Trains CIS-Net with ArcFace + SupCon + Palette Decorrelation Loss.
"""

import os
import json
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.model import CISNet, ArcFaceHead
from src.dataset import SareePatternDataset, BalancedPKBatchSampler
from src.losses import SupConLoss, PaletteDecorrelationLoss


def train_cisnet(
    data_dir: str = "data/saree_dataset/train",
    checkpoint_dir: str = "checkpoints",
    epochs: int = 6,
    p_classes: int = 6,
    k_instances: int = 2,
    lr: float = 8e-4,
    embedding_dim: int = 512,
    device_str: str = "auto"
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"Starting training on device: {device}")

    # Dataset & Sampler
    dataset = SareePatternDataset(root_dir=data_dir, is_train=True, img_size=224)
    num_classes = len(dataset.classes)
    print(f"Loaded {len(dataset)} training images across {num_classes} design classes.")

    sampler = BalancedPKBatchSampler(
        class_indices=dataset.class_indices,
        p_classes=p_classes,
        k_instances=k_instances,
        iterations_per_epoch=10
    )
    dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)

    # Models
    model = CISNet(embedding_dim=embedding_dim).to(device)
    arcface = ArcFaceHead(embedding_dim=embedding_dim, num_classes=num_classes, margin=0.35, scale=28.0).to(device)

    # Losses
    ce_loss_fn = nn.CrossEntropyLoss()
    supcon_loss_fn = SupConLoss(temperature=0.08)
    decorr_loss_fn = PaletteDecorrelationLoss()

    # Optimizer & LR Scheduler
    params = list(model.parameters()) + list(arcface.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_loss = float("inf")
    history = []

    print("-" * 75)
    print(f"{'Epoch':<8} {'Time':<8} {'Total Loss':<12} {'ArcFace':<10} {'SupCon':<10} {'Decorr':<10} {'LR':<8}")
    print("-" * 75)

    for epoch in range(1, epochs + 1):
        model.train()
        arcface.train()

        total_loss_acc = 0.0
        arcface_acc = 0.0
        supcon_acc = 0.0
        decorr_acc = 0.0
        num_batches = 0
        t0 = time.time()

        for imgs, labels, _ in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass: extract L2-normalized embeddings
            embeddings = model(imgs)

            # 1. ArcFace Angular Margin Loss
            logits = arcface(embeddings, labels)
            l_arcface = ce_loss_fn(logits, labels)

            # 2. Supervised Contrastive Loss across colorways
            l_supcon = supcon_loss_fn(embeddings, labels)

            # 3. Palette Decorrelation Regularizer
            l_decorr = decorr_loss_fn(embeddings, imgs)

            # Joint Objective
            loss = l_arcface + 0.4 * l_supcon + 0.1 * l_decorr
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            optimizer.step()

            total_loss_acc += loss.item()
            arcface_acc += l_arcface.item()
            supcon_acc += l_supcon.item()
            decorr_acc += l_decorr.item()
            num_batches += 1

        scheduler.step()
        elapsed = time.time() - t0

        avg_loss = total_loss_acc / max(1, num_batches)
        avg_arc = arcface_acc / max(1, num_batches)
        avg_sup = supcon_acc / max(1, num_batches)
        avg_dec = decorr_acc / max(1, num_batches)
        current_lr = scheduler.get_last_lr()[0]

        print(f"{epoch:<8} {elapsed:<8.1f}s {avg_loss:<12.4f} {avg_arc:<10.4f} {avg_sup:<10.4f} {avg_dec:<10.4f} {current_lr:<8.2e}")

        history.append({
            "epoch": epoch,
            "total_loss": avg_loss,
            "arcface_loss": avg_arc,
            "supcon_loss": avg_sup,
            "decorr_loss": avg_dec,
            "lr": current_lr,
            "time_sec": elapsed
        })

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, "cis_net_best.pt")
        if avg_loss < best_loss or epoch == epochs:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "arcface_state_dict": arcface.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "embedding_dim": embedding_dim,
                "classes": dataset.classes,
                "loss": avg_loss,
            }, checkpoint_path)

    # Save training history
    with open(os.path.join(checkpoint_dir, "train_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("-" * 75)
    print(f"Training completed successfully! Best model saved to: {checkpoint_path}")
    return checkpoint_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CIS-Net for Color-Invariant Saree Recognition")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    args = parser.parse_args()

    train_cisnet(epochs=args.epochs, lr=args.lr)
