"""
Comprehensive Evaluation Protocol for Color-Invariant Saree Recognition.
Covers:
1. Identification: Top-1, Top-5, mAP, and Cumulative Match Characteristic (CMC) curve.
2. Verification: Pairwise Cosine Similarity, ROC curve, AUC, Equal Error Rate (EER), and Optimal Threshold.
3. Color Invariance Diagnostic: Quantifies the gap between cross-colorway matches and same-palette distractors.
"""

import os
import json
import time
from typing import Dict, List, Tuple
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from src.model import CISNet


class SareeEvaluator:
    def __init__(self, model_checkpoint: str, device: str = "auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load model
        checkpoint = torch.load(model_checkpoint, map_location=self.device)
        self.embedding_dim = checkpoint.get("embedding_dim", 512)
        self.model = CISNet(embedding_dim=self.embedding_dim).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
        ])

    @torch.no_grad()
    def extract_embedding(self, image_path: str) -> np.ndarray:
        """Extracts 512-d L2-normalized embedding from image file."""
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            tensor = self.transform(im).unsqueeze(0).to(self.device)
            emb = self.model(tensor)
            return emb.cpu().squeeze(0).numpy()

    @torch.no_grad()
    def extract_batch_embeddings(self, image_paths: List[str]) -> np.ndarray:
        """Extracts embeddings for a list of image paths."""
        tensors = []
        for p in image_paths:
            with Image.open(p) as im:
                im = im.convert("RGB")
                tensors.append(self.transform(im))
        batch = torch.stack(tensors, dim=0).to(self.device)
        embs = self.model(batch)
        return embs.cpu().numpy()

    def run_full_evaluation(self, splits_path: str = "data/splits.json", output_dir: str = "artifacts") -> Dict:
        """
        Executes identification and verification evaluation across gallery and queries.
        Generates plots and structured metrics.
        """
        os.makedirs(output_dir, exist_ok=True)
        with open(splits_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        gallery_items = metadata["gallery_items"]
        query_items = metadata["query_items"]

        print(f"Loaded {len(gallery_items)} gallery designs and {len(query_items)} query items.")

        # 1. Extract Gallery Embeddings
        gal_paths = [g["path"] for g in gallery_items]
        gal_embs = self.extract_batch_embeddings(gal_paths)  # [N_gal, D]
        gal_designs = [g["design_id"] for g in gallery_items]

        # 2. Extract Query Embeddings
        q_paths = [q["path"] for q in query_items]
        q_embs = self.extract_batch_embeddings(q_paths)  # [N_query, D]

        # Cosine similarity matrix: [N_query, N_gal]
        similarity_matrix = np.dot(q_embs, gal_embs.T)

        # -------------------------------------------------------------
        # Identification Evaluation (Closed/Open set retrieval)
        # -------------------------------------------------------------
        # Only evaluate identification on queries that have a valid target in gallery
        top1_hits = 0
        top5_hits = 0
        ap_list = []
        cmc_counts = np.zeros(len(gal_items := gallery_items))

        cross_colorway_sims = []
        hard_distractor_sims = []

        eval_query_count = 0
        for i, q in enumerate(query_items):
            target_design = q["design_id"]
            q_type = q["query_type"]
            sims = similarity_matrix[i]
            sorted_indices = np.argsort(-sims)
            ranked_designs = [gal_designs[idx] for idx in sorted_indices]

            if q_type == "cross_colorway_match":
                eval_query_count += 1
                is_match = [d == target_design for d in ranked_designs]
                
                # Top-1
                if is_match[0]:
                    top1_hits += 1
                # Top-5
                if any(is_match[:5]):
                    top5_hits += 1

                # CMC rank count
                rank_pos = is_match.index(True) if True in is_match else len(gallery_items)
                if rank_pos < len(cmc_counts):
                    cmc_counts[rank_pos:] += 1

                # Average Precision
                num_relevant = sum(is_match)
                if num_relevant > 0:
                    precisions = []
                    rel_count = 0
                    for r, matched in enumerate(is_match):
                        if matched:
                            rel_count += 1
                            precisions.append(rel_count / (r + 1))
                    ap_list.append(np.mean(precisions))
                else:
                    ap_list.append(0.0)

                # Collect cross-colorway positive similarity
                correct_gal_idx = gal_designs.index(target_design) if target_design in gal_designs else -1
                if correct_gal_idx >= 0:
                    cross_colorway_sims.append(sims[correct_gal_idx])

            elif q_type == "same_palette_diff_motif":
                confuser_gal_id = q.get("intended_confuser_gallery_id")
                # Measure similarity to the confuser gallery item that shares its palette
                for g_idx, g in enumerate(gallery_items):
                    if g["gallery_id"] == confuser_gal_id:
                        hard_distractor_sims.append(sims[g_idx])
                        break

        top1_acc = (top1_hits / max(1, eval_query_count)) * 100.0
        top5_acc = (top5_hits / max(1, eval_query_count)) * 100.0
        mAP = (np.mean(ap_list) if ap_list else 0.0) * 100.0
        cmc_curve = ((cmc_counts / max(1, eval_query_count)) * 100.0).tolist()

        # -------------------------------------------------------------
        # Verification Evaluation (Pairwise ROC / AUC / EER)
        # -------------------------------------------------------------
        pair_scores = []
        pair_labels = []

        # Positive pairs: cross-colorway queries paired with their matching gallery item
        for i, q in enumerate(query_items):
            if q["query_type"] == "cross_colorway_match":
                target_design = q["design_id"]
                for g_idx, g in enumerate(gallery_items):
                    if g["design_id"] == target_design:
                        pair_scores.append(float(similarity_matrix[i, g_idx]))
                        pair_labels.append(1)

        # Negative pairs: hard distractors + random negative pairings
        for i, q in enumerate(query_items):
            target_design = q["design_id"]
            for g_idx, g in enumerate(gallery_items):
                if g["design_id"] != target_design:
                    # Sample negatives to keep balanced
                    if q["query_type"] == "same_palette_diff_motif" or np.random.rand() < 0.15:
                        pair_scores.append(float(similarity_matrix[i, g_idx]))
                        pair_labels.append(0)

        pair_scores = np.array(pair_scores)
        pair_labels = np.array(pair_labels)

        fpr, tpr, thresholds = roc_curve(pair_labels, pair_scores)
        roc_auc = auc(fpr, tpr)

        # Equal Error Rate (EER) calculation
        fnr = 1.0 - tpr
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
        optimal_threshold = float(thresholds[eer_idx])

        # Color Invariance Diagnostic Metrics
        mean_cross_cw = float(np.mean(cross_colorway_sims)) if cross_colorway_sims else 0.0
        mean_hard_dist = float(np.mean(hard_distractor_sims)) if hard_distractor_sims else 0.0
        color_bias_gap = mean_cross_cw - mean_hard_dist

        # -------------------------------------------------------------
        # Generate & Save Visualization Plots
        # -------------------------------------------------------------
        # 1. ROC Curve
        plt.figure(figsize=(6, 5), dpi=150)
        plt.plot(fpr, tpr, color="#2563eb", lw=2.5, label=f"CIS-Net ROC (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], color="#94a3b8", linestyle="--", lw=1.5, label="Random Guess")
        plt.scatter([fpr[eer_idx]], [tpr[eer_idx]], color="#ef4444", s=50, zorder=5, label=f"EER = {eer*100:.1f}% (thr={optimal_threshold:.2f})")
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.xlabel("False Positive Rate (FAR)", fontsize=11, fontweight="bold")
        plt.ylabel("True Positive Rate (1 - FRR)", fontsize=11, fontweight="bold")
        plt.title("Saree Design Verification ROC Curve", fontsize=12, fontweight="bold", pad=12)
        plt.legend(loc="lower right", frameon=True)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        roc_path = os.path.join(output_dir, "roc_curve.png")
        plt.savefig(roc_path)
        plt.close()

        # 2. CMC Curve
        plt.figure(figsize=(6, 5), dpi=150)
        ranks = list(range(1, min(16, len(cmc_curve) + 1)))
        plt.plot(ranks, cmc_curve[:len(ranks)], marker="o", color="#10b981", lw=2.5, label="CIS-Net CMC")
        plt.xlabel("Rank (k)", fontsize=11, fontweight="bold")
        plt.ylabel("Identification Rate (%)", fontsize=11, fontweight="bold")
        plt.title("Cumulative Match Characteristic (CMC) Curve", fontsize=12, fontweight="bold", pad=12)
        plt.ylim([0, 105])
        plt.xticks(ranks)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        cmc_path = os.path.join(output_dir, "cmc_curve.png")
        plt.savefig(cmc_path)
        plt.close()

        # 3. Compile Report
        results = {
            "identification": {
                "top1_accuracy_percent": round(top1_acc, 2),
                "top5_accuracy_percent": round(top5_acc, 2),
                "mean_average_precision_percent": round(mAP, 2),
                "cmc_top1": round(cmc_curve[0], 2) if cmc_curve else 0.0,
                "cmc_top5": round(cmc_curve[4], 2) if len(cmc_curve) > 4 else 0.0,
                "evaluated_query_count": eval_query_count,
                "gallery_size": len(gallery_items),
            },
            "verification": {
                "auc": round(float(roc_auc), 4),
                "eer_percent": round(eer * 100.0, 2),
                "optimal_threshold": round(optimal_threshold, 4),
                "total_pairs_evaluated": len(pair_scores),
                "positive_pairs": int(np.sum(pair_labels == 1)),
                "negative_pairs": int(np.sum(pair_labels == 0)),
            },
            "color_invariance_diagnostic": {
                "mean_same_motif_diff_colorway_sim": round(mean_cross_cw, 4),
                "mean_diff_motif_same_palette_sim": round(mean_hard_dist, 4),
                "color_bias_gap": round(color_bias_gap, 4),
                "assessment": "Demonstrates strong color invariance: motif geometry governs similarity rather than dye color."
                if color_bias_gap > 0.25 else "Moderate color invariance."
            },
            "plots": {
                "roc_curve": roc_path,
                "cmc_curve": cmc_path,
            }
        }

        # Save metrics json
        metrics_file = os.path.join(output_dir, "evaluation_metrics.json")
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print("\n" + "=" * 60)
        print("           EVALUATION RESULTS SUMMARY")
        print("=" * 60)
        print(f"Identification:")
        print(f"  • Top-1 Accuracy:     {top1_acc:.2f}%")
        print(f"  • Top-5 Accuracy:     {top5_acc:.2f}%")
        print(f"  • Mean Average Prec:  {mAP:.2f}%")
        print(f"\nVerification:")
        print(f"  • ROC AUC:            {roc_auc:.4f}")
        print(f"  • Equal Error Rate:   {eer * 100.0:.2f}%")
        print(f"  • Optimal Threshold:  {optimal_threshold:.4f}")
        print(f"\nColor Invariance Diagnostic:")
        print(f"  • Same Motif, Diff Palette:  {mean_cross_cw:.4f}")
        print(f"  • Diff Motif, Same Palette:  {mean_hard_dist:.4f}")
        print(f"  • Color Bias Gap:            +{color_bias_gap:.4f} (Positive & High = High Invariance)")
        print("=" * 60)

        return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/cis_net_best.pt")
    args = parser.parse_args()

    evaluator = SareeEvaluator(model_checkpoint=args.checkpoint)
    evaluator.run_full_evaluation()
