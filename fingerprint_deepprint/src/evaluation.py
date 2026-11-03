"""
Biometric Evaluation Module.

Implements all metrics reported in the DeepPrint research paper:
- Genuine and Impostor similarity score distributions
- Receiver Operating Characteristic (ROC) curve
- False Accept Rate (FAR), False Reject Rate (FRR), Equal Error Rate (EER)
- True Accept Rate at specified FAR thresholds (TAR @ FAR=0.1%, TAR @ FAR=0.01%)
- Rank-1 and Cumulative Match Characteristic (CMC) identification accuracy
- Hit Rate at Penetration Rate 1% (HR @ PR = 1.0%)
"""

import os
from typing import Dict, List, Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import brentq


class Evaluator:
    """
    Computes rigorous biometric verification and identification metrics
    following international biometrics standards and DeepPrint protocols.
    """

    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def compute_scores(
        self,
        probe_embeddings: np.ndarray,
        probe_labels: List[str],
        gallery_embeddings: np.ndarray,
        gallery_labels: List[str],
        max_impostors: int = 100000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes genuine and impostor similarity scores.
        Genuine: Same finger comparison.
        Impostor: Different finger comparison.
        """
        # Similarity matrix: (M, N)
        similarity_matrix = np.dot(probe_embeddings, gallery_embeddings.T)

        probe_labels = np.array(probe_labels)
        gallery_labels = np.array(gallery_labels)

        # Boolean mask where probe_label == gallery_label
        match_mask = (probe_labels[:, None] == gallery_labels[None, :])

        genuine_scores = similarity_matrix[match_mask]
        impostor_scores = similarity_matrix[~match_mask]

        # Subsample impostors if array is exceedingly large for fast evaluation
        if len(impostor_scores) > max_impostors:
            np.random.seed(42)
            impostor_scores = np.random.choice(impostor_scores, size=max_impostors, replace=False)

        return genuine_scores, impostor_scores

    def compute_roc_and_eer(
        self,
        genuine_scores: np.ndarray,
        impostor_scores: np.ndarray,
        num_thresholds: int = 1000
    ) -> Dict:
        """
        Generates ROC curve (TAR vs FAR) and calculates EER (Equal Error Rate).
        """
        all_scores = np.concatenate([genuine_scores, impostor_scores])
        min_s = float(np.min(all_scores))
        max_s = float(np.max(all_scores))
        thresholds = np.linspace(min_s, max_s, num_thresholds)

        far_list = []
        frr_list = []
        tar_list = []

        n_gen = len(genuine_scores)
        n_imp = len(impostor_scores)

        for th in thresholds:
            # Impostors accepted (score >= th)
            fa = np.sum(impostor_scores >= th)
            far = fa / n_imp
            far_list.append(far)

            # Genuines rejected (score < th)
            fr = np.sum(genuine_scores < th)
            frr = fr / n_gen
            frr_list.append(frr)

            # Genuines accepted (score >= th)
            tar = 1.0 - frr
            tar_list.append(tar)

        far_arr = np.array(far_list)
        frr_arr = np.array(frr_list)
        tar_arr = np.array(tar_list)

        # Calculate EER where FAR == FRR
        # Find index where difference changes sign
        diff = far_arr - frr_arr
        idx = np.argmin(np.abs(diff))
        eer = (far_arr[idx] + frr_arr[idx]) / 2.0
        eer_threshold = thresholds[idx]

        return {
            "thresholds": thresholds,
            "far": far_arr,
            "frr": frr_arr,
            "tar": tar_arr,
            "eer": float(eer),
            "eer_threshold": float(eer_threshold)
        }

    def compute_tar_at_far(
        self,
        genuine_scores: np.ndarray,
        impostor_scores: np.ndarray,
        target_far: float = 0.001
    ) -> Tuple[float, float]:
        """
        Calculates TAR at a specified operational FAR (e.g., 0.1% or 0.01%).
        Returns:
            (tar_value, operating_threshold)
        """
        # Threshold is the (1 - target_far) percentile of impostor scores
        # because impostor_score >= threshold is a false accept
        sorted_imp = np.sort(impostor_scores)
        idx = int(np.floor((1.0 - target_far) * len(sorted_imp)))
        idx = min(idx, len(sorted_imp) - 1)
        threshold = sorted_imp[idx]

        tar = np.sum(genuine_scores >= threshold) / len(genuine_scores)
        return float(tar), float(threshold)

    def compute_identification_metrics(
        self,
        probe_embeddings: np.ndarray,
        probe_labels: List[str],
        gallery_embeddings: np.ndarray,
        gallery_labels: List[str],
        max_rank: int = 50
    ) -> Dict:
        """
        Computes 1:N Identification metrics:
        - Rank-1 Identification Accuracy
        - Cumulative Match Characteristic (CMC) up to max_rank
        - Hit Rate at Penetration Rate 1% (HR @ PR = 1.0%)
        """
        similarity_matrix = np.dot(probe_embeddings, gallery_embeddings.T)
        gallery_labels = np.array(gallery_labels)

        n_probes = len(probe_labels)
        n_gallery = len(gallery_labels)
        cmc_hits = np.zeros(max_rank)

        # Penetration rate: 1% of the gallery size
        pr_1_pct_k = max(1, int(np.ceil(0.01 * n_gallery)))
        pr_1_pct_hits = 0

        for i in range(n_probes):
            p_label = probe_labels[i]
            scores = similarity_matrix[i]
            # Rank gallery in descending score
            ranked_indices = np.argsort(scores)[::-1]
            ranked_labels = gallery_labels[ranked_indices]

            # Find rank of true label
            matches = np.where(ranked_labels == p_label)[0]
            if len(matches) > 0:
                first_match_rank = matches[0]  # 0-indexed rank
                if first_match_rank < max_rank:
                    cmc_hits[first_match_rank:] += 1
                if first_match_rank < pr_1_pct_k:
                    pr_1_pct_hits += 1

        cmc_curve = (cmc_hits / n_probes) * 100.0
        rank_1_acc = cmc_curve[0]
        hr_at_pr_1 = (pr_1_pct_hits / n_probes) * 100.0

        return {
            "rank_1": float(rank_1_acc),
            "cmc_curve": cmc_curve,
            "hr_at_pr_1": float(hr_at_pr_1),
            "pr_1_pct_k": pr_1_pct_k,
            "total_gallery": n_gallery,
            "total_probes": n_probes
        }

    def plot_score_distributions(
        self,
        genuine_scores: np.ndarray,
        impostor_scores: np.ndarray,
        filename: str = "score_distributions.png"
    ):
        """Plots genuine vs impostor cosine similarity score histograms."""
        plt.figure(figsize=(8, 5))
        plt.hist(impostor_scores, bins=60, density=True, alpha=0.6, color='red', label='Impostor Scores')
        plt.hist(genuine_scores, bins=60, density=True, alpha=0.6, color='green', label='Genuine Scores')
        plt.xlabel("Cosine Similarity Score")
        plt.ylabel("Probability Density")
        plt.title("DeepPrint-Inspired Genuine vs Impostor Score Distributions")
        plt.legend(loc="upper right")
        plt.grid(True, linestyle="--", alpha=0.5)

        save_path = os.path.join(self.results_dir, filename)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Score distribution plot saved to: {save_path}")

    def plot_roc_curve(
        self,
        roc_dict: Dict,
        tar_01: float,
        tar_001: float,
        filename: str = "roc_curve.png"
    ):
        """Plots semi-log ROC curve highlighting TAR at FAR=0.1% and FAR=0.01%."""
        plt.figure(figsize=(8, 6))
        plt.plot(roc_dict["far"], roc_dict["tar"] * 100.0, color='blue', lw=2, label="DeepPrint-Inspired CNN")
        plt.xscale("log")
        plt.xlim([1e-4, 1.0])
        plt.ylim([0, 105])
        plt.xlabel("False Accept Rate (FAR)")
        plt.ylabel("True Accept Rate (%) (TAR = 1 - FRR)")
        plt.title("Receiver Operating Characteristic (ROC) Curve")

        # Mark operational points
        plt.scatter([0.001], [tar_01 * 100.0], color='red', s=50, zorder=5,
                    label=f"TAR @ FAR=0.1%: {tar_01*100.0:.2f}%")
        plt.scatter([0.0001], [tar_001 * 100.0], color='orange', s=50, zorder=5,
                    label=f"TAR @ FAR=0.01%: {tar_001*100.0:.2f}%")

        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.legend(loc="lower right")

        save_path = os.path.join(self.results_dir, filename)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"ROC curve plot saved to: {save_path}")

    def plot_cmc_curve(
        self,
        cmc_curve: np.ndarray,
        filename: str = "cmc_curve.png"
    ):
        """Plots Cumulative Match Characteristic (CMC) curve."""
        plt.figure(figsize=(8, 5))
        ranks = np.arange(1, len(cmc_curve) + 1)
        plt.plot(ranks, cmc_curve, marker='o', markersize=3, color='purple', lw=2)
        plt.xlabel("Rank (k)")
        plt.ylabel("Identification Rate (%)")
        plt.title(f"Cumulative Match Characteristic (CMC) Curve (Rank-1: {cmc_curve[0]:.2f}%)")
        plt.ylim([min(cmc_curve[0] - 5, 0), 105])
        plt.grid(True, linestyle="--", alpha=0.5)

        save_path = os.path.join(self.results_dir, filename)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"CMC curve plot saved to: {save_path}")

    def plot_comparison_roc(
        self,
        roc_dict_all: Dict[str, Dict],
        filename: str = "comparison_roc.png"
    ):
        """
        Plots overlaid semi-log ROC curves comparing multiple datasets/conditions.
        """
        plt.figure(figsize=(9, 6))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        styles = ['-', '--', '-.', ':', '-', '--']

        for idx, (name, data) in enumerate(roc_dict_all.items()):
            c = colors[idx % len(colors)]
            ls = styles[idx % len(styles)]
            tar_01 = data.get("tar_01", 0.0) * 100.0
            label = f"{name} (TAR@0.1%FAR: {tar_01:.1f}%)"
            plt.plot(data["far"], data["tar"] * 100.0, color=c, linestyle=ls, lw=2.2, label=label)

        plt.xscale("log")
        plt.xlim([1e-4, 1.0])
        plt.ylim([0, 105])
        plt.xlabel("False Accept Rate (FAR)", fontsize=11, fontweight="bold")
        plt.ylabel("True Accept Rate (%)", fontsize=11, fontweight="bold")
        plt.title("DeepPrint Verification ROC Comparison Across Benchmarks", fontsize=12, fontweight="bold")
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.legend(loc="lower right", fontsize=9, framealpha=0.9)

        save_path = os.path.join(self.results_dir, filename)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Comparison ROC plot saved to: {save_path}")

    def plot_comparison_cmc(
        self,
        cmc_dict_all: Dict[str, np.ndarray],
        max_rank: int = 20,
        filename: str = "comparison_cmc.png"
    ):
        """
        Plots overlaid Cumulative Match Characteristic (CMC) curves for multiple datasets.
        """
        plt.figure(figsize=(9, 6))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        markers = ['o', 's', '^', 'D', 'v', 'p']

        ranks = np.arange(1, max_rank + 1)
        for idx, (name, curve) in enumerate(cmc_dict_all.items()):
            c = colors[idx % len(colors)]
            m = markers[idx % len(markers)]
            sub_curve = curve[:max_rank]
            label = f"{name} (Rank-1: {sub_curve[0]:.1f}%)"
            plt.plot(ranks, sub_curve, marker=m, markersize=4, color=c, lw=2.0, label=label)

        plt.xlabel("Rank (k)", fontsize=11, fontweight="bold")
        plt.ylabel("Identification Rate (%)", fontsize=11, fontweight="bold")
        plt.title("DeepPrint Identification CMC Comparison Across Benchmarks", fontsize=12, fontweight="bold")
        plt.ylim([0, 105])
        plt.xticks(ranks[::2])
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="lower right", fontsize=9, framealpha=0.9)

        save_path = os.path.join(self.results_dir, filename)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Comparison CMC plot saved to: {save_path}")

    def plot_alteration_bar_chart(
        self,
        alt_metrics: Dict[str, Dict],
        filename: str = "alteration_type_analysis.png"
    ):
        """
        Plots a grouped bar chart comparing Rank-1 and TAR@FAR=0.1% across alteration types.
        """
        types = list(alt_metrics.keys())
        rank1_vals = [alt_metrics[t]["rank_1"] for t in types]
        tar01_vals = [alt_metrics[t]["tar_01"] * 100.0 for t in types]

        x = np.arange(len(types))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))
        rects1 = ax.bar(x - width/2, rank1_vals, width, label='Rank-1 Accuracy (%)', color='#2b5c8f')
        rects2 = ax.bar(x + width/2, tar01_vals, width, label='TAR @ FAR=0.1% (%)', color='#e26d5c')

        ax.set_ylabel('Performance (%)', fontsize=11, fontweight='bold')
        ax.set_title('Robustness by Alteration Category (DeepPrint ResNet-50)', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(types, fontsize=10)
        ax.set_ylim([0, 115])
        ax.legend(loc='upper right')
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        for rect in rects1:
            h = rect.get_height()
            ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
        for rect in rects2:
            h = rect.get_height()
            ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

        save_path = os.path.join(self.results_dir, filename)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Alteration bar chart saved to: {save_path}")

    def plot_compression_fidelity(
        self,
        original_embs: np.ndarray,
        decompressed_embs: np.ndarray,
        filename: str = "compression_fidelity.png"
    ):
        """
        Plots reconstruction analysis comparing original 768-byte float32 vectors
        vs 200-byte min-max quantized vectors.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Absolute error distribution
        abs_err = np.abs(original_embs - decompressed_embs).flatten()
        axes[0].hist(abs_err, bins=60, color='#38b000', alpha=0.75, edgecolor='black', lw=0.5)
        axes[0].set_xlabel("Absolute Value Difference (|R - R_decomp|)", fontsize=10)
        axes[0].set_ylabel("Frequency", fontsize=10)
        axes[0].set_title(f"Reconstruction Error Distribution (MAE: {np.mean(abs_err):.6f})", fontsize=11, fontweight="bold")
        axes[0].grid(True, linestyle="--", alpha=0.5)

        # Correlation scatter plot for sample embedding
        sample_orig = original_embs[0]
        sample_decomp = decompressed_embs[0]
        axes[1].scatter(sample_orig, sample_decomp, color='#0077b6', alpha=0.6, s=20)
        min_v, max_v = min(sample_orig), max(sample_orig)
        axes[1].plot([min_v, max_v], [min_v, max_v], 'r--', lw=1.5, label='Ideal Identity (y = x)')
        axes[1].set_xlabel("Original Float32 Representation (768 Bytes)", fontsize=10)
        axes[1].set_ylabel("Decompressed Representation (200 Bytes)", fontsize=10)
        corr = np.dot(sample_orig, sample_decomp)
        axes[1].set_title(f"200-Byte Template Fidelity (Cosine Corr: {corr:.6f})", fontsize=11, fontweight="bold")
        axes[1].legend(loc="upper left")
        axes[1].grid(True, linestyle="--", alpha=0.5)

        save_path = os.path.join(self.results_dir, filename)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Compression fidelity plot saved to: {save_path}")

