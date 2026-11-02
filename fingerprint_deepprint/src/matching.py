"""
Fingerprint Matching Module.

Implements 1:1 Verification and 1:N Identification using cosine similarity,
candidate retrieval, and rank-based search over fixed-length representations.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np


class FingerprintMatcher:
    """
    Handles fast cosine-similarity comparison for biometric verification and identification.
    """

    def __init__(self, gallery_embeddings: np.ndarray, gallery_labels: List[str]):
        """
        Args:
            gallery_embeddings: Array of shape (N, 192) unit-normalized.
            gallery_labels: List of length N with unique identity strings.
        """
        assert gallery_embeddings.shape[0] == len(gallery_labels)
        self.gallery_embeddings = gallery_embeddings  # (N, d)
        self.gallery_labels = np.array(gallery_labels)
        self.num_gallery = len(gallery_labels)

        # Precompute label to index lookup
        self.label_to_indices = {}
        for idx, lbl in enumerate(gallery_labels):
            self.label_to_indices.setdefault(lbl, []).append(idx)

    def verify_1_to_1(self, probe_embedding: np.ndarray, claimed_label: str) -> float:
        """
        1:1 Verification: Calculates cosine similarity score against enrolled gallery template.
        Score s = R_p^T * R_g in [-1, 1].
        """
        if claimed_label not in self.label_to_indices:
            raise ValueError(f"Claimed label '{claimed_label}' not enrolled in gallery.")

        target_indices = self.label_to_indices[claimed_label]
        target_embs = self.gallery_embeddings[target_indices]
        # Maximum score among enrolled impressions of the claimed identity
        scores = np.dot(target_embs, probe_embedding)
        return float(np.max(scores))

    def identify_1_to_N(
        self,
        probe_embedding: np.ndarray,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        1:N Identification: Compares probe against all gallery templates.
        Returns top_k (identity, similarity_score) pairs in descending score order.
        """
        scores = np.dot(self.gallery_embeddings, probe_embedding)
        sorted_indices = np.argsort(scores)[::-1][:top_k]

        results = [
            (self.gallery_labels[idx], float(scores[idx]))
            for idx in sorted_indices
        ]
        return results

    def batch_identify(
        self,
        probe_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Computes the complete similarity score matrix between all probes and all gallery templates.
        Args:
            probe_embeddings: Array of shape (M, 192)
        Returns:
            Similarity matrix S of shape (M, N) where S[i, j] = probe[i] * gallery[j]
        """
        return np.dot(probe_embeddings, self.gallery_embeddings.T)

