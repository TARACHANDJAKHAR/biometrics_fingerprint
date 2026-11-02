"""
Fixed-Length Fingerprint Representation and Template Compression Module.

Extracts 192-dimensional L2-normalized deep representations and implements the exact
200-byte min-max quantization compression method defined in Section 3.5 of the DeepPrint paper.
"""

import struct
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def compress_embedding(embedding: np.ndarray) -> bytes:
    """
    Implements Section 3.5 'Template Compression' from the DeepPrint paper:
    1. Quantizes 192 float32 values (768 bytes uncompressed) to 8-bit unsigned integers [0, 255]
       using min-max normalization:
           R' = round( 255 * (R - min(R)) / (max(R) - min(R)) )
    2. Appends min(R) and max(R) as 32-bit floats (4 bytes each = 8 bytes metadata).
    Total compressed footprint: 192 * 1 byte + 8 bytes = EXACTLY 200 BYTES.
    """
    assert embedding.ndim == 1 and embedding.shape[0] == 192, "Expected 192-dim vector"
    min_val = float(np.min(embedding))
    max_val = float(np.max(embedding))
    val_range = max_val - min_val

    if val_range < 1e-12:
        quantized = np.zeros(192, dtype=np.uint8)
    else:
        norm = (embedding - min_val) / val_range
        quantized = np.clip(np.round(norm * 255.0), 0, 255).astype(np.uint8)

    # 192 bytes payload + 8 bytes floats (min_val, max_val)
    compressed_bytes = quantized.tobytes() + struct.pack("<ff", min_val, max_val)
    assert len(compressed_bytes) == 200, f"Expected 200 bytes, got {len(compressed_bytes)}"
    return compressed_bytes


def decompress_embedding(compressed_bytes: bytes) -> np.ndarray:
    """
    Reverses 200-byte min-max quantization to restore the 192-dim float representation.
    Re-applies L2 normalization.
    """
    assert len(compressed_bytes) == 200, "Compressed payload must be exactly 200 bytes"
    quantized = np.frombuffer(compressed_bytes[:192], dtype=np.uint8).astype(np.float32)
    min_val, max_val = struct.unpack("<ff", compressed_bytes[192:])

    decompressed = (quantized / 255.0) * (max_val - min_val) + min_val
    norm = np.linalg.norm(decompressed)
    if norm > 1e-12:
        decompressed = decompressed / norm
    return decompressed


class EmbeddingExtractor:
    """
    Extracts fixed-length representations for datasets using trained PyTorch models.
    """

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def extract_single(self, tensor_image: torch.Tensor) -> np.ndarray:
        """
        Extracts 192-dim normalized feature vector for a single image tensor of shape (3, H, W).
        """
        if tensor_image.ndim == 3:
            tensor_image = tensor_image.unsqueeze(0)
        tensor_image = tensor_image.to(self.device)
        emb = self.model.extract_features(tensor_image)
        return emb.cpu().numpy()[0]

    @torch.no_grad()
    def extract_dataset(
        self,
        loader: DataLoader,
        show_progress: bool = True
    ) -> Tuple[np.ndarray, List[str], List[Dict]]:
        """
        Extracts representations for all items in a DataLoader.

        Returns:
            embeddings: Array of shape (N, 192)
            finger_ids: List of N unique finger IDs
            records: Metadata dictionaries if available
        """
        embeddings_list = []
        finger_ids_list = []

        iterator = tqdm(loader, desc="Extracting Embeddings") if show_progress else loader
        for images, _, fids in iterator:
            images = images.to(self.device)
            embs = self.model.extract_features(images)
            embeddings_list.append(embs.cpu().numpy())
            finger_ids_list.extend(fids)

        all_embeddings = np.concatenate(embeddings_list, axis=0)
        return all_embeddings, finger_ids_list

