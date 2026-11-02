"""
DeepPrint-Inspired Fingerprint Recognition Package.
Provides modules for preprocessing, dataset handling, architecture,
training, fixed-length embedding extraction, matching, and evaluation.
"""

from .preprocessing import FingerprintPreprocessor
from .dataset import SOCOFingDataset, create_subject_independent_splits
from .model import DeepPrintResNet, DeepPrintMobileNet
from .embeddings import EmbeddingExtractor, compress_embedding, decompress_embedding
from .matching import FingerprintMatcher
from .evaluation import Evaluator

__all__ = [
    "FingerprintPreprocessor",
    "SOCOFingDataset",
    "create_subject_independent_splits",
    "DeepPrintResNet",
    "DeepPrintMobileNet",
    "EmbeddingExtractor",
    "compress_embedding",
    "decompress_embedding",
    "FingerprintMatcher",
    "Evaluator",
]

