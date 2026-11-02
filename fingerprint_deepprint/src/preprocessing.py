"""
Fingerprint Image Preprocessing and Augmentation Module.

Implements domain-aware fingerprint enhancement, contrast adjustment (CLAHE),
and channel formatting compatible with ImageNet-pretrained CNN backbones.
"""

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image


class CLAHETransform:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to enhance fingerprint ridge-valley clarity.
    """
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img: Image.Image) -> Image.Image:
        # Convert PIL to grayscale numpy array
        np_img = np.array(img.convert('L'))
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size
        )
        enhanced = clahe.apply(np_img)
        return Image.fromarray(enhanced)


class FingerprintPreprocessor:
    """
    Constructs PyTorch transform pipelines for fingerprint training, validation,
    and inference.

    Why 3 channels?
    ImageNet-pretrained backbones (e.g., ResNet, MobileNet) expect 3-channel inputs.
    Replicating the enhanced grayscale channel across 3 channels (RGB with R=G=B)
    preserves compatibility with pretrained first-layer convolutional filters
    without retraining the input layer from scratch.
    """

    def __init__(self, img_size: int = 224, use_clahe: bool = True):
        self.img_size = img_size
        self.use_clahe = use_clahe
        self.imagenet_mean = [0.485, 0.456, 0.406]
        self.imagenet_std = [0.229, 0.224, 0.225]

    def get_train_transforms(self) -> transforms.Compose:
        """
        Training transformations with orientation-preserving augmentations.
        Note: Random horizontal flipping is deliberately EXCLUDED because flipping
        a fingerprint reverses ridge flow direction (e.g., transforming a left loop
        into a right loop), destroying identity-critical topological characteristics.
        """
        transform_list = []
        if self.use_clahe:
            transform_list.append(CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8)))

        transform_list.extend([
            transforms.Resize((self.img_size, self.img_size)),
            # Subtle rotation preserving ridge-valley patterns (+/- 15 degrees)
            transforms.RandomRotation(degrees=15, fill=255),
            # Subtle affine translation (+/- 5%)
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), fill=255),
            # Grayscale to 3-channel
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.imagenet_mean, std=self.imagenet_std),
        ])
        return transforms.Compose(transform_list)

    def get_eval_transforms(self) -> transforms.Compose:
        """
        Deterministic transformations for validation, testing, and embedding extraction.
        """
        transform_list = []
        if self.use_clahe:
            transform_list.append(CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8)))

        transform_list.extend([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.imagenet_mean, std=self.imagenet_std),
        ])
        return transforms.Compose(transform_list)


def load_fingerprint_image(path: str) -> Image.Image:
    """
    Safely opens any fingerprint image (BMP, TIF, PNG, JPG)
    and converts it to grayscale PIL Image.
    """
    img = Image.open(path)
    if img.mode != 'L':
        img = img.convert('L')
    return img

