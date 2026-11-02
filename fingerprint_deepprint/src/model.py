"""
DeepPrint-Inspired Architecture Module.

Implements feature extraction using pretrained CNN backbones (ResNet-50 and MobileNetV3-Large),
fixed-length 192-dimensional embedding projection, L2 normalization, and classification heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional


class DeepPrintResNet(nn.Module):
    """
    DeepPrint-inspired fingerprint recognition architecture using ResNet-50 backbone.
    
    Dataflow:
    Input (B, 3, 224, 224) 
        -> ResNet-50 Backbone (conv1 through layer4)
        -> Global Average Pooling (B, 2048)
        -> Linear Projection Layer (2048 -> 192)
        -> Batch Normalization 1D
        -> L2 Normalization -> Fixed-Length Embedding R in R^192 (Unit Sphere)
        -> Identity Classification Head (Linear 192 -> num_classes) [Training only]
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 192,
        pretrained: bool = True,
        dropout_p: float = 0.2
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        # Load ResNet-50 with ImageNet pretrained weights
        if pretrained:
            weights = models.ResNet50_Weights.DEFAULT
            base_model = models.resnet50(weights=weights)
        else:
            base_model = models.resnet50(weights=None)

        # Retain convolutional stem and residual stages (layer1 to layer4)
        self.stem = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool
        )
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # 2048 is the output channel depth of ResNet-50 layer4
        backbone_dim = 2048

        # Fixed-length representation projection
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity()
        self.fc_embed = nn.Linear(backbone_dim, embedding_dim, bias=False)
        self.bn_embed = nn.BatchNorm1d(embedding_dim)

        # Classification head for training
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def freeze_backbone(self):
        """Freezes all backbone layers for initial stage training."""
        for param in self.stem.parameters():
            param.requires_grad = False
        for param in self.layer1.parameters():
            param.requires_grad = False
        for param in self.layer2.parameters():
            param.requires_grad = False
        for param in self.layer3.parameters():
            param.requires_grad = False
        for param in self.layer4.parameters():
            param.requires_grad = False

    def unfreeze_deep_layers(self):
        """Unfreezes higher-level layers (layer4) for fine-tuning."""
        for param in self.layer4.parameters():
            param.requires_grad = True

    def unfreeze_all(self):
        """Unfreezes all network parameters for end-to-end fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts L2-normalized fixed-length 192-dimensional embeddings.
        Returns:
            Tensor of shape (B, 192) with unit Euclidean norm.
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)

        x = self.dropout(x)
        x = self.fc_embed(x)
        x = self.bn_embed(x)
        embeddings = F.normalize(x, p=2, dim=1)
        return embeddings

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass during training.
        Returns:
            logits: Shape (B, num_classes)
            embeddings: Shape (B, 192) unit normalized
        """
        embeddings = self.extract_features(x)
        logits = self.classifier(embeddings)
        return logits, embeddings


class DeepPrintMobileNet(nn.Module):
    """
    Lightweight DeepPrint alternative using MobileNetV3-Large backbone.
    Ideal for fast CPU execution and resource-constrained environments.
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 192,
        pretrained: bool = True,
        dropout_p: float = 0.2
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        if pretrained:
            weights = models.MobileNet_V3_Large_Weights.DEFAULT
            base = models.mobilenet_v3_large(weights=weights)
        else:
            base = models.mobilenet_v3_large(weights=None)

        self.features = base.features
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # MobileNetV3-Large features output channel dimension is 960
        backbone_dim = 960

        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity()
        self.fc_embed = nn.Linear(backbone_dim, embedding_dim, bias=False)
        self.bn_embed = nn.BatchNorm1d(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def freeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc_embed(x)
        x = self.bn_embed(x)
        embeddings = F.normalize(x, p=2, dim=1)
        return embeddings

    def forward(self, x: torch.Tensor) -> tuple:
        embeddings = self.extract_features(x)
        logits = self.classifier(embeddings)
        return logits, embeddings

