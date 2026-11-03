"""
DeepPrint-Inspired Architecture Module.

Implements feature extraction using pretrained CNN backbones (ResNet-50 and MobileNetV3-Large),
fixed-length 192-dimensional embedding projection, L2 normalization, and classification heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional, Tuple


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


class SpatialTransformerNetwork(nn.Module):
    """
    Shallow Localization Network & Bilinear Grid Sampler (Table 3 in DeepPrint paper).
    Learns canonical fingerprint alignment end-to-end without reference points.
    """
    def __init__(self):
        super().__init__()
        # Localization CNN (Table 3 in paper)
        self.localization = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(True),
            nn.Linear(64, 6)  # 2x3 affine matrix parameters
        )
        # Initialize affine matrix to identity transform: [[1, 0, 0], [0, 1, 0]]
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = self.localization(x)
        xs = xs.view(xs.size(0), -1)
        theta = self.fc_loc(xs).view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x_aligned = F.grid_sample(x, grid, align_corners=False)
        return x_aligned


class MinutiaeDecoderHead(nn.Module):
    """
    Convolutional Decoder reconstructing the 6-channel minutiae orientation heatmap
    (Section 3.4 'Side-Task Minutiae Map Estimation').
    """
    def __init__(self, in_channels: int = 512, out_channels: int = 6):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 128, kernel_size=4, stride=2, padding=1),  # 7x7 -> 14x14
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),           # 14x14 -> 28x28
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=0),            # 28x28 -> 58x58
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((64, 64)),                                             # Matches target 64x64
            nn.Sigmoid()
        )

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        return self.decoder(feat_map)


class DeepPrintPaperDualBranchModel(nn.Module):
    """
    Complete DeepPrint Architecture from IEEE TPAMI 2021:
    - STN Alignment Module
    - Shared Stem
    - Dual Branches (96-d Texture + 96-d Minutiae)
    - Minutiae Map Decoder Head (6-channel heatmap)
    - L2-Normalized 192-d Fixed-Length Fingerprint Representation
    """
    def __init__(self, num_classes: int, embedding_dim: int = 192, pretrained: bool = True):
        super().__init__()
        assert embedding_dim == 192, "DeepPrint paper specifies 192-dim representation (96 tex + 96 min)"
        self.num_classes = num_classes

        # 1. Spatial Transformer Network
        self.stn = SpatialTransformerNetwork()

        # 2. Fast Residual Base Model (ResNet-18 chosen for rapid CPU/GPU training)
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base_tex = models.resnet18(weights=weights)
        base_min = models.resnet18(weights=weights)

        # Shared convolutional stem
        self.shared_stem = nn.Sequential(
            base_tex.conv1,
            base_tex.bn1,
            base_tex.relu,
            base_tex.maxpool,
            base_tex.layer1
        )

        # 3. Texture Branch (Extracts ridge patterns & frequency)
        self.tex_layer2 = base_tex.layer2
        self.tex_layer3 = base_tex.layer3
        self.tex_layer4 = base_tex.layer4
        self.tex_gap = nn.AdaptiveAvgPool2d((1, 1))
        self.tex_fc = nn.Linear(512, 96, bias=False)
        self.tex_bn = nn.BatchNorm1d(96)
        self.tex_classifier = nn.Linear(96, num_classes)

        # 4. Minutiae Branch (Guides features via minutiae heatmap decoder)
        self.min_layer2 = base_min.layer2
        self.min_layer3 = base_min.layer3
        self.min_layer4 = base_min.layer4
        self.min_gap = nn.AdaptiveAvgPool2d((1, 1))
        self.min_fc = nn.Linear(512, 96, bias=False)
        self.min_bn = nn.BatchNorm1d(96)
        self.min_classifier = nn.Linear(96, num_classes)

        # Minutiae Heatmap Decoder
        self.minutiae_decoder = MinutiaeDecoderHead(in_channels=512, out_channels=6)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts the fixed-length 192-d representation R in S^191 (unit hypersphere)."""
        # 1. Align via STN
        x_aligned = self.stn(x)

        # 2. Shared Stem
        stem_feat = self.shared_stem(x_aligned)

        # 3. Texture Branch -> 96-d
        t = self.tex_layer2(stem_feat)
        t = self.tex_layer3(t)
        t = self.tex_layer4(t)
        t_pool = self.tex_gap(t).flatten(1)
        r_tex = F.normalize(self.tex_bn(self.tex_fc(t_pool)), p=2, dim=1)

        # 4. Minutiae Branch -> 96-d
        m = self.min_layer2(stem_feat)
        m = self.min_layer3(m)
        m = self.min_layer4(m)
        m_pool = self.min_gap(m).flatten(1)
        r_min = F.normalize(self.min_bn(self.min_fc(m_pool)), p=2, dim=1)

        # 5. Concatenate to 192-d and project to unit hypersphere
        r_combined = torch.cat([r_min, r_tex], dim=1)
        r_final = F.normalize(r_combined, p=2, dim=1)
        return r_final

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass during multi-task training."""
        x_aligned = self.stn(x)
        stem_feat = self.shared_stem(x_aligned)

        # Texture Branch
        t = self.tex_layer2(stem_feat)
        t = self.tex_layer3(t)
        t = self.tex_layer4(t)
        t_pool = self.tex_gap(t).flatten(1)
        r_tex = F.normalize(self.tex_bn(self.tex_fc(t_pool)), p=2, dim=1)
        logits_tex = self.tex_classifier(r_tex)

        # Minutiae Branch
        m = self.min_layer2(stem_feat)
        m = self.min_layer3(m)
        m = self.min_layer4(m)
        m_pool = self.min_gap(m).flatten(1)
        r_min = F.normalize(self.min_bn(self.min_fc(m_pool)), p=2, dim=1)
        logits_min = self.min_classifier(r_min)

        # Side-Task Minutiae Heatmap Prediction
        predicted_heatmap = self.minutiae_decoder(m)

        # 192-d Normalized Embedding
        embeddings = F.normalize(torch.cat([r_min, r_tex], dim=1), p=2, dim=1)
        return logits_tex, logits_min, predicted_heatmap, embeddings


def load_deepprint_model(checkpoint_path: str, device: torch.device) -> Tuple[nn.Module, dict]:
    """
    Intelligently inspects and loads either DeepPrintResNet, DeepPrintMobileNet,
    or DeepPrintPaperDualBranchModel according to checkpoint state dict metadata.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    num_classes = ckpt.get("num_classes", 700)
    embedding_dim = ckpt.get("embedding_dim", 192)
    state = ckpt.get("model_state_dict", ckpt)
    arch = ckpt.get("architecture", "")

    if "stn.localization.0.weight" in state or "DeepPrintPaperDualBranch" in arch:
        model = DeepPrintPaperDualBranchModel(num_classes=num_classes, embedding_dim=embedding_dim, pretrained=False)
        model_name = "DeepPrint Paper Dual-Branch (ResNet Backbone + STN + Minutiae Decoder)"
    elif "features.0.0.weight" in state:
        model = DeepPrintMobileNet(num_classes=num_classes, embedding_dim=embedding_dim, pretrained=False)
        model_name = "DeepPrint MobileNetV3"
    else:
        model = DeepPrintResNet(num_classes=num_classes, embedding_dim=embedding_dim, pretrained=False)
        model_name = "DeepPrint ResNet-50 Baseline"

    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, {
        "model_name": model_name,
        "num_classes": num_classes,
        "embedding_dim": embedding_dim,
        "checkpoint_path": checkpoint_path
    }

