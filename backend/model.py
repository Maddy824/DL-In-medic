"""
Dermatology image classification model
Uses MobileNetV2 backbone with custom classification head for skin lesion classification.
Supports seven HAM10000 categories for educational use.
"""

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

SKIN_CLASSES = [
    "Actinic Keratosis",
    "Basal Cell Carcinoma",
    "Benign Keratosis",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic Nevi",
    "Vascular Lesion",
]

class AttentionBlock(nn.Module):
    """Channel attention module for focusing on discriminative features."""

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        attn = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * attn


class DermClassifier(nn.Module):
    """
    Dermatological image classifier with:
    - MobileNetV2 backbone (optionally pretrained on ImageNet)
    - Channel attention mechanism
    - Dropout regularization
    """

    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()
        self.backbone = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        )
        self.feature_dim = self.backbone.classifier[1].in_features

        # Remove original classifier
        self.backbone.classifier = nn.Identity()

        # Attention module
        self.attention = AttentionBlock(1280)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

        # Legacy auxiliary head (not used for reported scores)
        self.confidence_head = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Extract features from backbone
        features = self.backbone.features(x)

        # Apply attention
        features = self.attention(features)

        # Global average pooling
        features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
        features = features.view(features.size(0), -1)

        # Classification
        logits = self.classifier(features)
        confidence = self.confidence_head(features)

        return logits, confidence


def get_preprocessing_transform():
    """Standard preprocessing for inference."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_model(weights_path, device="cpu"):
    """Load an explicit DermClassifier state dict without downloading weights."""
    if not weights_path:
        raise ValueError("A trained weights path is required")
    model = DermClassifier(num_classes=len(SKIN_CLASSES), pretrained=False)
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    if not isinstance(state_dict, Mapping):
        raise TypeError("Checkpoint must contain a model state dict")
    validate_finite_tensors(state_dict)
    model.load_state_dict(state_dict, strict=True)
    # Copying into the model's dtype can overflow even when the source is finite.
    validate_finite_tensors(model.state_dict())
    model.to(device)
    model.eval()
    return model


def validate_finite_tensors(state_dict):
    for name, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor) and not torch.isfinite(tensor).all():
            raise ValueError(f"Checkpoint contains non-finite values in {name}")


class NonFiniteInferenceError(RuntimeError):
    """The classifier cannot provide finite class scores."""


@torch.no_grad()
def predict(model, image, transform=None, device="cpu"):
    """Return softmax class scores, not calibrated probabilities of disease."""
    if transform is None:
        transform = get_preprocessing_transform()

    input_tensor = transform(image).unsqueeze(0).to(device)
    logits, _ = model(input_tensor)
    if not torch.isfinite(logits).all():
        raise NonFiniteInferenceError("Model produced non-finite logits")
    scores = F.softmax(logits, dim=1)
    if not torch.isfinite(scores).all():
        raise NonFiniteInferenceError("Model produced non-finite probabilities")
    probabilities = scores.squeeze(0).cpu().numpy()
    predicted_idx = probabilities.argmax()
    return {
        "prediction": SKIN_CLASSES[predicted_idx],
        "top_probability": round(float(probabilities[predicted_idx]), 4),
        "all_predictions": [
            {"class": SKIN_CLASSES[idx], "probability": round(float(probabilities[idx]), 4)}
            for idx in probabilities.argsort()[::-1]
        ],
    }
