"""
Deep Learning Model for Dermatological Screening
Uses MobileNetV2 backbone with custom classification head for skin lesion classification.
Supports 7 HAM10000 categories + confidence scoring.
"""

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

RISK_LEVELS = {
    "Actinic Keratosis": "moderate",
    "Basal Cell Carcinoma": "high",
    "Benign Keratosis": "low",
    "Dermatofibroma": "low",
    "Melanoma": "critical",
    "Melanocytic Nevi": "low",
    "Vascular Lesion": "moderate",
}

DESCRIPTIONS = {
    "Actinic Keratosis": "Pre-cancerous scaly patch caused by UV exposure. Requires monitoring and possible treatment.",
    "Basal Cell Carcinoma": "Most common skin cancer. Rarely metastasizes but can cause local tissue damage. Seek dermatologist.",
    "Benign Keratosis": "Non-cancerous skin growth (seborrheic keratosis, solar lentigo). Generally harmless.",
    "Dermatofibroma": "Common benign fibrous nodule. Usually no treatment needed unless symptomatic.",
    "Melanoma": "Most dangerous form of skin cancer. URGENT — seek immediate professional evaluation.",
    "Melanocytic Nevi": "Common mole. Usually benign but monitor for changes (ABCDE criteria).",
    "Vascular Lesion": "Blood vessel-related lesion (angioma, pyogenic granuloma). Usually benign.",
}


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
    - MobileNetV2 backbone (pretrained on ImageNet)
    - Channel attention mechanism
    - Multi-scale feature aggregation
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

        # Auxiliary confidence head
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


def load_model(weights_path=None, device="cpu"):
    """Load model with optional trained weights."""
    model = DermClassifier(num_classes=len(SKIN_CLASSES), pretrained=True)
    if weights_path:
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict(model, image, transform=None, device="cpu"):
    """
    Run inference on a PIL image.
    Returns dict with class probabilities, predicted class, risk level, etc.
    """
    if transform is None:
        transform = get_preprocessing_transform()

    input_tensor = transform(image).unsqueeze(0).to(device)
    logits, confidence = model(input_tensor)

    probabilities = F.softmax(logits, dim=1).squeeze().cpu().numpy()
    confidence_score = confidence.squeeze().cpu().item()
    predicted_idx = probabilities.argmax()
    predicted_class = SKIN_CLASSES[predicted_idx]

    # Build top-k results
    top_indices = probabilities.argsort()[::-1]
    results = []
    for idx in top_indices:
        cls_name = SKIN_CLASSES[idx]
        results.append({
            "class": cls_name,
            "probability": float(probabilities[idx]),
            "risk_level": RISK_LEVELS[cls_name],
            "description": DESCRIPTIONS[cls_name],
        })

    return {
        "prediction": predicted_class,
        "confidence": float(confidence_score),
        "top_probability": float(probabilities[predicted_idx]),
        "risk_level": RISK_LEVELS[predicted_class],
        "description": DESCRIPTIONS[predicted_class],
        "all_predictions": results,
    }
