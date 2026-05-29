import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V2_Weights


SEAL_CLASSES = [
    "bird", "boar", "dog", "dragon", "hare", "horse",
    "monkey", "ox", "ram", "rat", "snake", "tiger"
]


def build_mobilenet(num_classes=12):
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.6),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, num_classes)
    )

    return model


def build_swin(num_classes=12):
    model = models.swin_s(weights=None)

    in_feats = model.head.in_features
    model.head = nn.Sequential(
        nn.LayerNorm(in_feats),
        nn.Dropout(0.4),
        nn.Linear(in_feats, 256),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )

    return model