import warnings

import torch
import torch.nn as nn
import torchvision.models as tv_models

from .base import BaseClassifier
from ...core.config import NUM_CLASSES


class EfficientNetV2S(BaseClassifier):
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.3, pretrained=True):
        super().__init__()
        try:
            base = tv_models.efficientnet_v2_s(
                weights=tv_models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
            )
        except Exception as exc:
            if not pretrained:
                raise
            warnings.warn(
                f"Could not load pretrained EfficientNet-V2-S weights ({exc}). "
                "Falling back to randomly initialized weights.",
                RuntimeWarning,
            )
            base = tv_models.efficientnet_v2_s(weights=None)

        self.features = base.features
        in_ch = base.classifier[1].in_features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(in_ch, num_classes)
        self.feature_channels = in_ch
        self.input_size = 224

    def forward_features(self, x):
        return self.features(x)

    def forward_head(self, feat):
        return self.head(self.drop(self.pool(feat).flatten(1)))

    def cam_layer(self):
        return self.features[-1][0]
