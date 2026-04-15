import torch
import torch.nn as nn
import torchvision.models as tv_models
import warnings

from .base import BaseClassifier, ChannelAttention
from ...core.config import NUM_CLASSES


class DenseNet201SE(BaseClassifier):
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.4, pretrained=True):
        super().__init__()
        try:
            base = tv_models.densenet201(
                weights=tv_models.DenseNet201_Weights.DEFAULT if pretrained else None
            )
        except Exception as exc:
            if not pretrained:
                raise
            warnings.warn(
                f"Could not load pretrained DenseNet201 weights ({exc}). "
                "Falling back to randomly initialized weights.",
                RuntimeWarning,
            )
            base = tv_models.densenet201(weights=None)
        self.features = base.features
        in_ch = base.classifier.in_features
        self.attention = ChannelAttention(in_ch)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(in_ch, num_classes)
        self.feature_channels = in_ch
        self.input_size = 224

    def forward_features(self, x):
        f = torch.relu(self.features(x))
        return self.attention(f)

    def forward_head(self, feat):
        return self.head(self.drop(self.pool(feat).flatten(1)))

    def cam_layer(self):
        return self.features.denseblock4.denselayer16.conv2
