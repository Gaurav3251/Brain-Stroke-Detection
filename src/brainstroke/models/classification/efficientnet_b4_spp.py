import torch
import torch.nn as nn
import torchvision.models as tv_models
import warnings

from .base import BaseClassifier, SPP
from ...core.config import NUM_CLASSES


class EfficientNetB4SPP(BaseClassifier):
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.4, pretrained=True):
        super().__init__()
        try:
            base = tv_models.efficientnet_b4(
                weights=tv_models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
            )
        except Exception as exc:
            if not pretrained:
                raise
            warnings.warn(
                f"Could not load pretrained EfficientNet-B4 weights ({exc}). "
                "Falling back to randomly initialized weights.",
                RuntimeWarning,
            )
            base = tv_models.efficientnet_b4(weights=None)
        self.features = base.features
        in_ch = 1792
        self.spp = SPP(in_ch)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(in_ch, num_classes)
        self.feature_channels = in_ch
        self.input_size = 224

    def forward_features(self, x):
        return self.features(x)

    def forward_head(self, feat):
        return self.head(self.drop(self.spp(feat)))

    def cam_layer(self):
        return self.features[5][-1].block[0][0]
