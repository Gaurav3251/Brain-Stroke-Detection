import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

from ..core.config import NUM_CLASSES


class SPP(nn.Module):
    def __init__(self, ch, levels=(1, 2, 4)):
        super().__init__()
        self.levels = levels
        self.project = nn.Sequential(
            nn.Linear(ch * len(levels), ch),
            nn.BatchNorm1d(ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        parts = []
        for l in self.levels:
            pooled = F.adaptive_avg_pool2d(x, l)
            pooled = pooled.flatten(2).mean(dim=2)
            parts.append(pooled)
        return self.project(torch.cat(parts, dim=1))


class EfficientNetB4(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.4, pretrained=True):
        super().__init__()
        base = tv_models.efficientnet_b4(
            weights=tv_models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
        )
        self.features = base.features
        in_ch = 1792
        self.spp = SPP(in_ch)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(in_ch, num_classes)
        self._in_ch = in_ch

    def forward(self, x):
        f = self.features(x)
        f = self.spp(f)
        return self.head(self.drop(f))

    def cam_layer(self):
        return self.features[5][-1].block[0][0]
