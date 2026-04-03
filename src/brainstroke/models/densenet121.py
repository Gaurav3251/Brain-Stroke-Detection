import torch
import torch.nn as nn
import torchvision.models as tv_models

from ..core.config import NUM_CLASSES


class ChannelAttention(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        mid = max(ch // r, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, mid), nn.ReLU(inplace=True),
            nn.Linear(mid, ch), nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x).unsqueeze(-1).unsqueeze(-1)


class DenseNet121(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.4, pretrained=True):
        super().__init__()
        base = tv_models.densenet121(
            weights=tv_models.DenseNet121_Weights.DEFAULT if pretrained else None
        )
        self.features = base.features
        in_ch = base.classifier.in_features
        self.attention = ChannelAttention(in_ch)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(in_ch, num_classes)
        self._in_ch = in_ch

    def forward(self, x):
        f = torch.relu(self.features(x))
        f = self.attention(f)
        f = self.pool(f).flatten(1)
        return self.head(self.drop(f))

    def cam_layer(self):
        return self.features.denseblock4.denselayer16.conv2
