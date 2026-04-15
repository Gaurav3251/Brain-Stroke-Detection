import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseClassifier(nn.Module):
    input_size = 224

    def forward_features(self, x):
        raise NotImplementedError

    def forward_head(self, feat):
        raise NotImplementedError

    def forward(self, x):
        feat = self.forward_features(x)
        return self.forward_head(feat)

    def cam_layer(self):
        return None


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
