import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

from ..core.config import NUM_CLASSES
from .efficientnetb4 import SPP


class SegGuidedAttention(nn.Module):
    def __init__(self, feature_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(feature_channels + 1, feature_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.Sigmoid(),
        )

    def forward(self, features, seg_map):
        seg_scaled = F.interpolate(seg_map, size=features.shape[2:],
                                   mode="bilinear", align_corners=False)
        gate = self.conv(torch.cat([features, seg_scaled], dim=1))
        return features * gate


class SegGuidedDenseNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.4, pretrained=True):
        super().__init__()
        base = tv_models.densenet121(
            weights=tv_models.DenseNet121_Weights.DEFAULT if pretrained else None
        )
        self.features = base.features
        in_ch = base.classifier.in_features
        self.seg_attn = SegGuidedAttention(in_ch)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(in_ch, num_classes)
        self._in_ch = in_ch

    def forward(self, x, seg_map=None):
        f = torch.relu(self.features(x))
        if seg_map is not None:
            f = self.seg_attn(f, seg_map)
        return self.head(self.drop(self.pool(f).flatten(1)))

    def cam_layer(self):
        return self.features.denseblock4.denselayer16.conv2


class SegGuidedEfficientNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.4, pretrained=True):
        super().__init__()
        base = tv_models.efficientnet_b4(
            weights=tv_models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
        )
        self.features = base.features
        in_ch = 1792
        self.seg_attn = SegGuidedAttention(in_ch)
        self.spp = SPP(in_ch)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(in_ch, num_classes)
        self._in_ch = in_ch

    def forward(self, x, seg_map=None):
        f = self.features(x)
        if seg_map is not None:
            f = self.seg_attn(f, seg_map)
        return self.head(self.drop(self.spp(f)))

    def cam_layer(self):
        return self.features[-1][0]


class ConfidenceEnsemble(nn.Module):
    def __init__(self, sg_dense, sg_effnet, num_classes=NUM_CLASSES):
        super().__init__()
        self.sg_dense = sg_dense
        self.sg_effnet = sg_effnet
        self.logit_w = nn.Parameter(torch.ones(2))
        self.temps = nn.Parameter(torch.ones(2))

    def forward(self, x, seg_map=None):
        w = F.softmax(self.logit_w, dim=0)
        t0 = torch.clamp(self.temps[0], 0.5, 5.0)
        t1 = torch.clamp(self.temps[1], 0.5, 5.0)

        p0 = F.softmax(self.sg_dense(x, seg_map) / t0, dim=-1) * w[0]
        p1 = F.softmax(self.sg_effnet(x, seg_map) / t1, dim=-1) * w[1]
        return p0 + p1

    def freeze_backbones(self):
        for param in self.sg_dense.features.parameters():
            param.requires_grad = False
        for param in self.sg_effnet.features.parameters():
            param.requires_grad = False
        print("[Ensemble] Backbones frozen - training fusion weights and temperatures only.")

    def unfreeze_backbones(self):
        for param in self.sg_dense.features.parameters():
            param.requires_grad = True
        for param in self.sg_effnet.features.parameters():
            param.requires_grad = True
        print("[Ensemble] Backbones unfrozen - fine-tuning full ensemble.")
