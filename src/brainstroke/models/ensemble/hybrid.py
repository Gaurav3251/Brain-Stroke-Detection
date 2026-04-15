import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.config import NUM_CLASSES
from ..classification.base import BaseClassifier


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


class SegGuidedClassifier(nn.Module):
    def __init__(self, base: BaseClassifier, num_classes=NUM_CLASSES):
        super().__init__()
        self.base = base
        self.seg_attn = SegGuidedAttention(base.feature_channels)
        self.num_classes = num_classes
        self.seg_guided = True
        self.input_size = getattr(base, "input_size", 224)

    def forward(self, x, seg_map=None):
        feat = self.base.forward_features(x)
        if seg_map is not None:
            feat = self.seg_attn(feat, seg_map)
        return self.base.forward_head(feat)

    def cam_layer(self):
        return self.base.cam_layer()


class ConfidenceEnsemble(nn.Module):
    def __init__(self, models, num_classes=NUM_CLASSES):
        super().__init__()
        self.models = nn.ModuleList([m for _, m, _ in models])
        self.model_names = [n for n, _, _ in models]
        self.uses_seg = [u for _, _, u in models]
        self.logit_w = nn.Parameter(torch.ones(len(models)))
        self.temps = nn.Parameter(torch.ones(len(models)))
        self.num_classes = num_classes

    def _resize_if_needed(self, x, size):
        if size is None:
            return x
        if x.shape[-1] != size:
            return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        return x

    def forward(self, x, seg_map=None):
        w = F.softmax(self.logit_w, dim=0)
        probs = []
        for i, model in enumerate(self.models):
            t = torch.clamp(self.temps[i], 0.5, 5.0)
            target_size = getattr(model, "input_size", x.shape[-1])
            xx = self._resize_if_needed(x, target_size)
            seg = self._resize_if_needed(seg_map, target_size) if seg_map is not None else None
            if self.uses_seg[i]:
                logits = model(xx, seg)
            else:
                logits = model(xx)
            p = F.softmax(logits / t, dim=-1) * w[i]
            probs.append(p)
        return torch.stack(probs, dim=0).sum(dim=0)

    def freeze_backbones(self):
        for model in self.models:
            for p in model.parameters():
                p.requires_grad = False
        print("[Ensemble] Backbones frozen - training fusion weights and temperatures only.")

    def unfreeze_backbones(self):
        for model in self.models:
            for p in model.parameters():
                p.requires_grad = True
        print("[Ensemble] Backbones unfrozen - fine-tuning full ensemble.")
