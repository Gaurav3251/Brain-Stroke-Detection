import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.config import CLASS_WEIGHTS, DEVICE


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, p, t):
        p = torch.sigmoid(p).flatten(1)
        t = t.flatten(1)
        return 1 - (2 * (p * t).sum(1) + self.smooth) / (p.sum(1) + t.sum(1) + self.smooth)


class BCEDiceLoss(nn.Module):
    def __init__(self, w=0.5, pos_weight=10.0):
        super().__init__()
        self.w = w
        self.dice = DiceLoss()
        self.pos_weight = pos_weight

    def forward(self, p, t):
        pw = torch.tensor([self.pos_weight], device=p.device)
        bce = F.binary_cross_entropy_with_logits(p, t, pos_weight=pw)
        return self.w * bce + (1 - self.w) * self.dice(p, t).mean()


def ds_loss(main, ds_outs, target, criterion):
    total = criterion(main, target)
    for i, out in enumerate(ds_outs):
        total = total + (0.5 ** (i + 1)) * criterion(out, target)
    return total


def get_cls_weights():
    return torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(DEVICE)
