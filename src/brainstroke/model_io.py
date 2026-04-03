import os
from typing import Dict, Optional

import torch

from .core.config import IMG_CLS, MODEL_DIR, PRETRAINED_DIR
from .models import (
    DenseNet121,
    EfficientNetB4,
    UNet,
    SwinUNet,
    TIMM_AVAILABLE,
    SegGuidedDenseNet,
    SegGuidedEfficientNet,
    ConfidenceEnsemble,
)
from .core.utils import load_ckpt

CHECKPOINTS = {
    "densenet121": "densenet121_best.pth",
    "efficientnet_b4": "efficientnet_b4_best.pth",
    "unet": "unet_best.pth",
    "swin_unet": "swin_unet_best.pth",
    "sg_densenet121": "sg_densenet121_best.pth",
    "sg_efficientnet_b4": "sg_efficientnet_b4_best.pth",
    "ensemble": "ensemble_best.pth",
}


def _ckpt_path(name: str) -> str:
    p = PRETRAINED_DIR / CHECKPOINTS[name]
    if p.exists():
        return str(p)
    p2 = MODEL_DIR / CHECKPOINTS[name]
    return str(p2)


def load_models(device=None) -> Dict[str, Optional[torch.nn.Module]]:
    if device is None:
        from .core.config import DEVICE as device

    models = {
        "densenet121": None,
        "efficientnet_b4": None,
        "unet": None,
        "swin_unet": None,
        "sg_densenet121": None,
        "sg_efficientnet_b4": None,
        "ensemble": None,
    }

    dn = DenseNet121(pretrained=False)
    dn, _, _, _ = load_ckpt(_ckpt_path("densenet121"), dn)
    models["densenet121"] = dn.to(device)

    eff = EfficientNetB4(pretrained=False)
    eff, _, _, _ = load_ckpt(_ckpt_path("efficientnet_b4"), eff)
    models["efficientnet_b4"] = eff.to(device)

    unet = UNet(deep_sup=True)
    unet, _, _, _ = load_ckpt(_ckpt_path("unet"), unet)
    models["unet"] = unet.to(device)

    if TIMM_AVAILABLE and os.path.exists(_ckpt_path("swin_unet")):
        swin = SwinUNet(pretrained=False, img_size=IMG_CLS)
        swin, _, _, _ = load_ckpt(_ckpt_path("swin_unet"), swin)
        models["swin_unet"] = swin.to(device)

    sg_dn = SegGuidedDenseNet(pretrained=False)
    sg_dn, _, _, _ = load_ckpt(_ckpt_path("sg_densenet121"), sg_dn)
    models["sg_densenet121"] = sg_dn.to(device)

    sg_eff = SegGuidedEfficientNet(pretrained=False)
    sg_eff, _, _, _ = load_ckpt(_ckpt_path("sg_efficientnet_b4"), sg_eff)
    models["sg_efficientnet_b4"] = sg_eff.to(device)

    ens = ConfidenceEnsemble(sg_dn, sg_eff)
    ens, _, _, _ = load_ckpt(_ckpt_path("ensemble"), ens)
    models["ensemble"] = ens.to(device)

    return models
