import json
import os
from typing import Dict, Optional, Tuple

import torch

from .core.config import IMG_CLS, MODEL_DIR, PRETRAINED_DIR, CHAMPION_FILE
from .models import (
    get_classifier,
    MODEL_REGISTRY,
    UNet,
    ResUNet,
    AttentionUNet,
    SwinUNet,
    TIMM_AVAILABLE,
    SegGuidedClassifier,
    ConfidenceEnsemble,
)
from .core.utils import load_ckpt

CHECKPOINTS = {
    "resnet50": "resnet50_best.pth",
    "resnet101": "resnet101_best.pth",
    "densenet121_se": "densenet121_se_best.pth",
    "densenet201_se": "densenet201_se_best.pth",
    "efficientnet_b4_spp": "efficientnet_b4_spp_best.pth",
    "efficientnetv2_s": "efficientnetv2_s_best.pth",
    "mobilenet_v2": "mobilenet_v2_best.pth",
    "inception_v3": "inception_v3_best.pth",
    "xception": "xception_best.pth",
    "convnext_small": "convnext_small_best.pth",
    "swin_unet": "swin_unet_best.pth",
    "swin_unet_stroke_only": "swin_unet_stroke_only_best.pth",
    "unet": "unet_best.pth",
    "resunet": "resunet_best.pth",
    "attention_unet": "attention_unet_best.pth",
    "ensemble": "ensemble_best.pth",
    # seg-guided classifiers
    "sg_resnet50": "sg_resnet50_best.pth",
    "sg_resnet101": "sg_resnet101_best.pth",
    "sg_densenet121_se": "sg_densenet121_se_best.pth",
    "sg_densenet201_se": "sg_densenet201_se_best.pth",
    "sg_efficientnet_b4_spp": "sg_efficientnet_b4_spp_best.pth",
    "sg_efficientnetv2_s": "sg_efficientnetv2_s_best.pth",
    "sg_mobilenet_v2": "sg_mobilenet_v2_best.pth",
    "sg_inception_v3": "sg_inception_v3_best.pth",
    "sg_xception": "sg_xception_best.pth",
    "sg_convnext_small": "sg_convnext_small_best.pth",
}


def _ckpt_path(name: str) -> str:
    p = PRETRAINED_DIR / CHECKPOINTS[name]
    if p.exists():
        return str(p)
    p2 = MODEL_DIR / CHECKPOINTS[name]
    return str(p2)


def get_input_size(model_key: str) -> int:
    key = model_key.lower()
    if key in MODEL_REGISTRY:
        return MODEL_REGISTRY[key]["img_size"]
    return IMG_CLS


def build_classifier(model_key: str, pretrained=False):
    key = model_key.lower()
    if key == "swin_unet":
        return SwinUNet(pretrained=pretrained, img_size=IMG_CLS)
    model, _ = get_classifier(key, pretrained=pretrained)
    return model


def build_segmenter(model_key: str, pretrained=False):
    key = model_key.lower()
    if key == "unet":
        return UNet(deep_sup=True)
    if key == "resunet":
        return ResUNet(pretrained=pretrained)
    if key == "attention_unet":
        return AttentionUNet()
    if key in {"swin_unet", "swin_unet_stroke_only"}:
        return SwinUNet(pretrained=pretrained, img_size=IMG_CLS)
    raise ValueError(f"Unknown segmenter: {model_key}")


def load_model_checkpoint(model, key: str, device=None):
    if device is None:
        from .core.config import DEVICE as device
    model, _, _, _ = load_ckpt(_ckpt_path(key), model, device=device)
    return model.to(device)


def load_champions() -> dict:
    if os.path.exists(CHAMPION_FILE):
        with open(CHAMPION_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "group_champions": {},
        "best_segmentation": "unet",
    }


def load_ensemble(device=None) -> Tuple[ConfidenceEnsemble, torch.nn.Module]:
    if device is None:
        from .core.config import DEVICE as device

    champs = load_champions()
    group_champs = champs.get("group_champions", {})
    best_seg = champs.get("best_segmentation", "unet")

    seg_model = build_segmenter(best_seg, pretrained=False)
    seg_model = load_model_checkpoint(seg_model, best_seg, device=device)

    sg_models = []
    for grp in ["A", "B", "C", "D", "E"]:
        key = group_champs.get(grp)
        if key is None:
            continue
        base = build_classifier(key, pretrained=False)
        sg = SegGuidedClassifier(base)
        sg = load_model_checkpoint(sg, f"sg_{key}", device=device)
        sg_models.append((f"sg_{key}", sg, True))

    ensemble = ConfidenceEnsemble(sg_models)
    ensemble = load_model_checkpoint(ensemble, "ensemble", device=device)

    return ensemble, seg_model


def load_models(device=None) -> Dict[str, Optional[torch.nn.Module]]:
    if device is None:
        from .core.config import DEVICE as device

    champs = load_champions()
    best_seg = champs.get("best_segmentation", "unet")

    models: Dict[str, Optional[torch.nn.Module]] = {}

    seg_model = build_segmenter(best_seg, pretrained=False)
    seg_model = load_model_checkpoint(seg_model, best_seg, device=device)
    models["seg_prior"] = seg_model

    # Load seg-guided champion classifiers (for UI selection)
    group_champs = champs.get("group_champions", {})
    for key in group_champs.values():
        try:
            base = build_classifier(key, pretrained=False)
            sg = SegGuidedClassifier(base)
            models[f"sg_{key}"] = load_model_checkpoint(sg, f"sg_{key}", device=device)
        except Exception:
            models[f"sg_{key}"] = None

    # Segmentation models
    for seg_key in ["unet", "resunet", "attention_unet", "swin_unet_stroke_only"]:
        try:
            sm = build_segmenter(seg_key, pretrained=False)
            models[seg_key] = load_model_checkpoint(sm, seg_key, device=device)
        except Exception:
            models[seg_key] = None

    # Ensemble
    try:
        ensemble, seg_model = load_ensemble(device=device)
        models["ensemble"] = ensemble
        models["seg_prior"] = seg_model
    except Exception:
        models["ensemble"] = None

    return models
