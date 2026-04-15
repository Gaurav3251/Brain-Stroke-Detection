import json
from itertools import product

import torch
import torch.nn as nn
import torch.optim as optim

from ..core.config import DEVICE, METRIC_DIR, ensure_dirs
from ..core.data import build_loaders
from ..core.utils import setup_logger
from ..models import get_classifier
from .losses import get_cls_weights
from .loops import train_cls_epoch, eval_cls_epoch


def tune_classifier(model_key: str, pretrained: bool = True, epochs: int = 5):
    ensure_dirs()
    model, img_size = get_classifier(model_key, pretrained=pretrained)
    tr_ld, va_ld, _, _, _, _ = build_loaders("classify", img_size, False)

    search_lr = [1e-4, 3e-4, 1e-5]
    search_wd = [1e-4, 1e-5]

    logger = setup_logger(f"tune_{model_key}")
    criterion = nn.CrossEntropyLoss(weight=get_cls_weights(), label_smoothing=0.1)

    results = []
    best = {"macro_recall": -1}

    for lr, wd in product(search_lr, search_wd):
        model_trial, _ = get_classifier(model_key, pretrained=pretrained)
        model_trial = model_trial.to(DEVICE)
        opt = optim.AdamW(model_trial.parameters(), lr=lr, weight_decay=wd)
        scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")

        for _ in range(epochs):
            train_cls_epoch(model_trial, tr_ld, opt, criterion, scaler)

        va_loss, va_acc, va_recall, *_ = eval_cls_epoch(model_trial, va_ld, criterion)
        trial = {
            "lr": lr,
            "weight_decay": wd,
            "val_loss": va_loss,
            "val_acc": va_acc,
            "val_macro_recall": va_recall,
        }
        results.append(trial)
        logger.info(f"Trial lr={lr} wd={wd} -> macro_recall={va_recall:.4f}")
        if va_recall > best.get("macro_recall", -1):
            best = {"macro_recall": va_recall, "lr": lr, "weight_decay": wd}

    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "model": model_key,
        "best": best,
        "trials": results,
    }
    with open(METRIC_DIR / f"{model_key}_tuning.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    return out
