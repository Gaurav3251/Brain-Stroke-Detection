import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..core.config import (
    DEVICE, IMG_CLS, IMG_SEG, LR, NUM_EPOCHS, WEIGHT_DECAY,
    EARLY_PAT, MODEL_DIR, METRIC_DIR, REPORT_DIR, ensure_dirs,
    SEG_THRESHOLD_UNET, SEG_THRESHOLD_SWIN,
)
from ..core.data import build_loaders
from ..models import SwinUNet
from ..core.utils import set_seed, setup_logger, log_epoch, count_params, save_ckpt
from ..analysis.evaluation import evaluate_classifier, evaluate_segmentation
from ..analysis.visualization import (
    plot_confusion_matrix,
    plot_sample_preds,
    plot_roc,
    plot_pr,
    plot_confidence_hist,
)
from .losses import BCEDiceLoss, get_cls_weights
from .loops import EarlyStopping, warmup_lr, train_cls_epoch, eval_cls_epoch, train_seg_epoch, eval_seg_epoch


def _json_ready(value):
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_ready(payload), f, indent=2)


def _save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _seg_eval_threshold(model, model_name: str) -> float:
    if isinstance(model, SwinUNet) or "swin" in model_name.lower():
        return SEG_THRESHOLD_SWIN
    return SEG_THRESHOLD_UNET


def save_classifier_outputs(model, te_ld, model_name):
    metrics, cm, y_true, y_pred, y_prob, report = evaluate_classifier(model, te_ld, model_name)
    _save_json(METRIC_DIR / f"{model_name}_metrics.json", metrics)
    _save_json(REPORT_DIR / f"{model_name}_confusion_matrix.json", cm.tolist())
    _save_text(REPORT_DIR / f"{model_name}_classification_report.txt", report)
    plot_confusion_matrix(cm, model_name, out_dir="predictions")
    plot_roc(y_true, y_prob, model_name)
    plot_pr(y_true, y_prob, model_name)
    plot_confidence_hist(y_prob, y_true, model_name)
    plot_sample_preds(model, te_ld.dataset, model_name)
    return metrics


def finalize_model_outputs(model, model_name, task="classify", img_size=IMG_CLS, stroke_only_override: bool | None = None):
    if task == "classify":
        _, _, te_ld, *_ = build_loaders("classify", img_size, False)
        return save_classifier_outputs(model, te_ld, model_name)

    seg_size = img_size if isinstance(model, SwinUNet) else IMG_SEG
    stroke_only = stroke_only_override if stroke_only_override is not None else not isinstance(model, SwinUNet)
    _, _, te_ld, *_ = build_loaders("segment", seg_size, True, stroke_only=stroke_only)
    threshold = _seg_eval_threshold(model, model_name)
    metrics = evaluate_segmentation(model, te_ld, model_name, threshold=threshold)
    metrics["eval_threshold"] = threshold
    metrics["stroke_only"] = stroke_only
    _save_json(METRIC_DIR / f"{model_name}_seg_metrics.json", metrics)
    return metrics


def train_model(
    model,
    model_name,
    task="classify",
    img_size=IMG_CLS,
    num_epochs: int | None = None,
    stroke_only_override: bool | None = None,
):
    ensure_dirs()
    set_seed()
    logger = setup_logger(model_name)
    logger.info(f"{model_name} | Params: {count_params(model):,} | Device: {DEVICE}")

    epochs = num_epochs or NUM_EPOCHS
    is_swin = isinstance(model, SwinUNet)

    if task == "classify":
        tr_ld, va_ld, te_ld, tr_s, va_s, te_s = build_loaders("classify", img_size, False)
        criterion = nn.CrossEntropyLoss(weight=get_cls_weights(), label_smoothing=0.1)
    else:
        seg_size = img_size if isinstance(model, SwinUNet) else IMG_SEG
        stroke_only = stroke_only_override if stroke_only_override is not None else not isinstance(model, SwinUNet)
        tr_ld, va_ld, te_ld, tr_s, va_s, te_s = build_loaders("segment", seg_size, True, stroke_only=stroke_only)
        criterion = BCEDiceLoss(pos_weight=10.0)

    model = model.to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=5e-5
    )
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")
    stopper = EarlyStopping(patience=EARLY_PAT)

    best_metric = 0.0
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [], "val_macro_recall": []
    } if task == "classify" else {
        "train_loss": [], "val_loss": [], "val_dice": []
    }

    for epoch in range(1, epochs + 1):
        warmup_lr(opt, epoch - 1, LR)

        if task == "classify":
            tr_loss, tr_acc = train_cls_epoch(model, tr_ld, opt, criterion, scaler)
            va_loss, va_acc, va_recall, _, _, _ = eval_cls_epoch(model, va_ld, criterion)
            metric = va_recall
            log_epoch(logger, epoch, epochs, {
                "Train": {"loss": tr_loss, "acc": tr_acc},
                "Val": {"loss": va_loss, "acc": va_acc, "macro_recall": va_recall}
            })
            history["train_loss"].append(tr_loss)
            history["val_loss"].append(va_loss)
            history["train_acc"].append(tr_acc)
            history["val_acc"].append(va_acc)
            history["val_macro_recall"].append(va_recall)
        else:
            tr_loss = train_seg_epoch(model, tr_ld, opt, criterion, scaler, is_swin)
            va_loss, dice = eval_seg_epoch(model, va_ld, criterion, is_swin)
            metric = dice
            log_epoch(logger, epoch, epochs, {
                "Train": {"loss": tr_loss},
                "Val": {"loss": va_loss, "dice": dice}
            })
            history["train_loss"].append(tr_loss)
            history["val_loss"].append(va_loss)
            history["val_dice"].append(dice)

        scheduler.step()

        if metric > best_metric:
            best_metric = metric
            ckpt_path = str(MODEL_DIR / f"{model_name}_best.pth")
            save_ckpt({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": opt.state_dict(),
                "best_metric": best_metric,
                "history": history,
            }, ckpt_path)
            logger.info(f"  Best saved ({'macro_recall' if task == 'classify' else 'dice'}={best_metric:.4f})")

        if stopper(metric):
            logger.info(f"Early stop at epoch {epoch}")
            break

    logger.info(f"Done. Best: {best_metric:.4f}")

    metrics = finalize_model_outputs(
        model,
        model_name,
        task=task,
        img_size=img_size,
        stroke_only_override=stroke_only_override,
    )
    if task == "classify":
        metrics["val_macro_recall_best"] = best_metric
        _save_json(METRIC_DIR / f"{model_name}_metrics.json", metrics)
    else:
        metrics["val_dice_best"] = best_metric
        _save_json(METRIC_DIR / f"{model_name}_seg_metrics.json", metrics)

    return model, history, tr_ld, va_ld, te_ld, tr_s, va_s, te_s
