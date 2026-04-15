import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import recall_score
from tqdm import tqdm

from ..core.config import DEVICE, IMG_CLS, IMG_SEG, LR, WEIGHT_DECAY, LR_PATIENCE, MODEL_DIR, METRIC_DIR, REPORT_DIR, ensure_dirs
from ..core.data import build_loaders
from ..models import get_seg_output
from ..core.utils import set_seed, setup_logger, log_epoch, count_params, save_ckpt
from .losses import get_cls_weights
from .loops import EarlyStopping, warmup_lr


def get_seg_map_batch(seg_model, imgs, seg_img_size=IMG_SEG, cls_img_size=IMG_CLS):
    seg_model.eval()
    with torch.no_grad():
        target_seg_size = getattr(seg_model, "input_size", seg_img_size)
        imgs_seg = torch.nn.functional.interpolate(
            imgs, size=(target_seg_size, target_seg_size), mode="bilinear", align_corners=False
        )
        out = seg_model(imgs_seg)
        seg_out = get_seg_output(out, seg_model)
        seg_map = torch.sigmoid(seg_out)
        seg_map = torch.nn.functional.interpolate(
            seg_map, size=(cls_img_size, cls_img_size), mode="bilinear", align_corners=False
        )
    return seg_map


@torch.no_grad()
def _eval_sg_cls_epoch(model, loader, criterion, seg_model, cls_img_size=IMG_CLS):
    """Evaluate seg-guided classifier WITH segmentation maps."""
    model.eval()
    total_loss = correct = total = 0
    all_p, all_l, all_prob = [], [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        seg_map = get_seg_map_batch(seg_model, imgs, cls_img_size=cls_img_size)
        logits = model(imgs, seg_map)
        loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        all_p.extend(preds.cpu().numpy())
        all_l.extend(labels.cpu().numpy())
        all_prob.extend(F.softmax(logits, -1).cpu().numpy())
    macro_recall = recall_score(all_l, all_p, average="macro", zero_division=0)
    return (total_loss / total, correct / total, macro_recall,
            np.array(all_p), np.array(all_l), np.array(all_prob))


def train_seg_guided_classifier(model, model_name, seg_model, img_size=IMG_CLS, n_epochs=10, use_amp=True):
    ensure_dirs()
    set_seed()
    logger = setup_logger(model_name)
    logger.info(f"{model_name} (seg-guided) | Params: {count_params(model):,}")

    tr_ld, va_ld, te_ld, tr_s, va_s, te_s = build_loaders("classify", img_size, False)

    model = model.to(DEVICE)
    seg_model.eval()
    criterion = nn.CrossEntropyLoss(weight=get_cls_weights(), label_smoothing=0.1)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=LR_PATIENCE)
    amp_enabled = use_amp and DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    stopper = EarlyStopping(patience=LR_PATIENCE)

    best_recall = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_macro_recall": []}

    for epoch in range(1, n_epochs + 1):
        warmup_lr(opt, epoch - 1, LR)

        model.train()
        tr_loss = tr_correct = tr_total = 0
        for imgs, labels in tqdm(tr_ld, leave=False, desc=f"[{model_name}] Train"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            seg_map = get_seg_map_batch(seg_model, imgs, cls_img_size=img_size)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(imgs, seg_map)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            tr_loss += loss.item() * imgs.size(0)
            tr_correct += (logits.argmax(1) == labels).sum().item()
            tr_total += imgs.size(0)
        tr_loss /= tr_total
        tr_acc = tr_correct / tr_total

        torch.cuda.empty_cache()
        va_loss, va_acc, va_recall, _, _, _ = _eval_sg_cls_epoch(
            model, va_ld, criterion, seg_model, cls_img_size=img_size
        )

        sched.step(va_recall)
        log_epoch(logger, epoch, n_epochs, {
            "Train": {"loss": tr_loss, "acc": tr_acc},
            "Val": {"loss": va_loss, "acc": va_acc, "macro_recall": va_recall}
        })
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        history["val_macro_recall"].append(va_recall)

        if va_recall > best_recall:
            best_recall = va_recall
            save_ckpt({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "best_metric": best_recall,
                "history": history,
            }, str(MODEL_DIR / f"{model_name}_best.pth"))
            logger.info(f"  Best saved (macro_recall={best_recall:.4f})")

        if stopper(va_recall):
            logger.info(f"Early stop at epoch {epoch}")
            break

    # --- evaluate on test set with seg guidance & save metrics ---
    from ..analysis.evaluation import evaluate_classifier

    metrics, cm, _, _, _, report = evaluate_classifier(
        model, te_ld, model_name, seg_model=seg_model
    )
    metrics["val_macro_recall_best"] = best_recall

    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    def _json_safe(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        return v

    with open(METRIC_DIR / f"{model_name}_metrics.json", "w") as f:
        json.dump({k: _json_safe(v) for k, v in metrics.items()}, f, indent=2)
    with open(REPORT_DIR / f"{model_name}_confusion_matrix.json", "w") as f:
        json.dump(cm.tolist(), f, indent=2)
    with open(REPORT_DIR / f"{model_name}_classification_report.txt", "w") as f:
        f.write(report)

    logger.info(f"Test metrics -> {METRIC_DIR / f'{model_name}_metrics.json'}")

    return model, history, tr_ld, va_ld, te_ld, tr_s, va_s, te_s
