import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import recall_score

from ..core.config import DEVICE, LR_PATIENCE, MODEL_DIR, IMG_CLS, ensure_dirs
from ..core.data import build_loaders
from ..core.utils import setup_logger, log_epoch, save_ckpt
from .losses import get_cls_weights
from .seg_guided import get_seg_map_batch


def _eval_probs_epoch(model, loader, seg_model, cls_img_size=IMG_CLS):
    model.eval()
    total_loss = correct = total = 0
    all_p, all_l = [], []
    crit = nn.NLLLoss(weight=get_cls_weights())
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            seg_map = get_seg_map_batch(seg_model, imgs, cls_img_size=cls_img_size)
            probs = model(imgs, seg_map)
            logp = torch.log(torch.clamp(probs, 1e-8, 1.0))
            loss = crit(logp, labels)
            total_loss += loss.item() * imgs.size(0)
            preds = probs.argmax(1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
            all_p.extend(preds.cpu().numpy())
            all_l.extend(labels.cpu().numpy())
    macro_recall = recall_score(all_l, all_p, average="macro", zero_division=0)
    return total_loss / total, correct / total, macro_recall


def train_ensemble_fusion(ensemble_model, seg_model, epochs=15, lr=1e-3, fine_tune_backbones=False, img_size=IMG_CLS):
    ensure_dirs()
    ensemble_model.to(DEVICE)
    seg_model.to(DEVICE)
    seg_model.eval()

    if not fine_tune_backbones:
        ensemble_model.freeze_backbones()
    else:
        ensemble_model.unfreeze_backbones()

    tr_ld, va_ld, te_ld, tr_s, va_s, te_s = build_loaders("classify", img_size=img_size, use_overlay=False)

    logger = setup_logger("ensemble")
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, ensemble_model.parameters()), lr=lr)
    crit = nn.NLLLoss(weight=get_cls_weights())
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=LR_PATIENCE)

    best_recall = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_macro_recall": []}

    for epoch in range(1, epochs + 1):
        ensemble_model.train()
        tr_loss = tr_correct = tr_total = 0
        for imgs, labels in tqdm(tr_ld, leave=False, desc="[Ensemble] Train"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            seg_map = get_seg_map_batch(seg_model, imgs, cls_img_size=img_size)
            opt.zero_grad()
            probs = ensemble_model(imgs, seg_map)
            logp = torch.log(torch.clamp(probs, 1e-8, 1.0))
            loss = crit(logp, labels)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * imgs.size(0)
            tr_correct += (probs.argmax(1) == labels).sum().item()
            tr_total += imgs.size(0)
        tr_loss /= tr_total
        tr_acc = tr_correct / tr_total

        va_loss, va_acc, va_recall = _eval_probs_epoch(ensemble_model, va_ld, seg_model, cls_img_size=img_size)

        sched.step(va_recall)
        log_epoch(logger, epoch, epochs, {
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
                "model_state": ensemble_model.state_dict(),
                "best_metric": best_recall,
                "history": history,
            }, str(MODEL_DIR / "ensemble_best.pth"))
            logger.info(f"  Best ensemble saved (macro_recall={best_recall:.4f})")

    logger.info(f"Learned fusion weights: {ensemble_model.logit_w.detach().cpu().tolist()}")
    logger.info(f"Learned temperatures: {ensemble_model.temps.detach().cpu().tolist()}")

    return ensemble_model, history, tr_ld, va_ld, te_ld, tr_s, va_s, te_s
