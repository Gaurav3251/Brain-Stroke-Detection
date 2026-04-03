import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from ..core.config import DEVICE, LR_PATIENCE, MODEL_DIR
from ..core.data import build_loaders
from ..core.utils import setup_logger, log_epoch, save_ckpt
from .losses import get_cls_weights
from .seg_guided import get_seg_map_batch


def train_ensemble_fusion(ensemble_model, unet_model, epochs=15, lr=1e-3):
    ensemble_model.to(DEVICE)
    unet_model.to(DEVICE)
    unet_model.eval()

    tr_ld, va_ld, te_ld, tr_s, va_s, te_s = build_loaders("classify", img_size=224, use_overlay=False)

    logger = setup_logger("ensemble")
    opt = optim.Adam(filter(lambda p: p.requires_grad, ensemble_model.parameters()), lr=lr)
    crit = nn.CrossEntropyLoss(weight=get_cls_weights())
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=LR_PATIENCE)

    best_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        ensemble_model.train()
        tr_loss = tr_correct = tr_total = 0
        for imgs, labels in tqdm(tr_ld, leave=False, desc="[Ensemble] Train"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            seg_map = get_seg_map_batch(unet_model, imgs)
            opt.zero_grad()
            probs = ensemble_model(imgs, seg_map)
            loss = crit(probs, labels)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * imgs.size(0)
            tr_correct += (probs.argmax(1) == labels).sum().item()
            tr_total += imgs.size(0)
        tr_loss /= tr_total
        tr_acc = tr_correct / tr_total

        ensemble_model.eval()
        va_loss = va_correct = va_total = 0
        with torch.no_grad():
            for imgs, labels in va_ld:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                seg_map = get_seg_map_batch(unet_model, imgs)
                probs = ensemble_model(imgs, seg_map)
                loss = crit(probs, labels)
                va_loss += loss.item() * imgs.size(0)
                va_correct += (probs.argmax(1) == labels).sum().item()
                va_total += imgs.size(0)
        va_loss /= va_total
        va_acc = va_correct / va_total

        sched.step(va_acc)
        log_epoch(logger, epoch, epochs, {
            "Train": {"loss": tr_loss, "acc": tr_acc},
            "Val": {"loss": va_loss, "acc": va_acc}
        })
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        if va_acc > best_acc:
            best_acc = va_acc
            save_ckpt({
                "epoch": epoch,
                "model_state": ensemble_model.state_dict(),
                "best_metric": best_acc,
                "history": history,
            }, str(MODEL_DIR / "ensemble_best.pth"))
            logger.info(f"  Best ensemble saved (acc={best_acc:.4f})")

    w = F.softmax(ensemble_model.logit_w.detach().cpu(), dim=0)
    t = ensemble_model.temps.detach().cpu()
    logger.info(f"Learned fusion weights: DenseNet={w[0]:.3f} EfficientNet={w[1]:.3f}")
    logger.info(f"Learned temperatures: DenseNet={t[0]:.3f} EfficientNet={t[1]:.3f}")

    return ensemble_model, history, tr_ld, va_ld, te_ld, tr_s, va_s, te_s
