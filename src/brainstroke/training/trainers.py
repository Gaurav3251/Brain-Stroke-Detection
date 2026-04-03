import torch
import torch.nn as nn
import torch.optim as optim

from ..core.config import DEVICE, IMG_CLS, IMG_SEG, LR, NUM_EPOCHS, WEIGHT_DECAY, EARLY_PAT, MODEL_DIR
from ..core.data import build_loaders
from ..models import SwinUNet
from ..core.utils import set_seed, setup_logger, log_epoch, count_params, save_ckpt
from .losses import BCEDiceLoss, get_cls_weights
from .loops import EarlyStopping, warmup_lr, train_cls_epoch, eval_cls_epoch, train_seg_epoch, eval_seg_epoch


def train_model(model, model_name, task="classify"):
    set_seed()
    logger = setup_logger(model_name)
    logger.info(f"{model_name} | Params: {count_params(model):,} | Device: {DEVICE}")

    is_swin = isinstance(model, SwinUNet)

    if task == "classify":
        tr_ld, va_ld, te_ld, tr_s, va_s, te_s = build_loaders("classify", IMG_CLS, False)
        criterion = nn.CrossEntropyLoss(weight=get_cls_weights(), label_smoothing=0.1)
    else:
        seg_size = IMG_CLS if isinstance(model, SwinUNet) else IMG_SEG
        tr_ld, va_ld, te_ld, tr_s, va_s, te_s = build_loaders("segment", seg_size, True)
        criterion = BCEDiceLoss(pos_weight=10.0)

    model = model.to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=NUM_EPOCHS, eta_min=5e-5
    )
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")
    stopper = EarlyStopping(patience=EARLY_PAT)

    best_metric = 0.0
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": []
    } if task == "classify" else {
        "train_loss": [], "val_loss": [], "val_dice": []
    }

    for epoch in range(1, NUM_EPOCHS + 1):
        warmup_lr(opt, epoch - 1, LR)

        if task == "classify":
            tr_loss, tr_acc = train_cls_epoch(model, tr_ld, opt, criterion, scaler)
            va_loss, va_acc, _, _, _ = eval_cls_epoch(model, va_ld, criterion)
            metric = va_acc
            log_epoch(logger, epoch, NUM_EPOCHS, {
                "Train": {"loss": tr_loss, "acc": tr_acc},
                "Val": {"loss": va_loss, "acc": va_acc}
            })
            history["train_loss"].append(tr_loss)
            history["val_loss"].append(va_loss)
            history["train_acc"].append(tr_acc)
            history["val_acc"].append(va_acc)
        else:
            tr_loss = train_seg_epoch(model, tr_ld, opt, criterion, scaler, is_swin)
            va_loss, dice = eval_seg_epoch(model, va_ld, criterion, is_swin)
            metric = dice
            log_epoch(logger, epoch, NUM_EPOCHS, {
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
            logger.info(f"  Best saved ({'acc' if task == 'classify' else 'dice'}={best_metric:.4f})")

        if stopper(metric):
            logger.info(f"Early stop at epoch {epoch}")
            break

    logger.info(f"Done. Best: {best_metric:.4f}")
    return model, history, tr_ld, va_ld, te_ld, tr_s, va_s, te_s
