import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from ..core.config import DEVICE, IMG_CLS, IMG_SEG, LR, WEIGHT_DECAY, LR_PATIENCE, MODEL_DIR
from ..core.data import build_loaders
from ..models import get_seg_output
from ..core.utils import set_seed, setup_logger, log_epoch, count_params, save_ckpt
from .losses import get_cls_weights
from .loops import EarlyStopping, warmup_lr


def get_seg_map_batch(unet, imgs, seg_img_size=IMG_SEG, cls_img_size=IMG_CLS):
    unet.eval()
    with torch.no_grad():
        imgs_seg = torch.nn.functional.interpolate(
            imgs, size=(seg_img_size, seg_img_size), mode="bilinear", align_corners=False
        )
        out = unet(imgs_seg)
        seg_out = get_seg_output(out, unet)
        seg_map = torch.sigmoid(seg_out)
        seg_map = torch.nn.functional.interpolate(
            seg_map, size=(cls_img_size, cls_img_size), mode="bilinear", align_corners=False
        )
    return seg_map


def train_seg_guided_classifier(model, model_name, unet_model, n_epochs=10):
    set_seed()
    logger = setup_logger(model_name)
    logger.info(f"{model_name} (seg-guided) | Params: {count_params(model):,}")

    tr_ld, va_ld, te_ld, tr_s, va_s, te_s = build_loaders("classify", IMG_CLS, False)

    model = model.to(DEVICE)
    unet_model.eval()
    criterion = nn.CrossEntropyLoss(weight=get_cls_weights(), label_smoothing=0.1)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=LR_PATIENCE)
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")
    stopper = EarlyStopping(patience=LR_PATIENCE)

    best_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, n_epochs + 1):
        warmup_lr(opt, epoch - 1, LR)

        model.train()
        tr_loss = tr_correct = tr_total = 0
        for imgs, labels in tqdm(tr_ld, leave=False, desc=f"[{model_name}] Train"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            seg_map = get_seg_map_batch(unet_model, imgs)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
                logits = model(imgs, seg_map)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tr_loss += loss.item() * imgs.size(0)
            tr_correct += (logits.argmax(1) == labels).sum().item()
            tr_total += imgs.size(0)
        tr_loss /= tr_total
        tr_acc = tr_correct / tr_total

        model.eval()
        va_loss = va_correct = va_total = 0
        with torch.no_grad():
            for imgs, labels in va_ld:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                seg_map = get_seg_map_batch(unet_model, imgs)
                logits = model(imgs, seg_map)
                loss = criterion(logits, labels)
                va_loss += loss.item() * imgs.size(0)
                va_correct += (logits.argmax(1) == labels).sum().item()
                va_total += imgs.size(0)
        va_loss /= va_total
        va_acc = va_correct / va_total

        sched.step(va_acc)
        log_epoch(logger, epoch, n_epochs, {
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
                "model_state": model.state_dict(),
                "best_metric": best_acc,
                "history": history,
            }, str(MODEL_DIR / f"{model_name}_best.pth"))
            logger.info(f"  Best saved (acc={best_acc:.4f})")

        if stopper(va_acc):
            logger.info(f"Early stop at epoch {epoch}")
            break

    return model, history, tr_ld, va_ld, te_ld, tr_s, va_s, te_s
