import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import recall_score

from ..core.config import DEVICE, GRAD_CLIP, WARMUP_EPOCHS
from ..models import SwinUNet, get_seg_output
from .losses import ds_loss, get_cls_weights


class EarlyStopping:
    def __init__(self, patience, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best = None
        self.stop = False

    def __call__(self, score):
        if self.best is None:
            self.best = score
        elif score < self.best + self.min_delta:
            self.counter += 1
            self.stop = self.counter >= self.patience
        else:
            self.best = score
            self.counter = 0
        return self.stop


def warmup_lr(opt, epoch, base_lr):
    if epoch < WARMUP_EPOCHS:
        lr = base_lr * (epoch + 1) / WARMUP_EPOCHS
        for pg in opt.param_groups:
            pg["lr"] = lr


def train_cls_epoch(model, loader, opt, criterion, scaler):
    model.train()
    total_loss = correct = total = 0
    for imgs, labels in tqdm(loader, leave=False, desc="Train"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        opt.zero_grad()
        with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
            logits = model(imgs)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(opt)
        scaler.update()
        total_loss += loss.item() * imgs.size(0)
        correct += (logits.detach().argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_cls_epoch(model, loader, criterion):
    model.eval()
    total_loss = correct = total = 0
    all_p, all_l, all_prob = [], [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits = model(imgs)
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


def train_seg_epoch(model, loader, opt, criterion, scaler, is_swin=False):
    model.train()
    total_loss = total = 0
    cls_crit = nn.CrossEntropyLoss(weight=get_cls_weights(), label_smoothing=0.1)
    for batch in tqdm(loader, leave=False, desc="Train-Seg"):
        imgs, masks, labels = batch
        imgs, masks, labels = imgs.to(DEVICE), masks.to(DEVICE), labels.to(DEVICE)
        opt.zero_grad()
        with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
            out = model(imgs)
            if is_swin:
                cls_o = out[0]
                seg_o = out[1]
                loss = 0.3 * cls_crit(cls_o, labels) + 0.7 * criterion(seg_o, masks)
            else:
                seg_o = out[0] if isinstance(out, tuple) else out
                ds_outs = out[1] if isinstance(out, tuple) else []
                if ds_outs:
                    loss = ds_loss(seg_o, ds_outs, masks, criterion)
                else:
                    loss = criterion(seg_o, masks)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(opt)
        scaler.update()
        total_loss += loss.item() * imgs.size(0)
        total += imgs.size(0)
    return total_loss / total


@torch.no_grad()
def eval_seg_epoch(model, loader, criterion, is_swin=False):
    model.eval()
    total_loss = total = 0
    dice_scores, sens_scores = [], []
    for batch in loader:
        imgs, masks, _ = batch
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        out = model(imgs)
        seg_o = get_seg_output(out, model)
        loss = criterion(seg_o, masks)
        total_loss += loss.item() * imgs.size(0)
        total += imgs.size(0)
        pred_bin = (torch.sigmoid(seg_o) > 0.5).float()
        inter = (pred_bin * masks).sum((1, 2, 3))
        union = pred_bin.sum((1, 2, 3)) + masks.sum((1, 2, 3))
        dice_scores.extend(((2 * inter + 1) / (union + 1)).cpu().numpy())
        for i in range(masks.shape[0]):
            gt_px = masks[i].sum().item()
            if gt_px > 0:
                tp = (pred_bin[i] * masks[i]).sum().item()
                sens_scores.append(tp / gt_px)
    mean_sens = np.mean(sens_scores) if sens_scores else 0.0
    print(f"  Sensitivity (stroke slices only): {mean_sens:.4f} over {len(sens_scores)} slices")
    return total_loss / total, np.mean(dice_scores)
