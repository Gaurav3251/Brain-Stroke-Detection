import glob
import os
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

from .config import (
    DATA_ROOT,
    CLS2IDX,
    CLASSES,
    CLASS_COUNTS,
    SPLIT_RATIOS,
    SEED,
    IMG_CLS,
    IMG_SEG,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    PNG_ONLY,
    SEG_STROKE_ONLY,
    SEG_MIN_MASK_PX,
)
from .preprocessing import overlay_to_mask, get_train_aug_cls, get_val_aug, get_train_aug_seg, get_val_aug_seg
from .utils import load_dicom


def collect_samples(use_overlay: bool = False, stroke_only: bool | None = None) -> List[dict]:
    # stroke_only=None falls back to global SEG_STROKE_ONLY
    if stroke_only is None:
        stroke_only = SEG_STROKE_ONLY

    samples = []
    skipped_empty = 0

    for cls_name in CLASSES:
        cls_dir = os.path.join(DATA_ROOT, cls_name)
        label = CLS2IDX[cls_name]
        png_dir = os.path.join(cls_dir, "PNG")
        overlay_dir = os.path.join(cls_dir, "OVERLAY") if use_overlay else None

        if use_overlay and stroke_only and cls_name == "Normal":
            continue

        for img_path in sorted(glob.glob(os.path.join(png_dir, "*.png"))):
            mask_path = None
            if overlay_dir and os.path.isdir(overlay_dir):
                cand = os.path.join(overlay_dir, os.path.basename(img_path))
                if os.path.exists(cand):
                    if use_overlay and stroke_only:
                        test_mask = overlay_to_mask(cand, 128)
                        if test_mask.sum() < SEG_MIN_MASK_PX * (128 / 256) ** 2:
                            skipped_empty += 1
                            continue
                    mask_path = cand

            if use_overlay and stroke_only and mask_path is None:
                continue

            samples.append({
                "image_path": img_path,
                "mask_path": mask_path,
                "label": label,
                "is_dicom": False,
            })

        if not PNG_ONLY:
            for dcm_path in sorted(glob.glob(os.path.join(cls_dir, "DICOM", "*.dcm"))):
                if use_overlay:
                    continue
                samples.append({
                    "image_path": dcm_path,
                    "mask_path": None,
                    "label": label,
                    "is_dicom": True,
                })

    if skipped_empty > 0:
        print(f"[Dataset] Skipped {skipped_empty} samples with empty/tiny masks")
    return samples


def split_samples(samples: List[dict]) -> Tuple[List[dict], List[dict], List[dict]]:
    labels = [s["label"] for s in samples]
    tr_r, va_r, te_r = SPLIT_RATIOS
    tr_idx, rest_idx = train_test_split(
        range(len(samples)),
        test_size=1 - tr_r,
        stratify=labels,
        random_state=SEED,
    )
    rest_lbl = [labels[i] for i in rest_idx]
    va_idx, te_idx = train_test_split(
        rest_idx,
        test_size=te_r / (va_r + te_r),
        stratify=rest_lbl,
        random_state=SEED,
    )
    return [samples[i] for i in tr_idx], [samples[i] for i in va_idx], [samples[i] for i in te_idx]


class StrokeDataset(Dataset):
    def __init__(self, samples, transform=None, task="classify", img_size=IMG_CLS):
        self.samples = samples
        self.transform = transform
        self.task = task
        self.img_size = img_size

    def __len__(self):
        return len(self.samples)

    def _load(self, path, is_dicom):
        if is_dicom:
            gray = load_dicom(path)
            arr = np.stack([gray] * 3, axis=-1)
            arr = (arr * 255).astype(np.uint8)
            arr = cv2.resize(arr, (self.img_size, self.img_size))
        else:
            img = Image.open(path).convert("RGB").resize((self.img_size, self.img_size))
            arr = np.array(img, dtype=np.uint8)
        return arr

    def __getitem__(self, idx):
        s = self.samples[idx]
        label = s["label"]
        img = self._load(s["image_path"], s["is_dicom"])
        mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)

        if self.transform:
            if self.task == "segment" and s["mask_path"]:
                m = overlay_to_mask(s["mask_path"], self.img_size)
                result = self.transform(image=img, mask=m)
                img = result["image"]
                mask = result["mask"].astype(np.float32)
            else:
                img = self.transform(image=img)["image"]
        elif self.task == "segment" and s["mask_path"]:
            mask = overlay_to_mask(s["mask_path"], self.img_size)

        img_t = torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32))
        if self.task == "segment":
            return img_t, torch.from_numpy(mask).unsqueeze(0), torch.tensor(label, dtype=torch.long)
        return img_t, torch.tensor(label, dtype=torch.long)


def make_sampler(samples):
    labels = [s["label"] for s in samples]
    w_cls = [1.0 / c for c in CLASS_COUNTS]
    weights = [w_cls[l] for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def build_loaders(task: str = "classify", img_size: int = IMG_CLS, use_overlay: bool = False, stroke_only: bool | None = None):
    all_s = collect_samples(use_overlay=use_overlay, stroke_only=stroke_only)
    tr_s, va_s, te_s = split_samples(all_s)
    print(f"[Dataset] Train:{len(tr_s)} Val:{len(va_s)} Test:{len(te_s)}")

    if task == "classify":
        tr_aug, va_aug = get_train_aug_cls(img_size), get_val_aug(img_size)
    else:
        tr_aug, va_aug = get_train_aug_seg(img_size), get_val_aug_seg(img_size)

    tr_ds = StrokeDataset(tr_s, tr_aug, task, img_size)
    va_ds = StrokeDataset(va_s, va_aug, task, img_size)
    te_ds = StrokeDataset(te_s, va_aug, task, img_size)

    tr_ld = DataLoader(tr_ds, BATCH_SIZE, sampler=make_sampler(tr_s),
                       num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    va_ld = DataLoader(va_ds, BATCH_SIZE, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    te_ld = DataLoader(te_ds, BATCH_SIZE, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    return tr_ld, va_ld, te_ld, tr_s, va_s, te_s
