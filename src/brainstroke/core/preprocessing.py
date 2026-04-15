import numpy as np
import cv2
from PIL import Image
import albumentations as A

from .config import IMG_CLS, IMG_SEG

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def overlay_to_mask(path: str, size: int) -> np.ndarray:
    """
    Convert color overlay to binary mask using multiple color heuristics.
    Returns float32 [H, W] with 1=stroke, 0=background.
    """
    img = Image.open(path).convert("RGB").resize((size, size), Image.NEAREST)
    arr = np.array(img, dtype=np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    mask_red = (r > 0.5) & (g < 0.6) & (b < 0.5)
    mask_yellow = (r > 0.6) & (g > 0.6) & (b < 0.4)
    grey_diff = np.abs(r - g) + np.abs(g - b) + np.abs(r - b)
    mask_sat = (grey_diff > 0.3) & (r + g + b > 0.3)

    mask = (mask_red | mask_yellow | mask_sat).astype(np.float32)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask.astype(np.float32)


def get_train_aug_cls(size: int = IMG_CLS) -> A.Compose:
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.Affine(
            scale=(0.92, 1.08),
            translate_percent=(-0.05, 0.05),
            rotate=(-12, 12),
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            p=0.6,
        ),
        A.RandomResizedCrop(size=(size, size), scale=(0.9, 1.0), ratio=(0.95, 1.05), p=0.4),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.6),
        A.RandomGamma(gamma_limit=(85, 115), p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.4),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(std_range=(0.01, 0.06), p=0.3),
        A.CoarseDropout(num_holes_range=(1, 6),
                        hole_height_range=(8, size // 16),
                        hole_width_range=(8, size // 16),
                        fill=0, p=0.25),
        A.Normalize(mean=MEAN, std=STD),
    ])


def get_val_aug(size: int = IMG_CLS) -> A.Compose:
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=MEAN, std=STD),
    ])


def get_train_aug_seg(size: int = IMG_SEG) -> A.Compose:
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=0.5),
        A.ElasticTransform(alpha=12, sigma=4, p=0.3),
        A.CLAHE(clip_limit=2.0, p=0.5),
        A.RandomGamma(gamma_limit=(90, 110), p=0.35),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.12, p=0.35),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
        A.Normalize(mean=MEAN, std=STD),
    ], additional_targets={"mask": "mask"})


def get_val_aug_seg(size: int = IMG_SEG) -> A.Compose:
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=MEAN, std=STD),
    ], additional_targets={"mask": "mask"})
