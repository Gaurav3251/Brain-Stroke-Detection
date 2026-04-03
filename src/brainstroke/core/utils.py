import logging
import os
from datetime import datetime

import numpy as np
import pydicom
import torch

from .config import DEVICE, LOG_DIR, SEED, WC, WW


def set_seed(seed: int = SEED) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(name: str) -> logging.Logger:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(LOG_DIR, exist_ok=True)
    logpath = os.path.join(LOG_DIR, f"{name}_{ts}.log")
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    logger.addHandler(logging.FileHandler(logpath))
    logger.addHandler(logging.StreamHandler())
    for h in logger.handlers:
        h.setFormatter(fmt)
    logger.info(f"Logger ready -> {logpath}")
    return logger


def log_epoch(logger: logging.Logger, epoch: int, total: int, phase_metrics: dict) -> None:
    parts = [f"Epoch [{epoch:03d}/{total}]"]
    for phase, metrics in phase_metrics.items():
        m_str = "  ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        parts.append(f"{phase} -> {m_str}")
    logger.info(" | ".join(parts))


def apply_hu_window(arr: np.ndarray, wc: float = WC, ww: float = WW) -> np.ndarray:
    lo = wc - ww / 2
    hi = wc + ww / 2
    arr = np.clip(arr, lo, hi)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def load_dicom(path: str) -> np.ndarray:
    """
    Load a DICOM file -> windowed float32 [H, W] array.
    Returns a blank image if decoding fails to avoid dataloader crashes.
    """
    try:
        ds = pydicom.dcmread(path)
        img = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1))
        inter = float(getattr(ds, "RescaleIntercept", 0))
        return apply_hu_window(img * slope + inter)
    except Exception as e:
        print(f"[DICOM Warning] Could not decode {os.path.basename(path)}: {e}")
        return np.zeros((512, 512), dtype=np.float32)


def save_ckpt(state: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_ckpt(path: str, model: torch.nn.Module, device=DEVICE):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt.get("epoch", 0), ckpt.get("best_metric", 0.0), ckpt.get("history", {})


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
