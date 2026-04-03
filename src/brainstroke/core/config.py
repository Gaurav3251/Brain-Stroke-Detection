from __future__ import annotations

import os
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Paths (override via environment variables if needed)
DATA_ROOT = Path(os.environ.get(
    "STROKE_DATA_ROOT",
    str(PROJECT_ROOT / "data" / "Brain_Stroke_CT_Dataset"),
))

ARTIFACTS_DIR = Path(os.environ.get(
    "STROKE_ARTIFACTS_DIR",
    str(PROJECT_ROOT / "artifacts"),
))

OUTPUT_DIR = Path(os.environ.get(
    "STROKE_OUTPUT_DIR",
    str(ARTIFACTS_DIR / "outputs"),
))

MODEL_DIR = Path(os.environ.get(
    "STROKE_MODEL_DIR",
    str(ARTIFACTS_DIR / "checkpoints"),
))

LOG_DIR = Path(os.environ.get(
    "STROKE_LOG_DIR",
    str(ARTIFACTS_DIR / "logs"),
))

PLOT_DIR = Path(os.environ.get(
    "STROKE_PLOT_DIR",
    str(OUTPUT_DIR / "plots"),
))

GCAM_DIR = Path(os.environ.get(
    "STROKE_GCAM_DIR",
    str(OUTPUT_DIR / "gradcam"),
))

INFER_DIR = Path(os.environ.get(
    "STROKE_INFER_DIR",
    str(OUTPUT_DIR / "inference"),
))

PRETRAINED_DIR = Path(os.environ.get(
    "STROKE_PRETRAINED_DIR",
    str(MODEL_DIR),
))

# Dataset
CLASSES = ["Normal", "Ischemia", "Bleeding"]
NUM_CLASSES = 3
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
CLASS_COUNTS = [4428, 1131, 1094]
TOTAL = 6653
CLASS_WEIGHTS = [TOTAL / (NUM_CLASSES * c) for c in CLASS_COUNTS]
SPLIT_RATIOS = (0.70, 0.15, 0.15)
SEED = 42

# Image / CT window
IMG_CLS = 224
IMG_SEG = 256
WC, WW = 40, 80

PNG_ONLY = False

# Segmentation
SEG_STROKE_ONLY = False
SEG_MIN_MASK_PX = 50

# Training
BATCH_SIZE = 16
NUM_EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-4
LR_PATIENCE = 5
EARLY_PAT = 10
WARMUP_EPOCHS = 3
GRAD_CLIP = 1.0

# Thresholds
SEG_THRESHOLD_UNET = 0.25
SEG_THRESHOLD_SWIN = 0.85

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 2
PIN_MEMORY = torch.cuda.is_available()


def ensure_dirs() -> None:
    for d in [MODEL_DIR, LOG_DIR, PLOT_DIR, GCAM_DIR, INFER_DIR, OUTPUT_DIR, ARTIFACTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
