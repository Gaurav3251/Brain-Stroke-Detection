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

TRAIN_DIR = Path(os.environ.get(
    "STROKE_TRAIN_DIR",
    str(ARTIFACTS_DIR / "train"),
))

UI_DIR = Path(os.environ.get(
    "STROKE_UI_DIR",
    str(ARTIFACTS_DIR / "ui"),
))

WEB_DIR = Path(os.environ.get(
    "STROKE_WEB_DIR",
    str(ARTIFACTS_DIR / "web"),
))

TRAIN_OUTPUT_DIR = Path(os.environ.get(
    "STROKE_OUTPUT_DIR",
    str(TRAIN_DIR / "outputs"),
))

MODEL_DIR = Path(os.environ.get(
    "STROKE_MODEL_DIR",
    str(TRAIN_DIR / "checkpoints"),
))

LOG_DIR = Path(os.environ.get(
    "STROKE_LOG_DIR",
    str(TRAIN_DIR / "logs"),
))

PLOT_DIR = Path(os.environ.get(
    "STROKE_PLOT_DIR",
    str(TRAIN_OUTPUT_DIR / "plots"),
))

REPORT_DIR = Path(os.environ.get(
    "STROKE_REPORT_DIR",
    str(TRAIN_OUTPUT_DIR / "reports"),
))

PRED_DIR = Path(os.environ.get(
    "STROKE_PRED_DIR",
    str(TRAIN_OUTPUT_DIR / "predictions"),
))

METRIC_DIR = Path(os.environ.get(
    "STROKE_METRIC_DIR",
    str(TRAIN_OUTPUT_DIR / "metrics"),
))

CHAMPION_FILE = Path(os.environ.get(
    "STROKE_CHAMPION_FILE",
    str(METRIC_DIR / "champions.json"),
))

GCAM_DIR = Path(os.environ.get(
    "STROKE_GCAM_DIR",
    str(TRAIN_OUTPUT_DIR / "gradcam"),
))

INFER_DIR = Path(os.environ.get(
    "STROKE_INFER_DIR",
    str(UI_DIR / "inference"),
))

PRETRAINED_DIR = Path(os.environ.get(
    "STROKE_PRETRAINED_DIR",
    str(MODEL_DIR),
))

# Dataset
CLASSES = ["Normal", "Bleeding", "Ischemia"]
NUM_CLASSES = 3
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
CLASS_COUNTS = [3123, 2500, 2500]
TOTAL = sum(CLASS_COUNTS)
CLASS_WEIGHTS = [TOTAL / (NUM_CLASSES * c) for c in CLASS_COUNTS]
SPLIT_RATIOS = (0.70, 0.15, 0.15)
SEED = 42

# Image / CT window
IMG_CLS = 224
IMG_SEG = 256
WC, WW = 40, 80

PNG_ONLY = False

# Segmentation
'''True --> default for Unet : exclude Normal images from segmentation training like for unet,
False --> swinunet overrides to false (using build_loaders()) : include Normal images in segmentation training like for swin-unet
(it does classif + segmentation)'''
SEG_STROKE_ONLY = True
SEG_MIN_MASK_PX = 50

# Training
BATCH_SIZE = 32
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
    for d in [
        ARTIFACTS_DIR, TRAIN_DIR, UI_DIR, WEB_DIR,
        MODEL_DIR, LOG_DIR, PLOT_DIR, REPORT_DIR, PRED_DIR, METRIC_DIR, GCAM_DIR,
        INFER_DIR, TRAIN_OUTPUT_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)
