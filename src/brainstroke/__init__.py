"""Brain stroke detection package."""

from .core.config import (
    DATA_ROOT,
    ARTIFACTS_DIR,
    OUTPUT_DIR,
    MODEL_DIR,
    LOG_DIR,
    PLOT_DIR,
    GCAM_DIR,
    INFER_DIR,
    PRETRAINED_DIR,
    CLASSES,
    NUM_CLASSES,
    IMG_CLS,
    IMG_SEG,
    SEG_THRESHOLD_UNET,
    SEG_THRESHOLD_SWIN,
    DEVICE,
    ensure_dirs,
)

