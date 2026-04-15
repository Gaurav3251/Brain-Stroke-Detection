"""Evaluate already-trained seg-guided classifiers and save metrics/reports.

Usage:
    python scripts/training/eval_seg_guided.py --model densenet201_se
    python scripts/training/eval_seg_guided.py --model mobilenet_v2
    python scripts/training/eval_seg_guided.py --model densenet201_se mobilenet_v2
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from brainstroke.core.config import METRIC_DIR, REPORT_DIR, ensure_dirs
from brainstroke.core.data import build_loaders
from brainstroke.models import SegGuidedClassifier
from brainstroke.model_io import (
    build_classifier,
    build_segmenter,
    load_model_checkpoint,
    get_input_size,
    load_champions,
)
from brainstroke.analysis.evaluation import evaluate_classifier


def _json_safe(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return v


def eval_one(model_key, seg_model):
    sg_name = f"sg_{model_key}"
    print(f"\n{'=' * 55}\n  Evaluating {sg_name}\n{'=' * 55}")

    base = build_classifier(model_key, pretrained=False)
    sg = SegGuidedClassifier(base)
    sg = load_model_checkpoint(sg, sg_name)

    img_size = get_input_size(model_key)
    _, _, te_ld, *_ = build_loaders("classify", img_size, False)

    metrics, cm, _, _, _, report = evaluate_classifier(
        sg, te_ld, sg_name, seg_model=seg_model
    )

    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with open(METRIC_DIR / f"{sg_name}_metrics.json", "w") as f:
        json.dump({k: _json_safe(v) for k, v in metrics.items()}, f, indent=2)
    with open(REPORT_DIR / f"{sg_name}_confusion_matrix.json", "w") as f:
        json.dump(cm.tolist(), f, indent=2)
    with open(REPORT_DIR / f"{sg_name}_classification_report.txt", "w") as f:
        f.write(report)

    print(f"  Saved -> {METRIC_DIR / f'{sg_name}_metrics.json'}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained seg-guided classifiers")
    parser.add_argument(
        "--model", nargs="+", required=True,
        choices=[
            "resnet50", "resnet101",
            "densenet121_se", "densenet201_se",
            "efficientnet_b4_spp", "efficientnetv2_s",
            "mobilenet_v2", "inception_v3", "xception",
            "convnext_small",
        ],
        help="Base classifier name(s) whose sg_ checkpoint to evaluate",
    )
    parser.add_argument("--seg-prior", default=None,
                        help="Segmentation prior (default: best from champions.json)")
    args = parser.parse_args()

    ensure_dirs()
    champs = load_champions()
    seg_prior = args.seg_prior or champs.get("best_segmentation", "unet")

    seg_model = build_segmenter(seg_prior, pretrained=False)
    seg_model = load_model_checkpoint(seg_model, seg_prior)

    for model_key in args.model:
        eval_one(model_key, seg_model)

    print("\nDone.")


if __name__ == "__main__":
    main()
