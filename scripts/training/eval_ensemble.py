"""Evaluate the trained ensemble and generate all plots + metrics.

Usage:
    python scripts/training/eval_ensemble.py
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from brainstroke.core.config import METRIC_DIR, REPORT_DIR, MODEL_DIR, DEVICE, ensure_dirs
from brainstroke.core.data import build_loaders
from brainstroke.model_io import load_ensemble, get_input_size, load_champions
from brainstroke.analysis.ensemble_eval import evaluate_ensemble
from brainstroke.analysis.visualization import (
    plot_confusion_matrix,
    plot_roc,
    plot_pr,
    plot_confidence_hist,
    plot_training_curves,
    plot_sample_preds_with_predictor,
)
from brainstroke.training.seg_guided import get_seg_map_batch


def _json_safe(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return v


def main():
    ensure_dirs()

    # load ensemble + seg prior
    ensemble, seg_model = load_ensemble()
    ensemble.eval()
    seg_model.eval()

    # determine max input size from group champions
    champs = load_champions()
    group_champs = champs.get("group_champions", {})
    sizes = [get_input_size(k) for k in group_champs.values() if k]
    max_size = max(sizes) if sizes else 224

    _, _, te_ld, _, _, te_s = build_loaders("classify", max_size, False)

    # --- metrics ---
    print("Running ensemble evaluation on test set...")
    metrics, cm, y_true, y_pred, y_prob = evaluate_ensemble(
        ensemble, seg_model, te_ld, model_name="ensemble"
    )

    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with open(METRIC_DIR / "ensemble_metrics.json", "w") as f:
        json.dump({k: _json_safe(v) for k, v in metrics.items()}, f, indent=2)
    with open(REPORT_DIR / "ensemble_confusion_matrix.json", "w") as f:
        json.dump(cm.tolist(), f, indent=2)

    from sklearn.metrics import classification_report
    from brainstroke.core.config import CLASSES
    report = classification_report(y_true, y_pred, target_names=CLASSES)
    with open(REPORT_DIR / "ensemble_classification_report.txt", "w") as f:
        f.write(report)

    print(f"Metrics -> {METRIC_DIR / 'ensemble_metrics.json'}")

    # --- plots ---
    print("Generating plots...")
    plot_confusion_matrix(cm, "ensemble", out_dir="predictions")
    plot_roc(y_true, y_prob, "ensemble")
    plot_pr(y_true, y_prob, "ensemble")
    plot_confidence_hist(y_prob, y_true, "ensemble")

    # training curves from checkpoint history
    ckpt_path = MODEL_DIR / "ensemble_best.pth"
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        history = ckpt.get("history")
        if history:
            plot_training_curves(history, "ensemble")

    # sample predictions using ensemble with seg guidance
    def predictor(img_t):
        with torch.no_grad():
            imgs = img_t.unsqueeze(0).to(DEVICE)
            seg_map = get_seg_map_batch(seg_model, imgs, cls_img_size=max_size)
            probs = ensemble(imgs, seg_map)
            pred = probs.argmax(1).item()
            conf = probs[0, pred].item()
        return pred, conf

    plot_sample_preds_with_predictor(te_ld.dataset, "ensemble", predictor)

    print("Done. All plots saved to artifacts/train/outputs/")


if __name__ == "__main__":
    main()
