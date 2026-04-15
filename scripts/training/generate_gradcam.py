"""Generate sample GradCAM visualizations for all classification models + ensemble.

Usage:
    python scripts/training/generate_gradcam.py
    python scripts/training/generate_gradcam.py --n 12
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from brainstroke.core.config import CLASSES, DEVICE, GCAM_DIR, ensure_dirs
from brainstroke.core.data import build_loaders
from brainstroke.model_io import (
    load_champions, load_ensemble,
    build_classifier, build_segmenter, load_model_checkpoint, get_input_size,
)
from brainstroke.models import SegGuidedClassifier
from brainstroke.training.seg_guided import get_seg_map_batch
from brainstroke.analysis.explainability import GradCAMPP

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


class _SegGuidedWrapper(nn.Module):
    """Wraps a seg-guided model so GradCAMPP can call forward without passing seg_map."""
    def __init__(self, sg_model, seg_model, cls_img_size):
        super().__init__()
        self.sg_model = sg_model
        self.seg_model = seg_model
        self.cls_img_size = cls_img_size

    def forward(self, x):
        seg_map = get_seg_map_batch(self.seg_model, x, cls_img_size=self.cls_img_size)
        return self.sg_model(x, seg_map)

    def eval(self):
        self.sg_model.eval()
        self.seg_model.eval()
        return self

    def zero_grad(self, set_to_none=True):
        self.sg_model.zero_grad(set_to_none=set_to_none)


def _to_display(img_tensor):
    """Convert normalized tensor back to displayable RGB."""
    img = img_tensor.permute(1, 2, 0).numpy()
    return np.clip(img * STD + MEAN, 0, 1)


def generate_gradcam_grid(model, cam_layer, dataset, model_name, seg_model=None, img_size=224, n=9, group_size=3):
    """Generate and save GradCAM overlays in groups of `group_size` images each."""
    idx = np.random.choice(len(dataset), min(n, len(dataset)), replace=False)
    GCAM_DIR.mkdir(parents=True, exist_ok=True)
    cols = 3  # Original | GradCAM overlay | Attention map

    groups = [idx[i:i + group_size] for i in range(0, len(idx), group_size)]
    for g_num, group_idx in enumerate(groups, 1):
        rows = len(group_idx)
        fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
        if rows == 1:
            axes = axes[np.newaxis, :]

        for i, j in enumerate(group_idx):
            img_t, lbl = dataset[j]
            lbl_idx = int(lbl.item()) if hasattr(lbl, "item") else int(lbl)
            img_display = _to_display(img_t)
            img_uint8 = (img_display * 255).astype(np.uint8)

            # Generate seg mask for boundary focusing
            seg_mask_np = None
            if seg_model is not None:
                with torch.no_grad():
                    t = img_t.unsqueeze(0).to(DEVICE)
                    sm = get_seg_map_batch(seg_model, t, cls_img_size=img_size)
                    seg_mask_np = (sm.squeeze().cpu().numpy() > 0.5).astype(np.float32)

            gcpp = GradCAMPP(model, cam_layer)
            tensor = img_t.unsqueeze(0)
            cam, tc = gcpp.generate(tensor)
            overlay = gcpp.overlay_with_boundary(cam, img_uint8, threshold=0.5, seg_mask=seg_mask_np)

            correct = tc == lbl_idx
            color = "green" if correct else "red"

            axes[i, 0].imshow(img_display)
            axes[i, 0].set_title(f"GT: {CLASSES[lbl_idx]}", fontsize=9)
            axes[i, 0].axis("off")

            axes[i, 1].imshow(overlay)
            axes[i, 1].set_title(
                f"Pred: {CLASSES[tc]} ({'correct' if correct else 'wrong'})",
                fontsize=9, color=color,
            )
            axes[i, 1].axis("off")

            cam_resized = cv2.resize(cam, (img_uint8.shape[1], img_uint8.shape[0]))
            axes[i, 2].imshow(cam_resized, cmap="jet")
            axes[i, 2].set_title("Attention Map", fontsize=9)
            axes[i, 2].axis("off")

        plt.suptitle(f"{model_name} - GradCAM++ ({g_num}/{len(groups)})", fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        out_path = GCAM_DIR / f"{model_name}_gradcam_{g_num}.png"
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[GradCAM] -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate GradCAM samples for all models + ensemble")
    parser.add_argument("--n", type=int, default=9, help="Number of sample images per model")
    args = parser.parse_args()

    ensure_dirs()
    np.random.seed(42)

    champs = load_champions()
    best_seg = champs.get("best_segmentation", "unet")

    seg_model = build_segmenter(best_seg, pretrained=False)
    seg_model = load_model_checkpoint(seg_model, best_seg)
    seg_model.to(DEVICE).eval()

    # --- seg-guided champion classifiers ---
    group_champs = champs.get("group_champions", {})
    for grp, key in group_champs.items():
        sg_name = f"sg_{key}"
        print(f"\n--- {sg_name} (Group {grp}) ---")
        try:
            base = build_classifier(key, pretrained=False)
            sg = SegGuidedClassifier(base)
            sg = load_model_checkpoint(sg, sg_name)
        except Exception as e:
            print(f"  Skipping {sg_name}: {e}")
            continue

        cam_layer = sg.cam_layer()
        if cam_layer is None:
            print(f"  Skipping {sg_name}: no cam_layer")
            continue

        img_size = get_input_size(key)
        _, _, te_ld, *_ = build_loaders("classify", img_size, False)
        wrapper = _SegGuidedWrapper(sg, seg_model, img_size)
        generate_gradcam_grid(wrapper, cam_layer, te_ld.dataset, sg_name, seg_model=seg_model, img_size=img_size, n=args.n)

    # --- ensemble (uses top-weighted member with seg guidance) ---
    print("\n--- Ensemble ---")
    try:
        ensemble, seg_model_ens = load_ensemble()
        ensemble.eval()
        seg_model_ens.eval()
    except Exception as e:
        print(f"  Skipping ensemble: {e}")
        return

    weights = F.softmax(ensemble.logit_w, dim=0).detach().cpu().numpy()
    top_idx = int(weights.argmax())
    top_name = ensemble.model_names[top_idx]
    top_model = ensemble.models[top_idx]
    print(f"  Top member: {top_name} (weight={weights[top_idx]:.3f})")

    if isinstance(top_model, SegGuidedClassifier):
        cam_layer = top_model.base.cam_layer()
        img_size = getattr(top_model, "input_size", 224)
        cam_model = _SegGuidedWrapper(top_model, seg_model_ens, img_size)
    else:
        cam_layer = getattr(top_model, "cam_layer", lambda: None)()
        cam_model = top_model

    if cam_layer is not None:
        _, _, te_ld, *_ = build_loaders("classify", 224, False)
        generate_gradcam_grid(cam_model, cam_layer, te_ld.dataset, "ensemble", seg_model=seg_model_ens, img_size=224, n=args.n)
    else:
        print(f"  Skipping ensemble: top member {top_name} has no cam_layer")

    print("\nDone.")


if __name__ == "__main__":
    main()
