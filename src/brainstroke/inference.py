import os
from typing import Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .core.config import (
    CLASSES,
    IMG_CLS,
    IMG_SEG,
    DEVICE,
    SEG_THRESHOLD_UNET,
    SEG_THRESHOLD_SWIN,
    INFER_DIR,
)
from .models import (
    SwinUNet,
    SegGuidedClassifier,
    get_seg_output,
)
from .training.seg_guided import get_seg_map_batch
from .core.preprocessing import get_val_aug
# from .analysis.explainability import GradCAMPP, compute_model_attention_stats
from .analysis.explainability import GradCAMPP
from .model_io import get_input_size, load_champions


def _clean_mask(mask: np.ndarray, min_area: int = 120) -> np.ndarray:
    if mask is None:
        return mask
    m = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    clean = np.zeros_like(m, dtype=np.uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 1
    return clean


def _make_seg_overlay(img, mask, color=(255, 100, 0)):
    overlay = img.copy()
    if mask is not None and mask.sum() > 0:
        colored = overlay.copy()
        colored[mask.astype(bool)] = color
        overlay = cv2.addWeighted(overlay, 0.6, colored, 0.4, 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    return overlay


def _primary_result(results, model_choice):
    if model_choice == "ensemble":
        key = "Ensemble"
    else:
        key = model_choice
    if key not in results:
        key = "Ensemble" if "Ensemble" in results else list(results.keys())[0]
    return key, results[key]


def _normalize_input(img_np, size):
    aug = get_val_aug(size)
    norm = aug(image=img_np)["image"]
    tensor = torch.from_numpy(norm.transpose(2, 0, 1).astype(np.float32)).unsqueeze(0)
    return tensor


def predict_single_image(
    img_path: str,
    models: Dict[str, Optional[torch.nn.Module]],
    model_choice: str = "ensemble",
    threshold_unet: float = SEG_THRESHOLD_UNET,
    threshold_swin: float = SEG_THRESHOLD_SWIN,
    cam_threshold: float = 0.65,
    save_outputs: bool = True,
    output_dir=None,
) -> Dict[str, np.ndarray]:
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"File not found: {img_path}")

    out_dir = output_dir or INFER_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    img_raw = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
    img_np = cv2.resize(img_raw, (IMG_CLS, IMG_CLS))

    results = {}

    seg_prior = models.get("seg_prior")
    if seg_prior is None:
        raise RuntimeError("Missing segmentation prior model. Train a segmenter and set champions.json.")

    champs = load_champions()
    sg_keys = [f"sg_{k}" for k in champs.get("group_champions", {}).values()]

    if model_choice in sg_keys and models.get(model_choice) is None:
        raise RuntimeError(f"Model '{model_choice}' is not loaded. Check checkpoints.")

    for sg_key in sg_keys:
        model = models.get(sg_key)
        if model is None:
            continue
        base_key = sg_key[3:]  # "sg_resnet50" -> "resnet50"
        size = get_input_size(base_key)
        tensor = _normalize_input(img_np, size).to(DEVICE)
        seg_map = get_seg_map_batch(seg_prior, tensor, cls_img_size=size)
        model.eval()
        with torch.no_grad():
            probs = F.softmax(model(tensor, seg_map), -1)[0].cpu().numpy()
        results[sg_key] = probs

    swin_model = models.get("swin_unet")

    ensemble_model = models.get("ensemble")
    if model_choice == "ensemble" and ensemble_model is None:
        raise RuntimeError("Ensemble model is not loaded. Check checkpoints and champions.json.")
    if ensemble_model is not None:
        tensor = _normalize_input(img_np, IMG_CLS).to(DEVICE)
        seg_map = get_seg_map_batch(seg_prior, tensor)
        ensemble_model.eval()
        with torch.no_grad():
            ens_probs = ensemble_model(tensor, seg_map)[0].cpu().numpy()
        results["Ensemble"] = ens_probs

    primary_key, primary_probs = _primary_result(results, model_choice)
    primary_pred_idx = int(np.argmax(primary_probs))
    primary_pred_label = CLASSES[primary_pred_idx]

    seg_prior.eval()
    with torch.no_grad():
        seg_input_size = getattr(seg_prior, "input_size", IMG_SEG)
        seg_in = _normalize_input(img_np, seg_input_size).to(DEVICE)
        seg_out = seg_prior(seg_in)
        seg_prob = torch.sigmoid(get_seg_output(seg_out, seg_prior)).squeeze().cpu().numpy()
        seg_prob = cv2.resize(seg_prob, (IMG_CLS, IMG_CLS), interpolation=cv2.INTER_LINEAR)
        seg_mask = (seg_prob > threshold_unet).astype(np.uint8)
        seg_mask = _clean_mask(seg_mask, min_area=120)

    swin_mask = None
    if swin_model is not None:
        with torch.no_grad():
            tensor = _normalize_input(img_np, IMG_CLS).to(DEVICE)
            swin_out = swin_model(tensor)
            swin_prob = torch.sigmoid(get_seg_output(swin_out, swin_model)).squeeze().cpu().numpy()
            swin_prob = cv2.resize(swin_prob, (IMG_CLS, IMG_CLS), interpolation=cv2.INTER_LINEAR)
            swin_mask = (swin_prob > threshold_swin).astype(np.uint8)
            swin_mask = _clean_mask(swin_mask, min_area=120)

    # Segmentation priors were trained on stroke masks; suppress display for Normal predictions.
    if primary_pred_label == "Normal":
        seg_mask = np.zeros_like(seg_mask, dtype=np.uint8)
        if swin_mask is not None:
            swin_mask = np.zeros_like(swin_mask, dtype=np.uint8)

    seg_overlay = _make_seg_overlay(img_np, seg_mask, color=(255, 80, 0))
    swin_overlay = _make_seg_overlay(img_np, swin_mask, color=(80, 80, 255)) if swin_mask is not None else None

    cam_model = None
    cam_layer = None
    if model_choice == "swin_unet":
        cam_model = None
    elif model_choice == "ensemble" and ensemble_model is not None:
        with torch.no_grad():
            weights = F.softmax(ensemble_model.logit_w, dim=0).cpu().numpy()
        top_idx = int(weights.argmax()) if len(weights) > 0 else 0
        base_model = ensemble_model.models[top_idx]
        if isinstance(base_model, SegGuidedClassifier):
            cam_model = base_model
            cam_layer = base_model.cam_layer()
        elif isinstance(base_model, SwinUNet):
            cam_model = None
        else:
            cam_model = base_model
            cam_layer = getattr(base_model, "cam_layer", lambda: None)()
    elif model_choice.startswith("sg_"):
        chosen = models.get(model_choice)
        if chosen is not None and isinstance(chosen, SegGuidedClassifier):
            cam_layer = chosen.cam_layer()
            if cam_layer is not None:
                # Wrap so GradCAM forward pass includes seg_map
                class _SgWrap(torch.nn.Module):
                    def __init__(self, sg, sp, sz):
                        super().__init__()
                        self.sg, self.sp, self.sz = sg, sp, sz
                    def forward(self, x):
                        sm = get_seg_map_batch(self.sp, x, cls_img_size=self.sz)
                        return self.sg(x, sm)
                    def eval(self):
                        self.sg.eval(); self.sp.eval(); return self
                    def zero_grad(self, set_to_none=True):
                        self.sg.zero_grad(set_to_none=set_to_none)
                base_key = model_choice[3:]
                cam_model = _SgWrap(chosen, seg_prior, get_input_size(base_key))
    else:
        chosen = models.get(model_choice)
        if chosen is not None:
            cam_model = chosen
            cam_layer = getattr(chosen, "cam_layer", lambda: None)()

    if cam_model is not None and cam_layer is not None:
        size = getattr(cam_model, "input_size", IMG_CLS)
        tensor = _normalize_input(img_np, size)
        gcpp = GradCAMPP(cam_model, cam_layer)
        cam, tc = gcpp.generate(tensor)
        overlay_bound = gcpp.overlay_with_boundary(cam, img_np)
        # overlay_bound = gcpp.overlay_with_boundary(cam, img_np, threshold=cam_threshold, seg_mask=seg_mask)
        # _, _, attention_pct_cam = compute_model_attention_stats(cam, img_np, IMG_CLS, cam_threshold)
    else:
        tc = int(np.argmax(_primary_result(results, model_choice)[1]))
        overlay_bound = img_np.copy()
        # tc = int(np.argmax(_primary_result(results, model_choice)[1]))
        # overlay_bound = img_np.copy()
        # attention_pct_cam = 0.0

    seg_stroke_px = int(seg_mask.sum())
    total_px = seg_mask.size
    seg_coverage = seg_stroke_px / total_px * 100.0
    swin_stroke_px = int(swin_mask.sum()) if swin_mask is not None else 0
    swin_coverage = swin_stroke_px / total_px * 100.0 if swin_mask is not None else 0.0

    if save_outputs:
        if model_choice == "all":
            results_plot = results
        else:
            key, probs = _primary_result(results, model_choice)
            results_plot = {key: probs}

        n_models = len(results_plot)
        fig1, axes1 = plt.subplots(1, n_models + 1, figsize=(5 * (n_models + 1), 4))
        fig1.suptitle(
            f"Stroke Detection - {os.path.basename(img_path)}\n"
            f"Primary: {_primary_result(results, model_choice)[0]}  "
            f"({_primary_result(results, model_choice)[1].max():.1%} confidence)",
            fontsize=12,
            fontweight="bold",
        )
        axes1[0].imshow(img_np)
        axes1[0].set_title("Input CT", fontweight="bold")
        axes1[0].axis("off")
        colors_cls = ["#2563EB", "#DC2626", "#16A34A"]
        for col, (name, probs) in enumerate(results_plot.items(), 1):
            ax = axes1[col]
            pred_idx = probs.argmax()
            bars = ax.barh(CLASSES, probs, color=colors_cls, edgecolor="white", height=0.6)
            title_color = "#DC2626" if pred_idx > 0 else "#16A34A"
            ax.set_title(
                f"{name}\n{CLASSES[pred_idx]}  ({probs[pred_idx]:.1%})",
                fontweight="bold",
                fontsize=9,
                color=title_color,
            )
            ax.set_xlim(0, 1.05)
            ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8)
            for bar, prob in zip(bars, probs):
                ax.text(min(prob + 0.02, 0.98), bar.get_y() + bar.get_height() / 2,
                        f"{prob:.3f}", va="center", fontsize=8)
            ax.set_xlabel("Probability", fontsize=8)
        plt.tight_layout(rect=[0, 0, 1, 0.88])
        out1 = os.path.join(out_dir, f"inference_{os.path.splitext(os.path.basename(img_path))[0]}_classification.png")
        plt.savefig(out1, dpi=150, bbox_inches="tight")
        plt.close()

        stem = os.path.splitext(os.path.basename(img_path))[0]
        out_seg = os.path.join(out_dir, f"inference_{stem}_segmentation.png")
        if primary_pred_label != "Normal":
            seg_cols = 3 if swin_overlay is not None else 2
            fig_seg, axes_seg = plt.subplots(1, seg_cols, figsize=(5 * seg_cols, 4))
            if not isinstance(axes_seg, np.ndarray):
                axes_seg = np.array([axes_seg])
            fig_seg.suptitle(
                f"Segmentation - {os.path.basename(img_path)} | Prior Coverage: {seg_coverage:.1f}%",
                fontsize=11,
                fontweight="bold",
            )
            axes_seg[0].imshow(img_np)
            axes_seg[0].set_title("Original CT")
            axes_seg[0].axis("off")
            axes_seg[1].imshow(seg_overlay)
            axes_seg[1].set_title(f"Seg Prior Overlay\nCoverage: {seg_coverage:.1f}%")
            axes_seg[1].axis("off")
            if swin_overlay is not None:
                axes_seg[2].imshow(swin_overlay)
                axes_seg[2].set_title(f"Swin Overlay\nCoverage: {swin_coverage:.1f}%")
                axes_seg[2].axis("off")
            plt.tight_layout(rect=[0, 0, 1, 0.88])
            plt.savefig(out_seg, dpi=150, bbox_inches="tight")
            plt.close()
        elif os.path.exists(out_seg):
            os.remove(out_seg)

        if primary_pred_label != "Normal":
            visual_panels = [
                ("Input CT", img_np),
                (f"Segmentation Output\nPrior Coverage: {seg_coverage:.1f}%", seg_overlay),
                ("GradCAM++ Output", overlay_bound),
                # (f"GradCAM++ Output\nModel Attention: {attention_pct_cam:.1f}%", overlay_bound),
            ]
            fig_width = 15
        else:
            visual_panels = [
                ("Input CT", img_np),
                ("GradCAM++ Output", overlay_bound),
                # (f"GradCAM++ Output\nModel Attention: {attention_pct_cam:.1f}%", overlay_bound),
            ]
            fig_width = 10  # 2 panels at same per-panel width as 3-panel stroke output (15/3 * 2)

        fig2, axes2 = plt.subplots(1, len(visual_panels), figsize=(fig_width, 4))

        if not isinstance(axes2, np.ndarray):
            axes2 = np.array([axes2])
        fig2.suptitle(
            f"Visual Review - {os.path.basename(img_path)} | Model Output: {primary_key} | Prediction: {primary_pred_label}",
            fontsize=10,
            fontweight="bold",
        )
        for ax, (title, image) in zip(axes2, visual_panels):
            ax.imshow(image)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        plt.tight_layout(rect=[0, 0, 1, 0.86])
        out2 = os.path.join(out_dir, f"inference_{stem}_explainability.png")
        plt.savefig(out2, dpi=150, bbox_inches="tight")
        plt.close()

        key, primary_probs = _primary_result(results, model_choice)
        pred_label = CLASSES[primary_probs.argmax()]
        pred_color = "#DC2626" if primary_probs.argmax() > 0 else "#16A34A"
        conf_val = primary_probs.max()

        fig3, ax3 = plt.subplots(figsize=(6, 5))
        ax3.axis("off")
        lines = [
            ("STROKE DETECTION REPORT", 0.95, 11, "#1a1a1a", True),
            (f"File: {os.path.basename(img_path)}", 0.87, 8, "gray", False),
            ("-" * 38, 0.83, 9, "#cccccc", False),
            (f"Prediction:   {pred_label}", 0.76, 12, pred_color, True),
            (f"Confidence:   {conf_val:.1%}", 0.68, 11, "#1a1a1a", False),
            ("-" * 38, 0.62, 9, "#cccccc", False),
            ("Segmentation Coverage", 0.56, 10, "#1a1a1a", True),
            (f"  Prior: {seg_coverage:.1f}%  ({seg_stroke_px:,} px)", 0.49, 10, "#EA580C", False),
            (f"  Swin:  {swin_coverage:.1f}%  ({swin_stroke_px:,} px)", 0.42, 10, "#2563EB", False),
            ("-" * 38, 0.36, 9, "#cccccc", False),
            ("GradCAM = model attention region", 0.30, 7, "gray", False),
            ("Seg models = lesion area", 0.24, 7, "gray", False),
        ]
        for text, y, size, color, bold in lines:
            ax3.text(0.05, y, text, fontsize=size, color=color,
                     fontweight="bold" if bold else "normal",
                     transform=ax3.transAxes, va="top")
        fig3.patch.set_facecolor("#f9f9f9")
        plt.tight_layout()
        out3 = os.path.join(out_dir, f"inference_{os.path.splitext(os.path.basename(img_path))[0]}_metrics.png")
        plt.savefig(out3, dpi=150, bbox_inches="tight")
        plt.close()

    return results
