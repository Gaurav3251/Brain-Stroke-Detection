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
    DenseNet121,
    EfficientNetB4,
    UNet,
    SwinUNet,
    TIMM_AVAILABLE,
    SegGuidedDenseNet,
    SegGuidedEfficientNet,
    ConfidenceEnsemble,
    get_seg_output,
)
from .training.seg_guided import get_seg_map_batch
from .core.preprocessing import get_val_aug
from .analysis.explainability import GradCAMPP, compute_damage_stats
from .model_io import load_models


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
    map_key = {
        "ensemble": "Ensemble",
        "densenet121": "DenseNet121",
        "efficientnet_b4": "EfficientNetB4",
        "swin_unet": "SwinUNet",
    }
    key = map_key.get(model_choice, "Ensemble")
    if key not in results:
        key = "Ensemble"
    return key, results[key]


def predict_single_image(
    img_path: str,
    models: Dict[str, Optional[torch.nn.Module]],
    model_choice: str = "ensemble",
    threshold_unet: float = SEG_THRESHOLD_UNET,
    threshold_swin: float = SEG_THRESHOLD_SWIN,
    cam_threshold: float = 0.65,
    save_outputs: bool = True,
) -> Dict[str, np.ndarray]:
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"File not found: {img_path}")

    required = ["densenet121", "efficientnet_b4", "unet", "ensemble", "sg_densenet121"]
    for name in required:
        if models.get(name) is None:
            raise RuntimeError(f"Missing model: {name}. Load checkpoints first.")

    INFER_DIR.mkdir(parents=True, exist_ok=True)

    img_np = np.array(Image.open(img_path).convert("RGB").resize((IMG_CLS, IMG_CLS)), dtype=np.uint8)
    aug = get_val_aug(IMG_CLS)
    norm = aug(image=img_np)["image"]
    tensor = torch.from_numpy(norm.transpose(2, 0, 1).astype(np.float32)).unsqueeze(0)

    results = {}
    dn_model = models["densenet121"]
    eff_model = models["efficientnet_b4"]
    unet_model = models["unet"]
    sg_dense = models["sg_densenet121"]
    ensemble_model = models["ensemble"]
    swin_model = models.get("swin_unet")

    for name, model in [("DenseNet121", dn_model), ("EfficientNetB4", eff_model)]:
        model.eval()
        with torch.no_grad():
            probs = F.softmax(model(tensor.to(DEVICE)), -1)[0].cpu().numpy()
        results[name] = probs

    seg_map = get_seg_map_batch(unet_model, tensor.to(DEVICE))
    ensemble_model.eval()
    with torch.no_grad():
        ens_probs = ensemble_model(tensor.to(DEVICE), seg_map)[0].cpu().numpy()
    results["Ensemble"] = ens_probs

    if swin_model is not None:
        swin_model.eval()
        with torch.no_grad():
            swin_out = swin_model(tensor.to(DEVICE))
            swin_cls_probs = F.softmax(swin_out[0], dim=-1)[0].cpu().numpy()
        results["SwinUNet"] = swin_cls_probs

    unet_model.eval()
    with torch.no_grad():
        seg_in_unet = F.interpolate(tensor, (IMG_SEG, IMG_SEG), mode="bilinear", align_corners=False)
        unet_out = unet_model(seg_in_unet.to(DEVICE))
        unet_prob = torch.sigmoid(get_seg_output(unet_out, unet_model)).squeeze().cpu().numpy()
        unet_prob = cv2.resize(unet_prob, (IMG_CLS, IMG_CLS), interpolation=cv2.INTER_LINEAR)
        unet_mask = (unet_prob > threshold_unet).astype(np.uint8)

    swin_mask = None
    if swin_model is not None:
        with torch.no_grad():
            swin_out = swin_model(tensor.to(DEVICE))
            swin_prob = torch.sigmoid(get_seg_output(swin_out, swin_model)).squeeze().cpu().numpy()
            swin_prob = cv2.resize(swin_prob, (IMG_CLS, IMG_CLS), interpolation=cv2.INTER_LINEAR)
            swin_mask = (swin_prob > threshold_swin).astype(np.uint8)

    _ = _make_seg_overlay(img_np, unet_mask, color=(255, 80, 0))
    _ = _make_seg_overlay(img_np, swin_mask, color=(80, 80, 255)) if swin_mask is not None else img_np.copy()

    seg_map_single = get_seg_map_batch(unet_model, tensor.to(DEVICE))

    class _SGWrap(torch.nn.Module):
        def __init__(self, m, s):
            super().__init__()
            self.model = m
            self.seg_map = s
        def forward(self, x):
            return self.model(x, self.seg_map)
        @property
        def features(self):
            return self.model.features

    wrapped = _SGWrap(sg_dense, seg_map_single)

    if model_choice == "densenet121":
        cam_model = dn_model
        cam_layer = dn_model.cam_layer()
    elif model_choice == "efficientnet_b4":
        cam_model = eff_model
        cam_layer = eff_model.cam_layer()
    else:
        cam_model = wrapped
        cam_layer = sg_dense.cam_layer()

    gcpp = GradCAMPP(cam_model, cam_layer)
    cam, tc = gcpp.generate(tensor)
    overlay_bound = gcpp.overlay_with_boundary(cam, img_np, threshold=cam_threshold)
    _, _, dmg_pct_cam = compute_damage_stats(cam, img_np, IMG_CLS, cam_threshold)

    unet_stroke_px = int(unet_mask.sum())
    total_px = unet_mask.size
    unet_coverage = unet_stroke_px / total_px * 100.0
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
        out1 = os.path.join(INFER_DIR, f"inference_{os.path.splitext(os.path.basename(img_path))[0]}_classification.png")
        plt.savefig(out1, dpi=150, bbox_inches="tight")
        plt.close()

        pred_name = CLASSES[tc]
        pred_color = "#DC2626" if tc > 0 else "#16A34A"
        fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))
        fig2.suptitle(
            f"GradCAM++ - {os.path.basename(img_path)} | Primary: {pred_name} | Affected: {dmg_pct_cam:.1f}%",
            fontsize=11,
            fontweight="bold",
        )
        axes2[0].imshow(img_np)
        axes2[0].set_title("Original CT")
        axes2[0].axis("off")
        axes2[1].imshow(overlay_bound)
        axes2[1].set_title(f"GradCAM++ | {pred_name}\nAffected: {dmg_pct_cam:.1f}%")
        axes2[1].axis("off")
        axes2[2].axis("off")
        axes2[2].text(0.1, 0.88, "GradCAM", fontsize=11, fontweight="bold",
                      transform=axes2[2].transAxes)
        axes2[2].text(0.1, 0.73, pred_name, fontsize=14, color=pred_color,
                      fontweight="bold", transform=axes2[2].transAxes)
        axes2[2].text(0.1, 0.55, "GradCAM affected area:", fontsize=9,
                      transform=axes2[2].transAxes)
        axes2[2].text(0.1, 0.43, f"{dmg_pct_cam:.1f}% of brain", fontsize=11,
                      color="#CA8A04", fontweight="bold", transform=axes2[2].transAxes)
        axes2[2].text(0.1, 0.25, "Green boundary = highest\nattention region in brain",
                      fontsize=8, color="gray", transform=axes2[2].transAxes)
        plt.tight_layout(rect=[0, 0, 1, 0.88])
        out2 = os.path.join(INFER_DIR, f"inference_{os.path.splitext(os.path.basename(img_path))[0]}_explainability.png")
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
            ("GradCAM++ Region", 0.56, 10, "#92400E", True),
            (f"  Affected area: {dmg_pct_cam:.1f}%", 0.49, 10, "#CA8A04", False),
            ("-" * 38, 0.43, 9, "#cccccc", False),
            ("Segmentation Coverage", 0.37, 10, "#1a1a1a", True),
            (f"  U-Net: {unet_coverage:.1f}%  ({unet_stroke_px:,} px)", 0.30, 10, "#EA580C", False),
            (f"  Swin:  {swin_coverage:.1f}%  ({swin_stroke_px:,} px)", 0.23, 10, "#2563EB", False),
            ("-" * 38, 0.17, 9, "#cccccc", False),
            ("GradCAM = attention region", 0.11, 7, "gray", False),
            ("U-Net/Swin = detected lesion area", 0.05, 7, "gray", False),
        ]
        for text, y, size, color, bold in lines:
            ax3.text(0.05, y, text, fontsize=size, color=color,
                     fontweight="bold" if bold else "normal",
                     transform=ax3.transAxes, va="top")
        fig3.patch.set_facecolor("#f9f9f9")
        plt.tight_layout()
        out3 = os.path.join(INFER_DIR, f"inference_{os.path.splitext(os.path.basename(img_path))[0]}_metrics.png")
        plt.savefig(out3, dpi=150, bbox_inches="tight")
        plt.close()

    return results
