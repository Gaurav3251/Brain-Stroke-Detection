import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..core.config import CLASSES, IMG_CLS, GCAM_DIR, DEVICE
from ..core.preprocessing import get_val_aug


class GradCAMPP:
    def __init__(self, model, target_layer):
        self.model = model
        self.acts = None
        self.grads = None
        target_layer.register_forward_hook(lambda m, i, o: setattr(self, "acts", o.detach()))
        target_layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "grads", go[0].detach()))

    def generate(self, tensor, target_cls=None):
        self.model.eval()
        tensor = tensor.to(DEVICE)
        tensor.requires_grad_(True)
        out = self.model(tensor)
        if target_cls is None:
            target_cls = out.argmax(1).item()
        self.model.zero_grad()
        oh = torch.zeros_like(out)
        oh[0, target_cls] = 1.0
        out.backward(gradient=oh)
        g = self.grads
        a = self.acts
        g2 = g ** 2
        g3 = g ** 3
        denom = 2 * g2 + (g3 * a).sum(dim=(2, 3), keepdim=True)
        denom = torch.where(denom != 0, denom, torch.ones_like(denom))
        alpha = g2 / denom
        w = (alpha * F.relu(g)).sum(dim=(2, 3), keepdim=True)
        cam = F.relu((w * a).sum(dim=1, keepdim=True)).squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, target_cls

    def overlay_with_boundary(self, cam, img_np, alpha=0.4, threshold=0.5, min_contour_area=200, draw_boundary=True, seg_mask=None):
        h, w = img_np.shape[:2]
        cam_r = cv2.resize(cam, (w, h))
        cam_r = cv2.GaussianBlur(cam_r, (21, 21), 0)
        cam_r = (cam_r - cam_r.min()) / (cam_r.max() - cam_r.min() + 1e-8)

        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        brain_mask = (gray > 10).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel)

        # Use seg mask to focus boundary only (not the heatmap)
        if seg_mask is not None:
            seg_r = cv2.resize(seg_mask.astype(np.float32), (w, h))
            cam_boundary = cam_r * seg_r
        else:
            cam_boundary = cam_r

        binary = (cam_boundary > threshold).astype(np.uint8)
        binary = binary * brain_mask
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]

        # Heatmap overlay uses raw GradCAM (unmasked)
        hm = cv2.applyColorMap((cam_r * 255).astype(np.uint8), cv2.COLORMAP_JET)
        hm_rgb = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
        result = (alpha * hm_rgb + (1 - alpha) * img_np).astype(np.uint8)

        if draw_boundary and contours:
            cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
        return result


def compute_model_attention_stats(cam, img_np, img_size, cam_threshold=0.5, min_contour_area=200):
    cam_resized = cv2.resize(cam, (img_size, img_size))
    cam_resized = cv2.GaussianBlur(cam_resized, (21, 21), 0)
    cam_norm = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8)

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    brain_mask = (gray > 10).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel)

    binary_mask = (cam_norm > cam_threshold).astype(np.uint8)
    binary_mask = binary_mask * brain_mask

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(binary_mask)
    for c in contours:
        if cv2.contourArea(c) > min_contour_area:
            cv2.drawContours(clean_mask, [c], -1, 1, -1)

    brain_px = max(brain_mask.sum(), 1)
    attention_px = binary_mask.sum()
    attention_pct = (attention_px / brain_px) * 100.0
    return binary_mask, int(attention_px), attention_pct


def save_explanation(img_path, model, cam_layer, model_name, cam_threshold=0.5):
    if not os.path.exists(img_path):
        return
    img_np = np.array(Image.open(img_path).convert("RGB").resize((IMG_CLS, IMG_CLS)), dtype=np.uint8)
    aug = get_val_aug(IMG_CLS)
    tensor = torch.from_numpy(
        aug(image=img_np)["image"].transpose(2, 0, 1).astype(np.float32)
    ).unsqueeze(0)
    gcpp = GradCAMPP(model, cam_layer)
    cam, tc = gcpp.generate(tensor)
    overlay = gcpp.overlay_with_boundary(cam, img_np, threshold=cam_threshold)
    _, _, attention_pct = compute_model_attention_stats(cam, img_np, IMG_CLS, cam_threshold)
    pred_name = CLASSES[tc]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img_np)
    axes[0].set_title("Original CT")
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title(f"GradCAM++ | {pred_name}\nModel Attention Area: {attention_pct:.1f}%")
    axes[1].axis("off")
    plt.tight_layout()
    os.makedirs(GCAM_DIR, exist_ok=True)
    out = os.path.join(GCAM_DIR, f"{model_name}_{os.path.splitext(os.path.basename(img_path))[0]}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[GradCAM] -> {out}")
