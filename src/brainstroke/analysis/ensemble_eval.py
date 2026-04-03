import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, cohen_kappa_score, matthews_corrcoef,
    f1_score, precision_score, recall_score
)
from sklearn.preprocessing import label_binarize

from ..core.config import CLASSES, NUM_CLASSES, DEVICE
from .training.seg_guided import get_seg_map_batch


@torch.no_grad()
def evaluate_ensemble(ensemble_model, unet_model, loader, model_name="ensemble"):
    ensemble_model.eval()
    unet_model.eval()
    all_p, all_l, all_prob = [], [], []
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        seg_map = get_seg_map_batch(unet_model, imgs)
        probs = ensemble_model(imgs, seg_map)
        preds = probs.argmax(1)
        all_p.extend(preds.cpu().numpy())
        all_l.extend(labels.cpu().numpy())
        all_prob.extend(probs.cpu().numpy())

    y_true = np.array(all_l)
    y_pred = np.array(all_p)
    y_prob = np.array(all_prob)
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "accuracy": (y_true == y_pred).mean(),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "matthews_corr": matthews_corrcoef(y_true, y_pred),
    }
    try:
        metrics["macro_auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        metrics["weighted_auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
    except Exception:
        metrics["macro_auc"] = 0.0
        metrics["weighted_auc"] = 0.0

    y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    for i, cls in enumerate(CLASSES):
        metrics[f"{cls}_AP"] = average_precision_score(y_bin[:, i], y_prob[:, i])
        metrics[f"{cls}_f1"] = f1_score(y_true == i, y_pred == i, zero_division=0)

    print(f"\n{'=' * 55}\n  {model_name} - Test Results\n{'=' * 55}")
    for k, v in metrics.items():
        print(f"  {k:<28} {v:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=CLASSES)}")
    return metrics, cm, y_true, y_pred, y_prob
