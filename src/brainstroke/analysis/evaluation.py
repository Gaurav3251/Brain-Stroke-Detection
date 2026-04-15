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
from ..models import get_seg_output


@torch.no_grad()
def evaluate_classifier(model, loader, model_name, seg_model=None):
    model.eval()
    if seg_model is not None:
        from ..training.seg_guided import get_seg_map_batch
        seg_model.eval()
    all_p, all_l, all_prob = [], [], []
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        if seg_model is not None:
            seg_map = get_seg_map_batch(seg_model, imgs)
            out = model(imgs, seg_map)
        else:
            out = model(imgs)
        lg = out[0] if isinstance(out, tuple) else out
        all_p.extend(lg.argmax(1).cpu().numpy())
        all_l.extend(labels.numpy())
        all_prob.extend(F.softmax(lg, -1).cpu().numpy())
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

    report = classification_report(y_true, y_pred, target_names=CLASSES)
    print(f"\n{'=' * 55}\n  {model_name} - Test Results\n{'=' * 55}")
    for k, v in metrics.items():
        print(f"  {k:<28} {v:.4f}")
    print(f"\n{report}")
    return metrics, cm, y_true, y_pred, y_prob, report


@torch.no_grad()
def evaluate_segmentation(model, loader, model_name, threshold=0.5):
    model.eval()
    dice_s, iou_s, hd_s, sens_s, spec_s, lesion_counts = [], [], [], [], [], []
    for batch in loader:
        imgs, masks, _ = batch
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        out = model(imgs)
        seg = get_seg_output(out, model)
        preds = torch.sigmoid(seg).cpu().numpy()
        gts = masks.cpu().numpy()
        for pred, gt in zip(preds, gts):
            pb = (pred[0] > threshold).astype(np.float32)
            g = gt[0].astype(np.float32)
            inter = (pb * g).sum()
            union_ = pb.sum() + g.sum()
            dice_s.append((2 * inter + 1) / (union_ + 1))
            iou_s.append((inter + 1) / (union_ - inter + 1))
            try:
                from scipy.spatial.distance import directed_hausdorff
                from scipy.ndimage import binary_erosion as _erosion

                pb_bool = pb.astype(bool)
                g_bool = g.astype(bool)
                pb_bd = pb_bool & ~_erosion(pb_bool, iterations=1)
                g_bd = g_bool & ~_erosion(g_bool, iterations=1)
                if pb_bd.any() and g_bd.any():
                    pp = np.column_stack(np.where(pb_bd))
                    gp = np.column_stack(np.where(g_bd))
                    hd_s.append(max(directed_hausdorff(pp, gp)[0], directed_hausdorff(gp, pp)[0]))
                elif pb.sum() > 0 and g.sum() > 0:
                    pp = np.column_stack(np.where(pb_bool))
                    gp = np.column_stack(np.where(g_bool))
                    hd_s.append(max(directed_hausdorff(pp, gp)[0], directed_hausdorff(gp, pp)[0]))
            except Exception:
                pass

            if g.sum() > 0:
                tp = (pb * g).sum()
                fn = g.sum() - tp
                fp = pb.sum() - tp
                tn = ((1 - pb) * (1 - g)).sum()
                sens_s.append(tp / (tp + fn + 1e-8))
                spec_s.append(tn / (tn + fp + 1e-8))

            from scipy.ndimage import label as scipy_label

            _, n_lesions = scipy_label(pb)
            lesion_counts.append(n_lesions)

    metrics = {
        "mean_dice": np.mean(dice_s),
        "std_dice": np.std(dice_s),
        "mean_iou": np.mean(iou_s),
        "mean_hausdorff": np.mean(hd_s) if hd_s else 0.0,
        "mean_sensitivity": np.mean(sens_s) if sens_s else 0.0,
        "mean_specificity": np.mean(spec_s) if spec_s else 0.0,
        "mean_lesions_per_img": np.mean(lesion_counts),
    }
    print(f"\n{'=' * 55}\n  {model_name} - Segmentation Results\n{'=' * 55}")
    for k, v in metrics.items():
        print(f"  {k:<30} {v:.4f}")
    return metrics
