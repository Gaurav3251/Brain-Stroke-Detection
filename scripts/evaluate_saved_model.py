import argparse
import os
import sys

import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    cohen_kappa_score,
    matthews_corrcoef,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import label_binarize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from brainstroke.core.config import DEVICE, METRIC_DIR, REPORT_DIR, CLASSES, NUM_CLASSES
from brainstroke.core.data import build_loaders
from brainstroke.core.utils import load_ckpt
from brainstroke.model_io import build_classifier, build_segmenter, get_input_size, load_champions
from brainstroke.analysis.visualization import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_roc,
    plot_pr,
    plot_confidence_hist,
    plot_sample_preds_with_predictor,
)
from brainstroke.analysis.ensemble_eval import evaluate_ensemble
from brainstroke.models import SegGuidedClassifier, ConfidenceEnsemble
from brainstroke.training import finalize_model_outputs
from brainstroke.training.seg_guided import get_seg_map_batch
from brainstroke.training.trainers import _save_json, _save_text


CLASSIFIER_MODELS = [
    "resnet50", "resnet101",
    "densenet121_se", "densenet201_se",
    "efficientnet_b4_spp", "efficientnetv2_s",
    "mobilenet_v2", "inception_v3", "xception",
    "convnext_small", "swin_unet",
]

SEGMENTATION_MODELS = ["unet", "resunet", "attention_unet", "swin_unet_stroke_only"]


def _load_history_model(model, ckpt_name):
    ckpt_path = os.path.join("artifacts", "train", "checkpoints", f"{ckpt_name}_best.pth")
    model, _, _, history = load_ckpt(ckpt_path, model, device=DEVICE)
    return model.to(DEVICE), history


def _plot_history_if_present(history, model_name):
    if history:
        plot_training_curves(history, model_name)


def _classification_metrics(y_true, y_pred, y_prob):
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
    return metrics


def _save_classification_artifacts(model_name, metrics, cm, y_true, y_pred, y_prob):
    report = classification_report(y_true, y_pred, target_names=CLASSES)
    _save_json(METRIC_DIR / f"{model_name}_metrics.json", metrics)
    _save_json(REPORT_DIR / f"{model_name}_confusion_matrix.json", cm.tolist())
    _save_text(REPORT_DIR / f"{model_name}_classification_report.txt", report)
    plot_confusion_matrix(cm, model_name, out_dir="predictions")
    plot_roc(y_true, y_prob, model_name)
    plot_pr(y_true, y_prob, model_name)
    plot_confidence_hist(y_prob, y_true, model_name)


def _evaluate_seg_guided(base_model_key: str, seg_prior: str):
    base = build_classifier(base_model_key, pretrained=False)
    sg_model = SegGuidedClassifier(base)
    sg_name = f"sg_{base_model_key}"
    sg_model, history = _load_history_model(sg_model, sg_name)
    _plot_history_if_present(history, sg_name)
    seg_model = build_segmenter(seg_prior, pretrained=False)
    seg_model, _ = _load_history_model(seg_model, seg_prior)
    img_size = get_input_size(base_model_key)
    _, _, te_ld, *_ = build_loaders("classify", img_size, False)

    sg_model.eval()
    seg_model.eval()
    all_p, all_l, all_prob = [], [], []
    with torch.no_grad():
        for imgs, labels in te_ld:
            imgs = imgs.to(DEVICE)
            seg_map = get_seg_map_batch(seg_model, imgs, cls_img_size=img_size)
            logits = sg_model(imgs, seg_map)
            probs = torch.softmax(logits, dim=-1)
            all_p.extend(logits.argmax(1).cpu().numpy())
            all_l.extend(labels.numpy())
            all_prob.extend(probs.cpu().numpy())

    y_true = np.array(all_l)
    y_pred = np.array(all_p)
    y_prob = np.array(all_prob)
    cm = confusion_matrix(y_true, y_pred)
    metrics = _classification_metrics(y_true, y_pred, y_prob)
    _save_classification_artifacts(sg_name, metrics, cm, y_true, y_pred, y_prob)

    def predictor(img_t):
        with torch.no_grad():
            img_b = img_t.unsqueeze(0).to(DEVICE)
            seg_map = get_seg_map_batch(seg_model, img_b, cls_img_size=img_size)
            logits = sg_model(img_b, seg_map)
            probs = torch.softmax(logits, dim=-1)[0]
            pred = probs.argmax().item()
            return pred, probs[pred].item()

    plot_sample_preds_with_predictor(te_ld.dataset, sg_name, predictor)


def _build_ensemble(member_keys: list[str], seg_prior: str):
    seg_model = build_segmenter(seg_prior, pretrained=False)
    seg_model, _ = _load_history_model(seg_model, seg_prior)

    models = []
    sizes = []
    for key in member_keys:
        if key == "swin_unet":
            swin = build_classifier("swin_unet", pretrained=False)
            swin, _ = _load_history_model(swin, "swin_unet")
            models.append(("swin_unet", swin, False))
            sizes.append(224)
            continue

        base = build_classifier(key, pretrained=False)
        sg = SegGuidedClassifier(base)
        sg, _ = _load_history_model(sg, f"sg_{key}")
        models.append((f"sg_{key}", sg, True))
        sizes.append(get_input_size(key))

    ensemble = ConfidenceEnsemble(models)
    ensemble, history = _load_history_model(ensemble, "ensemble")
    _plot_history_if_present(history, "ensemble")
    img_size = max(sizes) if sizes else 224
    return ensemble, seg_model, img_size


def _evaluate_ensemble(member_keys: list[str], seg_prior: str):
    ensemble, seg_model, img_size = _build_ensemble(member_keys, seg_prior)
    _, _, te_ld, *_ = build_loaders("classify", img_size=img_size, use_overlay=False)
    metrics, cm, y_true, y_pred, y_prob = evaluate_ensemble(ensemble, seg_model, te_ld, model_name="ensemble")
    _save_classification_artifacts("ensemble", metrics, cm, y_true, y_pred, y_prob)

    def predictor(img_t):
        with torch.no_grad():
            img_b = img_t.unsqueeze(0).to(DEVICE)
            seg_map = get_seg_map_batch(seg_model, img_b, cls_img_size=img_size)
            probs = ensemble(img_b, seg_map)[0]
            pred = probs.argmax().item()
            return pred, probs[pred].item()

    plot_sample_preds_with_predictor(te_ld.dataset, "ensemble", predictor)


def main():
    champs = load_champions()
    default_seg_prior = champs.get("best_segmentation", "unet")

    parser = argparse.ArgumentParser(description="Generate evaluation outputs from a saved checkpoint")
    parser.add_argument(
        "--task",
        required=True,
        choices=["classify", "segment", "seg_guided", "ensemble"],
        help="Model task type for loading the right dataset/evaluation flow",
    )
    parser.add_argument(
        "--model",
        help="Model key. For seg_guided use the base model key, not the sg_ prefix. Not required for ensemble.",
    )
    parser.add_argument(
        "--checkpoint-name",
        help="Saved checkpoint/run name override, for example 'swin_unet_stroke_only'.",
    )
    parser.add_argument(
        "--seg-prior",
        default=default_seg_prior,
        choices=SEGMENTATION_MODELS,
        help="Segmentation prior to use for seg_guided or ensemble reconstruction. Defaults to best_segmentation from champions.json",
    )
    parser.add_argument(
        "--ensemble-members",
        nargs="+",
        help="Base model keys used in the ensemble. Use 'swin_unet' directly if included.",
    )
    args = parser.parse_args()

    if args.task == "classify":
        if args.model not in CLASSIFIER_MODELS:
            raise ValueError(f"--model must be one of: {', '.join(CLASSIFIER_MODELS)}")
        model = build_classifier(args.model, pretrained=False)
        img_size = get_input_size(args.model)
        ckpt_name = args.checkpoint_name or args.model
        model, history = _load_history_model(model, ckpt_name)
        _plot_history_if_present(history, ckpt_name)
        finalize_model_outputs(model, ckpt_name, task=args.task, img_size=img_size)
    elif args.task == "segment":
        if args.model not in SEGMENTATION_MODELS and not args.checkpoint_name:
            raise ValueError(f"--model must be one of: {', '.join(SEGMENTATION_MODELS)}")
        segment_arch = args.model
        if segment_arch is None:
            raise ValueError("--model is required for --task segment")
        if segment_arch not in SEGMENTATION_MODELS:
            raise ValueError(f"--model must be one of: {', '.join(SEGMENTATION_MODELS)}")
        model = build_segmenter(segment_arch, pretrained=False)
        img_size = 224
        ckpt_name = args.checkpoint_name or args.model
        model, history = _load_history_model(model, ckpt_name)
        _plot_history_if_present(history, ckpt_name)
        stroke_only_override = True if "stroke_only" in ckpt_name.lower() else None
        finalize_model_outputs(
            model,
            ckpt_name,
            task=args.task,
            img_size=img_size,
            stroke_only_override=stroke_only_override,
        )
    elif args.task == "seg_guided":
        if args.model not in CLASSIFIER_MODELS or args.model == "swin_unet":
            allowed = ", ".join(m for m in CLASSIFIER_MODELS if m != "swin_unet")
            raise ValueError(f"--model must be one of: {allowed}")
        _evaluate_seg_guided(args.model, args.seg_prior)
    else:
        if not args.ensemble_members:
            raise ValueError("--ensemble-members is required for --task ensemble")
        bad = [m for m in args.ensemble_members if m not in CLASSIFIER_MODELS]
        if bad:
            raise ValueError(f"Invalid ensemble member(s): {', '.join(bad)}")
        _evaluate_ensemble(args.ensemble_members, args.seg_prior)


if __name__ == "__main__":
    main()
