import argparse
import os
import sys
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from brainstroke.models import SwinUNet, get_classifier
from brainstroke.training import train_model
from brainstroke.analysis.visualization import plot_training_curves
from brainstroke.analysis.evaluation import evaluate_classifier
from brainstroke.core.data import build_loaders
from brainstroke.core.config import REPORT_DIR, METRIC_DIR


def main():
    parser = argparse.ArgumentParser(description="Train classifier")
    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "resnet50", "resnet101",
            "densenet121_se", "densenet201_se",
            "efficientnet_b4_spp", "efficientnetv2_s",
            "mobilenet_v2", "inception_v3", "xception",
            "convnext_small", "swin_unet",
        ],
    )
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    if args.model == "swin_unet":
        model = SwinUNet(pretrained=args.pretrained, img_size=224)
        name = "swin_unet"
        model, history, *_ = train_model(model, name, task="segment", img_size=224, num_epochs=args.epochs)
        # Classification report for SwinUNet (cls head) on classification test set
        _, _, te_ld, *_ = build_loaders("classify", img_size=224, use_overlay=False)
        metrics, cm, y_true, y_pred, y_prob, report = evaluate_classifier(model, te_ld, name)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        METRIC_DIR.mkdir(parents=True, exist_ok=True)
        with open(REPORT_DIR / f"{name}_classification_report.txt", "w", encoding="utf-8") as f:
            f.write(report)
        with open(METRIC_DIR / f"{name}_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        with open(REPORT_DIR / f"{name}_confusion_matrix.json", "w", encoding="utf-8") as f:
            json.dump(cm.tolist(), f, indent=2)
    else:
        model, img_size = get_classifier(args.model, pretrained=args.pretrained)
        name = args.model
        model, history, *_ = train_model(model, name, task="classify", img_size=img_size, num_epochs=args.epochs)

    plot_training_curves(history, name)


if __name__ == "__main__":
    main()
