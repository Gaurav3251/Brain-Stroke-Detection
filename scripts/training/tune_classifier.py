import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from brainstroke.training.tuning import tune_classifier


def main():
    parser = argparse.ArgumentParser(description="Lightweight hyperparameter tuning for classifiers")
    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "resnet50", "resnet101",
            "densenet121_se", "densenet201_se",
            "efficientnet_b4_spp", "efficientnetv2_s",
            "mobilenet_v2", "inception_v3", "xception",
            "convnext_small",
        ],
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()

    tune_classifier(args.model, pretrained=args.pretrained, epochs=args.epochs)


if __name__ == "__main__":
    main()
