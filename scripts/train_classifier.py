import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from brainstroke.models import DenseNet121, EfficientNetB4
from brainstroke.training import train_model
from brainstroke.analysis.visualization import plot_training_curves


def main():
    parser = argparse.ArgumentParser(description="Train classifier")
    parser.add_argument("--model", choices=["densenet121", "efficientnet_b4"], required=True)
    args = parser.parse_args()

    if args.model == "densenet121":
        model = DenseNet121(pretrained=True)
        name = "densenet121"
    else:
        model = EfficientNetB4(pretrained=True)
        name = "efficientnet_b4"

    model, history, *_ = train_model(model, name, task="classify")
    plot_training_curves(history, name)


if __name__ == "__main__":
    main()
