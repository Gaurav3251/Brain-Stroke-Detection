import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from brainstroke.models import UNet, SwinUNet, TIMM_AVAILABLE
from brainstroke.training import train_model
from brainstroke.analysis.visualization import plot_training_curves


def main():
    parser = argparse.ArgumentParser(description="Train segmentation model")
    parser.add_argument("--model", choices=["unet", "swin_unet"], required=True)
    args = parser.parse_args()

    if args.model == "unet":
        model = UNet(deep_sup=True)
        name = "unet"
    else:
        if not TIMM_AVAILABLE:
            raise RuntimeError("timm is not installed; Swin-UNet is unavailable")
        model = SwinUNet(pretrained=True, img_size=224)
        name = "swin_unet"

    model, history, *_ = train_model(model, name, task="segment")
    plot_training_curves(history, name)


if __name__ == "__main__":
    main()
