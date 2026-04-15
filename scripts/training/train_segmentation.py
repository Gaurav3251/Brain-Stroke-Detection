import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from brainstroke.models import UNet, ResUNet, AttentionUNet, SwinUNet
from brainstroke.training import train_model
from brainstroke.analysis.visualization import plot_training_curves


def main():
    parser = argparse.ArgumentParser(description="Train segmenter")
    parser.add_argument(
        "--model",
        choices=["unet", "resunet", "attention_unet", "swin_unet"],
        required=True,
    )
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained encoder weights where applicable")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--stroke-only", action="store_true", help="Train on stroke slices only, including for SwinUNet")
    parser.add_argument("--run-name", help="Custom artifact/checkpoint name to avoid overwriting another run")
    args = parser.parse_args()

    if args.model == "unet":
        model = UNet(deep_sup=True)
    elif args.model == "resunet":
        model = ResUNet(pretrained=args.pretrained)
    elif args.model == "attention_unet":
        model = AttentionUNet()
    else:
        model = SwinUNet(pretrained=args.pretrained, img_size=224)

    run_name = args.run_name or args.model
    model, history, *_ = train_model(
        model,
        run_name,
        task="segment",
        img_size=224,
        num_epochs=args.epochs,
        stroke_only_override=True if args.stroke_only else None,
    )
    plot_training_curves(history, run_name)


if __name__ == "__main__":
    main()
