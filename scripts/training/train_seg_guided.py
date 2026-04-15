import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from brainstroke.models import SegGuidedClassifier
from brainstroke.training import train_seg_guided_classifier
from brainstroke.model_io import (
    build_classifier,
    build_segmenter,
    load_model_checkpoint,
    get_input_size,
    load_champions,
)


def main():
    champs = load_champions()
    default_seg_prior = champs.get("best_segmentation", "unet")

    parser = argparse.ArgumentParser(description="Train segmentation-guided classifier")
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
    parser.add_argument(
        "--seg-prior",
        default=default_seg_prior,
        choices=["unet", "resunet", "attention_unet", "swin_unet_stroke_only"],
        help="Segmentation prior checkpoint to use. Defaults to best_segmentation from champions.json",
    )
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    base = build_classifier(args.model, pretrained=False)
    base = load_model_checkpoint(base, args.model)
    sg = SegGuidedClassifier(base)
    seg_model = build_segmenter(args.seg_prior, pretrained=False)
    seg_model = load_model_checkpoint(seg_model, args.seg_prior)

    img_size = get_input_size(args.model)
    use_amp = args.model not in ("xception", "convnext_small")
    train_seg_guided_classifier(sg, f"sg_{args.model}", seg_model, img_size=img_size, n_epochs=args.epochs, use_amp=use_amp)


if __name__ == "__main__":
    main()
