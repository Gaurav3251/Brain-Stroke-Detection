import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from brainstroke.model_io import load_models
from brainstroke.inference import predict_single_image


def main():
    parser = argparse.ArgumentParser(description="Run inference on a single image")
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--model",
        default="ensemble",
        choices=[
            "ensemble",
            "resnet50", "resnet101",
            "densenet121_se", "densenet201_se",
            "efficientnet_b4_spp", "efficientnetv2_s",
            "mobilenet_v2", "inception_v3", "xception",
            "convnext_small", "swin_unet", "all",
        ],
    )
    args = parser.parse_args()

    models = load_models()
    predict_single_image(args.image, models, model_choice=args.model)


if __name__ == "__main__":
    main()
