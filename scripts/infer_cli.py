import argparse
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from brainstroke.inference import load_models, predict_single_image


def main():
    parser = argparse.ArgumentParser(description="Brain stroke inference")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--model",
        default="ensemble",
        choices=["ensemble", "densenet121", "efficientnet_b4", "swin_unet", "all"],
        help="Model choice for reporting",
    )
    args = parser.parse_args()

    models = load_models()
    try:
        predict_single_image(args.image, models, model_choice=args.model)
    except Exception as e:
        print(f"[Error] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
