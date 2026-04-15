# Brain Stroke Detection Project

This project detects brain stroke patterns from CT scans, performs lesion-aware analysis, and generates explainability outputs for review.

## Overview

- Classification models (10): ResNet50, ResNet101, DenseNet121(+SE), DenseNet201(+SE), EfficientNetB4(+SPP), EfficientNetV2-S, MobileNetV2, InceptionV3 (299x299), Xception (timm), ConvNeXtSmall
- Segmentation models (4): U-Net, ResUNet (resnet34 encoder), Attention U-Net (scSE), Swin-UNet (joint cls+seg)
- Ensemble: 5 seg-guided classifier champions with learnable confidence weights and temperatures, all driven by the best stroke-only segmentation prior
- Explainability: GradCAM++ for CNNs, segmentation masks for SwinUNet, highest-weight model for ensemble
- Interfaces: CLI, Tkinter desktop app, and Flask web app

Dataset:
- Brain Stroke CT Dataset (Kaggle)
- https://www.kaggle.com/datasets/ozguraslank/brain-stroke-ct-dataset

## Project Structure

```text
Brain-Stroke-Detection/
|-- apps/
|   |-- ui/
|   |   `-- ui_app.py
|   `-- web/
|       |-- static/
|       |   `-- styles.css
|       |-- templates/
|       |   `-- index.html
|       `-- web_app.py
|-- artifacts/
|   |-- train/
|   |   |-- checkpoints/
|   |   |-- logs/
|   |   `-- outputs/
|   |       |-- gradcam/
|   |       |-- metrics/
|   |       |-- plots/
|   |       |-- predictions/
|   |       `-- reports/
|   |-- ui/
|   |   `-- inference/
|   `-- web/
|       |-- inference/
|       `-- reports/
|-- configs/
|   `-- train_config_example.json
|-- data/
|   `-- Brain_Stroke_CT_Dataset/
|-- scripts/
|   |-- create_venv.ps1
|   |-- evaluate_saved_model.py
|   |-- infer_cli.py
|   |-- select_champions.py
|   `-- training/
|       |-- train_classifier.py
|       |-- train_segmentation.py
|       |-- train_seg_guided.py
|       |-- train_ensemble.py
|       |-- tune_classifier.py
|       |-- train_runner.py
|       |-- eval_seg_guided.py
|       |-- eval_ensemble.py
|       `-- generate_gradcam.py
|-- src/
|   `-- brainstroke/
|       |-- analysis/
|       |-- core/
|       |-- models/
|       |   |-- classification/
|       |   |-- segmentation/
|       |   `-- ensemble/
|       |-- training/
|       |-- inference.py
|       |-- model_io.py
|       `-- web_support.py
|-- LICENSE.txt
|-- README.md
`-- requirements.txt
```

## Setup

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Place the dataset here:

```text
data/Brain_Stroke_CT_Dataset
```

4. (Optional) Copy pretrained checkpoints into:

```text
artifacts/train/checkpoints/
```

If your dataset or checkpoints are stored elsewhere, set paths with environment variables:

```powershell
$env:STROKE_DATA_ROOT = "D:\path\to\Brain_Stroke_CT_Dataset"
$env:STROKE_PRETRAINED_DIR = "D:\path\to\checkpoints"
```

## Repo Setup (Main vs New Branch)

Clone and use the updated structure branch:

```powershell
git clone <YOUR_REPO_URL>
cd Brain-Stroke-Detection
git checkout <NEW_BRANCH_NAME>
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Clone and use the original structure on `main`:

```powershell
git clone <YOUR_REPO_URL>
cd Brain-Stroke-Detection
git checkout main
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Quick Start

```powershell
python scripts\infer_cli.py --image path\to\ct.png --model ensemble
```

Inference outputs go to `artifacts/ui/inference/`.

## CLI Inference

```powershell
python scripts\infer_cli.py --image path\to\ct.png --model resnet50
```

Available model options:
- `ensemble` (recommended)
- `sg_densenet201_se`, `sg_mobilenet_v2`, `sg_resnet50`, `sg_xception`, `sg_convnext_small`
- `all`

## Desktop UI

```powershell
python apps\ui\ui_app.py
```

The desktop UI saves outputs in `artifacts/ui/inference/`.

## Web UI

```powershell
python apps\web\web_app.py
```

Then open:

```text
http://127.0.0.1:5000
```

Web outputs are saved in `artifacts/web/`.

## Training

All training artifacts go to `artifacts/train/`.
Classification sample predictions and confusion matrices are saved in `artifacts/train/outputs/predictions/`.

### 1. Train classification models (one at a time)

```powershell
python scripts\training\train_classifier.py --model resnet50 --pretrained
python scripts\training\train_classifier.py --model resnet101 --pretrained
python scripts\training\train_classifier.py --model densenet121_se --pretrained
python scripts\training\train_classifier.py --model densenet201_se --pretrained
python scripts\training\train_classifier.py --model efficientnet_b4_spp --pretrained
python scripts\training\train_classifier.py --model efficientnetv2_s --pretrained
python scripts\training\train_classifier.py --model mobilenet_v2 --pretrained
python scripts\training\train_classifier.py --model inception_v3 --pretrained
python scripts\training\train_classifier.py --model xception --pretrained
python scripts\training\train_classifier.py --model convnext_small --pretrained
```

### 2. Train segmentation models (stroke-only priors)

```powershell
python scripts\training\train_segmentation.py --model unet
python scripts\training\train_segmentation.py --model resunet --pretrained
python scripts\training\train_segmentation.py --model attention_unet
python scripts\training\train_segmentation.py --model swin_unet --pretrained --stroke-only --run-name swin_unet_stroke_only
```

### 3. Select group champions and best segmentation prior

```powershell
python scripts\select_champions.py
```

This writes `artifacts/train/outputs/metrics/champions.json`.
The segmentation champion pool is:
- `unet`
- `resunet`
- `attention_unet`
- `swin_unet_stroke_only`

### 4. Train seg-guided champions (A-E)

Run these for the classifier champions selected in `champions.json`. The segmentation prior defaults to `best_segmentation` from `champions.json`. Override with `--seg-prior` if needed.

```powershell
python scripts\training\train_seg_guided.py --model densenet201_se
python scripts\training\train_seg_guided.py --model mobilenet_v2
python scripts\training\train_seg_guided.py --model resnet50
python scripts\training\train_seg_guided.py --model xception
python scripts\training\train_seg_guided.py --model convnext_small
```

### 5. Train the confidence ensemble

```powershell
python scripts\training\train_ensemble.py
```

To fine-tune backbones end-to-end:

```powershell
python scripts\training\train_ensemble.py --fine-tune
```

### Optional: Hyperparameter Tuning (lightweight)

```powershell
python scripts\training\tune_classifier.py --model resnet50 --epochs 5
```

This saves trial results in `artifacts/train/outputs/metrics/*_tuning.json`.

### Train Runner (config-driven)

```powershell
python scripts\training\train_runner.py --config configs\train_config_example.json
```

Example config structure:

```json
{
  "task": "classify",
  "model": "resnet50",
  "pretrained": true,
  "epochs": 30
}
```

## Post-Training Evaluation Commands

Use these commands to regenerate outputs from saved checkpoints without retraining. This is separate from champion selection and does not change `champions.json`.

### Classification checkpoint evaluation

```powershell
python scripts\evaluate_saved_model.py --task classify --model resnet101
```

### Segmentation checkpoint evaluation

```powershell
python scripts\evaluate_saved_model.py --task segment --model unet
```

### Seg-guided checkpoint evaluation

Use the base classifier name with `--model`; the script loads `sg_<model>_best.pth` automatically. By default, `--seg-prior` comes from `best_segmentation` in `champions.json`.

```powershell
python scripts\evaluate_saved_model.py --task seg_guided --model resnet101
```

### Ensemble checkpoint evaluation

Pass the exact five classifier champions explicitly so this stays independent of champion selection.

```powershell
python scripts\evaluate_saved_model.py --task ensemble --ensemble-members densenet201_se mobilenet_v2 resnet50 xception convnext_small
```

### Outputs regenerated by the script

- `artifacts/train/outputs/plots/*_training_curves.png`
- `artifacts/train/outputs/plots/*_roc.png`
- `artifacts/train/outputs/plots/*_pr.png`
- `artifacts/train/outputs/plots/*_confidence_hist.png`
- `artifacts/train/outputs/predictions/*_confusion_matrix.png`
- `artifacts/train/outputs/predictions/*_sample_preds.png`
- `artifacts/train/outputs/metrics/*_metrics.json`
- `artifacts/train/outputs/reports/*_classification_report.txt`
- `artifacts/train/outputs/reports/*_confusion_matrix.json`

### 6. Generate evaluation outputs and GradCAM samples

```powershell
python scripts\training\eval_seg_guided.py --model densenet201_se mobilenet_v2 resnet50 xception convnext_small
python scripts\training\eval_ensemble.py
python scripts\training\generate_gradcam.py
```

## Where To Change Epochs

Update `NUM_EPOCHS` inside `src/brainstroke/core/config.py`.

## Configuration

Primary path settings live in `src/brainstroke/core/config.py`.

Useful environment variables:
- `STROKE_DATA_ROOT`
- `STROKE_ARTIFACTS_DIR`
- `STROKE_TRAIN_DIR`
- `STROKE_UI_DIR`
- `STROKE_WEB_DIR`
- `STROKE_MODEL_DIR`
- `STROKE_LOG_DIR`
- `STROKE_PLOT_DIR`
- `STROKE_REPORT_DIR`
- `STROKE_METRIC_DIR`
- `STROKE_GCAM_DIR`
- `STROKE_INFER_DIR`
- `STROKE_PRETRAINED_DIR`

## Notes

- If `timm` is not installed, Swin-UNet and Xception will be skipped automatically.
- For DICOM decoding support, install `pylibjpeg-libjpeg`.

## License

See `LICENSE.txt`.
