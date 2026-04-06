# Brain Stroke Detection Project

This project detects brain stroke patterns from CT scans, performs lesion-aware analysis, and generates explainability outputs for review.

## Overview

- Segmentation models: U-Net, Swin-UNet
- Classification models: DenseNet121, EfficientNetB4
- Hybrid ensemble: segmentation-guided DenseNet + segmentation-guided EfficientNet + confidence fusion
- Explainability: GradCAM++ and segmentation-based visual outputs
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
|   |-- checkpoints/
|   |-- logs/
|   `-- outputs/
|       |-- gradcam/
|       |-- inference/
|       `-- plots/
|-- data/
|   `-- Brain_Stroke_CT_Dataset/
|-- scripts/
|   |-- create_venv.ps1
|   |-- infer_cli.py
|   |-- train_classifier.py
|   |-- train_segmentation.py
|   |-- train_seg_guided.py
|   `-- train_ensemble.py
|-- src/
|   `-- brainstroke/
|       |-- analysis/
|       |-- core/
|       |-- models/
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

3. Download the dataset and place it at:

```text
data/Brain_Stroke_CT_Dataset
```

4. Copy pretrained checkpoints into:

```text
artifacts/checkpoints/
```

If your dataset or checkpoints are stored elsewhere, set the paths with environment variables:

```powershell
$env:STROKE_DATA_ROOT = "D:\path\to\Brain_Stroke_CT_Dataset"
$env:STROKE_PRETRAINED_DIR = "D:\path\to\checkpoints"
```

## Quick Start

1. Clone the repository:

```powershell
git clone https://github.com/Gaurav3251/Brain-Stroke-Detection.git
cd Brain-Stroke-Detection
```

2. Create and activate the virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install the project requirements:

```powershell
pip install -r requirements.txt
```

4. Add the dataset and checkpoints, then run any interface below.

## CLI Inference

```powershell
python scripts\infer_cli.py --image path\to\ct.png --model ensemble
```

Available model options:
- `ensemble`
- `densenet121`
- `efficientnet_b4`
- `swin_unet`
- `all`

Inference outputs are saved in `artifacts/outputs/inference/`.

## Desktop UI

```powershell
python apps\ui\ui_app.py
```

Use the desktop app to browse for a CT image, select a model, and run inference locally.

## Web UI

```powershell
python apps\web\web_app.py
```

Then open:

```text
http://127.0.0.1:5000
```

The web app supports:
- CT image upload from the browser
- Model selection for the primary prediction
- Prediction confidence and per-model comparison
- Visual classification, explainability, and metrics outputs
- Stroke-specific precaution and response guidance
- Downloadable PDF reports generated from the web workflow

## Training

```powershell
python scripts\train_classifier.py --model densenet121
python scripts\train_classifier.py --model efficientnet_b4
python scripts\train_segmentation.py --model unet
python scripts\train_segmentation.py --model swin_unet
python scripts\train_seg_guided.py
python scripts\train_ensemble.py
```

## Configuration

Primary path settings live in `src/brainstroke/core/config.py`.

Useful environment variables:
- `STROKE_DATA_ROOT`
- `STROKE_ARTIFACTS_DIR`
- `STROKE_OUTPUT_DIR`
- `STROKE_MODEL_DIR`
- `STROKE_LOG_DIR`
- `STROKE_PLOT_DIR`
- `STROKE_GCAM_DIR`
- `STROKE_INFER_DIR`
- `STROKE_PRETRAINED_DIR`

## Notes

- If `timm` is not installed, Swin-UNet is skipped automatically.
- For DICOM decoding support, install `pylibjpeg-libjpeg`.

## License

See `LICENSE.txt`.
