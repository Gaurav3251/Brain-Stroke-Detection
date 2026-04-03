# Brain Stroke Detection Project

This project detects brain stroke types from CT scans, performs lesion segmentation, and provides explainability outputs.

- Segmentation: U-Net, Swin-UNet
- Classification: DenseNet121, EfficientNetB4
- Hybrid ensemble: Seg-guided DenseNet + Seg-guided EfficientNet + confidence fusion
- Explainability: GradCAM++ and segmentation-based spatial XAI

Dataset: Brain Stroke CT Dataset (Kaggle)
- https://www.kaggle.com/datasets/ozguraslank/brain-stroke-ct-dataset

##  Project Structure

```
brain-stroke-detection-project/
│
├── apps/
│   └── ui/
│       └── ui_app.py                # Tkinter UI
│
├── artifacts/
│   ├── checkpoints/                # .pth checkpoints
│   ├── logs/                       # Training logs
│   └── outputs/                    # Plots, Grad-CAM, inference outputs
│       ├── gradcam/
│       ├── inference/
│       └── plots/
│
├── data/
│   └── Brain_Stroke_CT_Dataset/    # Dataset (from Kaggle)
│
├── scripts/
│   ├── create_venv.ps1             # Virtual environment helper
│   ├── infer_cli.py                # CLI inference
│   ├── train_classifier.py         # Train DenseNet / EfficientNet
│   ├── train_segmentation.py       # Train U-Net / Swin-UNet
│   ├── train_seg_guided.py         # Train segmentation-guided classifiers
│   └── train_ensemble.py           # Train ensemble fusion
│
├── src/
│   └── brainstroke/
│       ├── analysis/               # Evaluation, visualization, explainability
│       ├── core/                   # Config, preprocessing, data, utils
│       ├── models/                 # Model definitions (one per file)
│       ├── training/               # Losses, training loops, trainers
│       ├── inference.py            # Inference pipeline
│       └── model_io.py             # Model loading and paths
│
├── LICENSE.txt
├── requirements.txt
└── README.md
```

## Setup (Windows / PowerShell)

1. Create virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Place dataset at:

```
./data/Brain_Stroke_CT_Dataset
```

If it is elsewhere, update the path in `src/brainstroke/core/config.py` or set an environment variable:

```powershell
$env:STROKE_DATA_ROOT = "D:\path\to\Brain_Stroke_CT_Dataset"
```

4. Copy model checkpoints into `./artifacts/checkpoints/` or set:

```powershell
$env:STROKE_PRETRAINED_DIR = "D:\path\to\checkpoints"
```

## Collaborator Setup (Quick Start)

1. Clone the repo
```powershell
git clone <repo_url>
cd "brain stroke detection project"
```

2. Create and activate venv
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies
```powershell
pip install -r requirements.txt
```

4. Download dataset from Kaggle and place into:
```
data/Brain_Stroke_CT_Dataset
```

5. Copy checkpoints into:
```
artifacts/checkpoints/
```

6. Run inference (see commands below)

## Run Inference (CLI)

```powershell
python scripts\infer_cli.py --image path\to\ct.png --model ensemble
```

Model choices:
- `ensemble` (default)
- `densenet121`
- `efficientnet_b4`
- `swin_unet`
- `all`

Outputs are saved to `artifacts/outputs/inference/`.

## Run Inference (UI)

```powershell
python apps\ui\ui_app.py
```

Use the **Browse** button to select a CT image, choose the model, then run inference.

## Training (Optional)

```powershell
python scripts\train_classifier.py --model densenet121
python scripts\train_classifier.py --model efficientnet_b4
python scripts\train_segmentation.py --model unet
python scripts\train_segmentation.py --model swin_unet
python scripts\train_seg_guided.py
python scripts\train_ensemble.py
```

## Where to Update Paths

- Dataset path: `src/brainstroke/core/config.py` -> `DATA_ROOT`
- Pretrained checkpoints: `src/brainstroke/core/config.py` -> `PRETRAINED_DIR`
- Artifacts root: `src/brainstroke/core/config.py` -> `ARTIFACTS_DIR`
- Outputs:
  - Plots: `src/brainstroke/core/config.py` -> `PLOT_DIR`
  - GradCAM: `src/brainstroke/core/config.py` -> `GCAM_DIR`
  - Inference: `src/brainstroke/core/config.py` -> `INFER_DIR`
- Logs: `src/brainstroke/core/config.py` -> `LOG_DIR`

You can override any of the above using environment variables:

- `STROKE_DATA_ROOT`
- `STROKE_ARTIFACTS_DIR`
- `STROKE_OUTPUT_DIR`
- `STROKE_MODEL_DIR`
- `STROKE_LOG_DIR`
- `STROKE_PLOT_DIR`
- `STROKE_GCAM_DIR`
- `STROKE_INFER_DIR`
- `STROKE_PRETRAINED_DIR`

## License

See `LICENSE.txt`.

## Notes

- If `timm` is not installed, Swin-UNet will be skipped automatically.
- For DICOM decoding, `pylibjpeg-libjpeg` is required 