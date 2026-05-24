# RF-UAV Classification

Deep learning-based UAV identification via RF signal fingerprinting — benchmarking ResNet, EfficientNet, MobileNetV3, and ViT on 37 drone controller classes.

---

## Overview

This project classifies drone remote controllers by their **radio-frequency (RF) spectrograms** using transfer learning. Each controller produces a unique RF fingerprint; a CNN/ViT model trained on spectrogram images can identify the source device without decoding the signal.

The pipeline is controlled by two environment variables (`ALL_CLASSES`, `MODEL_NAME`), making it easy to swap models or switch between the 5-class DJI subset and the full 37-class benchmark.

---

## Dataset

**RF-UAV** — spectrogram images generated from raw IQ recordings via a Matlab pipeline.

| Split | Classes | Format |
|-------|---------|--------|
| Train + Valid (merged, re-split 80/20 stratified) | 37 controllers | 224 × 224 JPG |

The 37 controllers span brands including DJI, FrSky, Futaba, Radiolink, RadioMaster, Flysky, SIYI, Skydroid, JR Propo, Jumper, Herelink, Wfly, and YunZhuo.

> Dataset is **not included** in this repo (see `code/download_rfuav.py`).  
> Place the extracted data at `data/rfuav_subset/ImageSet-AllDrones-MatlabPipeline/`.

---

## Results

All models trained for **10 epochs**, AdamW + CosineAnnealingLR, batch size 64, on 37 classes.

| Model | Params | Val Accuracy |
|-------|--------|-------------|
| ResNet-18 | 11.7 M | **99.8 %** |
| ResNet-50 | 25.6 M | **99.8 %** |
| EfficientNet-B0 | 5.3 M | **99.8 %** |
| MobileNetV3-Small | 2.5 M | **99.8 %** |
| ViT-Small/16 | 22.1 M | 99.1 % |
| ResNet-18 (DJI 5-class) | 11.7 M | **100.0 %** |

Confusion matrices and per-class reports are in [`results/`](results/).

---

## Project Structure

```
rf-uav-classification/
├── code/
│   ├── config.py            # paths, hyperparameters, class lists
│   ├── dataset.py           # stratified dataset builder
│   ├── train.py             # training loop
│   ├── evaluate.py          # confusion matrix + classification report
│   ├── evaluate_snr.py      # accuracy vs. SNR curve
│   ├── download_rfuav.py    # dataset download helper
│   ├── inspect_dataset.py   # dataset statistics
│   └── run_all_models.ps1   # batch-train all 4 models (Windows)
├── notebooks/               # exploratory notebooks
├── results/                 # metrics, plots, training logs
│   ├── *_report.txt         # per-class precision / recall / F1
│   ├── *_history.json       # loss & accuracy curves
│   ├── *_snr_curve.json     # accuracy vs. SNR data
│   ├── *_confusion_matrix.png
│   └── training_logs/
└── data/                    # (gitignored) raw dataset
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install torch torchvision timm scikit-learn matplotlib seaborn tqdm pillow
```

### 2. Download the dataset

```bash
python code/download_rfuav.py
```

### 3. Train a model

```powershell
# ResNet-18, all 37 classes
$env:ALL_CLASSES = "1"; $env:MODEL_NAME = "resnet18"
python code/train.py

# ResNet-18, DJI 5-class subset
$env:ALL_CLASSES = "0"; $env:MODEL_NAME = "resnet18"
python code/train.py
```

### 4. Evaluate

```powershell
$env:ALL_CLASSES = "1"; $env:MODEL_NAME = "resnet18"
python code/evaluate.py       # confusion matrix + report
python code/evaluate_snr.py   # SNR curve
```

### 5. Train all 4 models sequentially (Windows)

```powershell
.\code\run_all_models.ps1
```

---

## Configuration

Key settings in `code/config.py` (also overridable via environment variables):

| Variable | Default | Description |
|----------|---------|-------------|
| `ALL_CLASSES` | `0` | `1` = all 37 classes, `0` = DJI 5-class |
| `MODEL_NAME` | `resnet18` | Any `timm` model name |
| `EPOCHS` | `10` | Training epochs |
| `BATCH_SIZE` | `64` | Batch size |
| `LEARNING_RATE` | `1e-3` | Initial LR (AdamW) |
| `IMG_SIZE` | `224` | Input image size |

---

## Tech Stack

- **PyTorch** — training framework  
- **timm** — pretrained model zoo (ResNet, EfficientNet, MobileNetV3, ViT)  
- **scikit-learn** — metrics and stratified splitting  
- **Matplotlib / Seaborn** — visualization  
