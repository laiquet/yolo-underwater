# 🐟 Salmon YOLO Pipeline

An end-to-end pipeline for **underwater salmon detection** using YOLO. The pipeline covers the full lifecycle—from raw underwater image enhancement through model training, inference, and post-processing crop extraction—in a single command-line tool.

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Stages](#pipeline-stages)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Layout](#dataset-layout)
- [Usage](#usage)
  - [1. Preprocess — Image Enhancement](#1-preprocess--image-enhancement)
  - [2. Train — YOLO Training](#2-train--yolo-training)
  - [3. Infer — YOLO Prediction](#3-infer--yolo-prediction)
  - [4. Crop — ROI Extraction](#4-crop--roi-extraction)
  - [5. All — Full Pipeline](#5-all--full-pipeline)
- [Configuration Reference](#configuration-reference)
- [Quality Metrics](#quality-metrics)
- [License](#license)

---

## Overview

Underwater imagery presents unique challenges for object detection: colour casts from water absorption, low contrast, and inconsistent lighting. This pipeline addresses those challenges by coupling domain-specific image enhancement with modern YOLO-based detection.

**Key features:**

- **Gray-World white balancing** to correct underwater colour casts
- **CLAHE** (Contrast Limited Adaptive Histogram Equalization) for local contrast enhancement
- **Optional Real-ESRGAN super-resolution** (×2 or ×4 upscaling)
- **No-reference quality metrics** (UIQM & UCIQE) computed before and after enhancement
- **YOLO training** with extensive augmentation control (YOLOv8/YOLO26x)
- **Inference & crop extraction** for downstream analysis

---

## Pipeline Stages

```
┌─────────────┐     ┌───────────┐     ┌───────────┐     ┌──────────────┐
│  Preprocess  │ ──▶ │   Train   │ ──▶ │   Infer   │ ──▶ │    Crop      │
│  (enhance)   │     │  (YOLO)   │     │ (predict) │     │ (ROI export) │
└─────────────┘     └───────────┘     └───────────┘     └──────────────┘
```

| Stage | Command | Description |
|-------|---------|-------------|
| **Preprocess** | `preprocess` | White balance → CLAHE → optional super-resolution → resize. Outputs enhanced images and quality metrics CSV. |
| **Train** | `train` | Fine-tunes a YOLO model on the (optionally enhanced) dataset with configurable augmentations. |
| **Infer** | `infer` | Runs prediction on test images using trained weights. Saves annotated images and YOLO-format labels. |
| **Crop** | `crop` | Parses predicted YOLO labels to extract individual fish bounding-box crops from source images. |
| **All** | `all` | Runs all four stages sequentially in one command. |

---

## Project Structure

```
yolo/
├── salmon_yolo_pipeline.py    # Main pipeline script (all stages)
├── yolo26x.pt                 # Pre-trained YOLO26x model weights
├── runs/
│   ├── detect/
│   │   ├── train-1/           # Training run outputs (weights, plots, metrics)
│   │   ├── predict/           # Prediction outputs (annotated images, labels)
│   │   └── ...
│   └── segment/               # (Segmentation runs, if applicable)
└── README.md
```

The **dataset** is expected at `../datasets/Healthy and Loser Salmon Dataset/yolo/` by default (configurable via `--dataset-root`).

---

## Requirements

### Core Dependencies

| Package | Purpose |
|---------|---------|
| Python ≥ 3.10 | Type-hint syntax (`list[str]`, `X \| Y`) |
| `opencv-python` | Image I/O and processing |
| `numpy` | Array operations |
| `scipy` | Uniform filtering for UIQM metric |
| `Pillow` | Crop extraction |
| `tqdm` | Progress bars |
| `pandas` | Metrics aggregation & CSV export |
| `ultralytics` | YOLO training and inference |

### Optional Dependencies

| Package | Purpose |
|---------|---------|
| `realesrgan` | Real-ESRGAN super-resolution |
| `basicsr` | Network architectures for Real-ESRGAN |
| `torch` (CUDA) | GPU acceleration for training & super-resolution |
| `matplotlib` | Quality metric distribution plots |

---

## Installation

```bash
# 1. Create and activate a conda environment
conda create -n yolo_env python=3.11 -y
conda activate yolo_env

# 2. Install PyTorch with CUDA (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install core dependencies
pip install ultralytics opencv-python numpy scipy Pillow tqdm pandas

# 4. (Optional) Install super-resolution support
pip install realesrgan basicsr

# 5. (Optional) Install plotting support
pip install matplotlib
```

---

## Dataset

This project uses the publicly available **Healthy and Loser Salmon Dataset** hosted on Mendeley Data:

> 📥 **Download:** [https://data.mendeley.com/datasets/rvrt4zs969/1](https://data.mendeley.com/datasets/rvrt4zs969/1)

### Layout

The pipeline expects a standard YOLO dataset structure:

```
datasets/Healthy and Loser Salmon Dataset/yolo/
├── data.yaml                  # YOLO dataset config
├── train/
│   ├── images/                # Training images
│   ├── labels/                # YOLO-format annotations (.txt)
│  
├── valid/
│   ├── images/
│   ├── labels/
│
└── test/
    ├── images/
    ├── labels/
```

Each label file contains one line per object in YOLO format:

```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are normalised to `[0, 1]` relative to image dimensions.

---

## Usage

### 1. Preprocess — Image Enhancement

Enhance underwater images with white balancing, CLAHE, and optional super-resolution:

```bash
# Default settings (1920×1080 output, CLAHE clip=3.0)
python salmon_yolo_pipeline.py preprocess

# Custom configuration
python salmon_yolo_pipeline.py preprocess \
    --dataset-root path/to/dataset \
    --target-size 1280 720 \
    --clahe-clip-limit 4.0 \
    --use-super-resolution --sr-scale 2 \
    --save-plots
```

**Key options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset-root` | `../datasets/.../yolo` | Path to the YOLO dataset root |
| `--splits` | `train valid test` | Dataset splits to process |
| `--target-size` | `1920 1080` | Output resolution (W H). Omit values to keep originals |
| `--clahe-clip-limit` | `3.0` | CLAHE contrast clip limit |
| `--clahe-tile-grid` | `8 8` | CLAHE tile grid size |
| `--use-super-resolution` | off | Enable Real-ESRGAN upscaling |
| `--sr-scale` | `2` | Super-resolution scale factor (2 or 4) |
| `--overwrite-originals` | off | Write enhanced images in-place |
| `--save-plots` | off | Save UIQM/UCIQE distribution plots |
| `--output-format` | `png` | Output image format |

### 2. Train — YOLO Training

Train a YOLO model on the dataset:

```bash
# Default training (250 epochs, AdamW, cosine LR)
python salmon_yolo_pipeline.py train

# Custom training
python salmon_yolo_pipeline.py train \
    --model yolo26x.pt \
    --data path/to/data_enhanced.yaml \
    --epochs 100 \
    --batch 8 \
    --imgsz 960 540 \
    --device 0
```

**Key options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `yolo26x.pt` | Base model weights |
| `--data` | `data.yaml` | YOLO data configuration file |
| `--epochs` | `250` | Number of training epochs |
| `--imgsz` | `960 540` | Training image size (W H) |
| `--batch` | `4` | Batch size |
| `--device` | `0` | CUDA device (`0`, `cpu`, etc.) |
| `--optimizer` | `AdamW` | Optimizer (SGD, Adam, AdamW) |
| `--patience` | `50` | Early stopping patience |
| `--dropout` | `0.2` | Dropout rate |
| `--mosaic` | `0.7` | Mosaic augmentation probability |

### 3. Infer — YOLO Prediction

Run inference on test images:

```bash
# Default inference (conf=0.50)
python salmon_yolo_pipeline.py infer

# Custom inference
python salmon_yolo_pipeline.py infer \
    --weights runs/detect/train/weights/best.pt \
    --source path/to/test/images_enhanced \
    --conf 0.25
```

**Key options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--weights` | `runs/detect/train-1/weights/best.pt` | Trained model weights |
| `--source` | `.../test/images` | Input images directory |
| `--conf` | `0.50` | Confidence threshold |
| `--no-save` | off | Disable saving annotated images |
| `--no-save-txt` | off | Disable saving YOLO label files |

### 4. Crop — ROI Extraction

Extract individual fish crops from predicted bounding boxes:

```bash
python salmon_yolo_pipeline.py crop

# Custom paths
python salmon_yolo_pipeline.py crop \
    --labels-dir runs/detect/predict/labels \
    --images-dir path/to/test/images_enhanced \
    --output-dir my_crops
```

**Key options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--labels-dir` | `runs/detect/predict/labels` | Directory with YOLO prediction labels |
| `--images-dir` | `.../test/images` | Source images to crop from |
| `--output-dir` | `crops` | Output directory for cropped images |
| `--crop-output-format` | `png` | Output image format for crops |

### 5. All — Full Pipeline

Run the entire pipeline end-to-end:

```bash
# Run all stages with defaults
python salmon_yolo_pipeline.py all

# Run all stages, training on enhanced images
python salmon_yolo_pipeline.py all --train-on-enhanced
```

The `all` command accepts every flag from all four stages. The `--train-on-enhanced` flag switches training to use the enhanced dataset YAML generated during preprocessing.

---

## Configuration Reference

All pipeline stages are controlled by dataclass-backed configuration objects, making it straightforward to use the pipeline programmatically:

```python
from salmon_yolo_pipeline import (
    PreprocessConfig,
    TrainConfig,
    InferenceConfig,
    CropConfig,
    preprocess_dataset,
    train_yolo,
    infer_yolo,
    crop_yolo_predictions,
)

# Preprocess with custom settings
preprocess_dataset(PreprocessConfig(
    clahe_clip_limit=4.0,
    use_super_resolution=True,
    sr_scale=2,
))

# Train
train_yolo(TrainConfig(epochs=100, batch=8))

# Infer
infer_yolo(InferenceConfig(conf=0.3))

# Crop
crop_yolo_predictions(CropConfig(output_dir=Path("my_crops")))
```

---

## Quality Metrics

The preprocessing stage computes two no-reference underwater image quality metrics for every image, both before and after enhancement:

| Metric | Full Name | Measures |
|--------|-----------|----------|
| **UIQM** | Underwater Image Quality Measure | Colourfulness, sharpness, and contrast |
| **UCIQE** | Underwater Colour Image Quality Evaluation | Chroma variance, luminance contrast, and saturation |

Results are saved to `enhancement_metrics.csv` in the dataset root. Use `--save-plots` to generate distribution and delta scatter plots.

---

## License

This project is provided as-is for research and educational purposes.
