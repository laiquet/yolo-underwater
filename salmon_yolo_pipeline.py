from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from multiprocessing import freeze_support
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter
from tqdm import tqdm


DEFAULT_DATASET_ROOT = Path("../datasets/Healthy and Loser Salmon Dataset/yolo")
DEFAULT_SPLITS = ["train", "valid", "test"]
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# -----------------------------------------------------------------------------
# Configuration containers
# -----------------------------------------------------------------------------

@dataclass
class PreprocessConfig:
    dataset_root: Path = DEFAULT_DATASET_ROOT
    splits: list[str] = None
    target_size: Optional[tuple[int, int]] = (1920, 1080)  # width, height
    clahe_clip_limit: float = 3.0
    clahe_tile_grid: tuple[int, int] = (8, 8)
    use_super_resolution: bool = False
    sr_scale: int = 2
    sr_half: bool = True
    output_format: str = "png"
    overwrite_originals: bool = False
    metrics_csv_name: str = "enhancement_metrics.csv"
    create_enhanced_yaml: bool = True
    copy_labels_for_enhanced: bool = True
    nc: int = 1
    names: list[str] = None
    save_plots: bool = False
    show_plots: bool = False

    def __post_init__(self) -> None:
        if self.splits is None:
            self.splits = DEFAULT_SPLITS.copy()
        if self.names is None:
            self.names = ["fish"]


@dataclass
class TrainConfig:
    model: str = "yolo26x.pt"
    data: Path = DEFAULT_DATASET_ROOT / "data.yaml"
    epochs: int = 250
    imgsz: tuple[int, int] = (960, 540)
    device: str = "0"
    batch: int = 4
    workers: int = 8
    amp: bool = True
    cache: bool = True
    pretrained: bool = True
    optimizer: str = "AdamW"
    cos_lr: bool = True
    seed: int = 123
    patience: int = 50
    weight_decay: float = 0.005
    val: bool = True
    plots: bool = True
    save_period: int = -1
    verbose: bool = True
    augment: bool = True
    rect: bool = True
    bgr: float = 0.3
    translate: float = 0.05
    scale: float = 0.05
    dropout: float = 0.2
    mosaic: float = 0.7
    hsv_h: float = 0.01
    hsv_s: float = 0.5
    hsv_v: float = 0.4
    flipud: float = 0.1
    fliplr: float = 0.7
    cutmix: float = 0.0
    copy_paste: float = 0.0
    shear: float = 3.0
    degrees: float = 5.0


@dataclass
class InferenceConfig:
    weights: Path = Path("runs/detect/train-1/weights/best.pt")
    source: Path = DEFAULT_DATASET_ROOT / "test/images"
    conf: float = 0.50
    save: bool = True
    save_conf: bool = True
    save_txt: bool = True
    show_conf: bool = True
    show_labels: bool = True
    show_boxes: bool = True
    project: Optional[str] = None
    name: Optional[str] = None


@dataclass
class CropConfig:
    labels_dir: Path = Path("runs/detect/predict/labels")
    images_dir: Path = DEFAULT_DATASET_ROOT / "test/images"
    output_dir: Path = Path("crops")
    output_format: str = "png"


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------

def absolute(path: Path | str) -> str:
    return str(Path(path).expanduser().resolve())


def list_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def find_image_by_stem(images_dir: Path, stem: str) -> Optional[Path]:
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    # Fallback: case-insensitive stem match
    for image_path in list_images(images_dir):
        if image_path.stem == stem:
            return image_path
    return None


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Underwater image enhancement functions
# -----------------------------------------------------------------------------

def gray_world_white_balance(img_bgr: np.ndarray) -> np.ndarray:
    """Apply Gray World white balance to reduce underwater color cast."""
    img = img_bgr.astype(np.float32)
    avg_b, avg_g, avg_r = img.mean(axis=(0, 1))
    avg_all = (avg_b + avg_g + avg_r) / 3.0

    img[:, :, 0] *= avg_all / (avg_b + 1e-6)
    img[:, :, 1] *= avg_all / (avg_g + 1e-6)
    img[:, :, 2] *= avg_all / (avg_r + 1e-6)

    return np.clip(img, 0, 255).astype(np.uint8)


def apply_clahe(
    img_bgr: np.ndarray,
    clip_limit: float = 3.0,
    tile_grid: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE on the LAB L-channel."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_enhanced = clahe.apply(l_channel)

    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def standardize_size(
    img_bgr: np.ndarray,
    target_size: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    """Resize with letterboxing to the target (width, height), or leave unchanged."""
    if target_size is None:
        return img_bgr

    target_w, target_h = target_size
    h, w = img_bgr.shape[:2]

    if w == target_w and h == target_h:
        return img_bgr

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    interpolation = cv2.INTER_LANCZOS4 if scale > 1 else cv2.INTER_AREA
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=interpolation)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas


def setup_realesrgan(scale: int = 2, half: bool = True):
    """Initialize Real-ESRGAN. Requires realesrgan, basicsr, and torch."""
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        import torch
    except ImportError as exc:
        raise ImportError(
            "Real-ESRGAN dependencies are not installed. Install them with:\n"
            "  pip install realesrgan basicsr\n"
            "or run preprocessing without --use-super-resolution."
        ) from exc

    if scale == 2:
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2,
        )
        model_name = "RealESRGAN_x2plus"
    elif scale == 4:
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        model_name = "RealESRGAN_x4plus"
    else:
        raise ValueError("Real-ESRGAN scale must be 2 or 4.")

    gpu_available = torch.cuda.is_available()
    use_half = half and gpu_available

    upsampler = RealESRGANer(
        scale=scale,
        model_path=None,  # Auto-downloads from GitHub releases on first use
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=use_half,
        gpu_id=0 if gpu_available else None,
    )

    device = "CUDA (FP16)" if use_half else ("CUDA (FP32)" if gpu_available else "CPU")
    print(f"Real-ESRGAN initialized: {model_name} | Device: {device}")
    return upsampler


def apply_super_resolution(img_bgr: np.ndarray, upsampler, outscale: Optional[int] = None) -> np.ndarray:
    """Upscale an image with Real-ESRGAN."""
    output, _ = upsampler.enhance(img_bgr, outscale=outscale)
    return output


def enhance_image(
    img_bgr: np.ndarray,
    upsampler=None,
    target_size: Optional[tuple[int, int]] = None,
    clip_limit: float = 3.0,
    tile_grid: tuple[int, int] = (8, 8),
    sr_scale: Optional[int] = None,
) -> np.ndarray:
    """White Balance -> CLAHE -> optional Real-ESRGAN -> optional standardization."""
    img = gray_world_white_balance(img_bgr)
    img = apply_clahe(img, clip_limit=clip_limit, tile_grid=tile_grid)

    if upsampler is not None:
        img = apply_super_resolution(img, upsampler, outscale=sr_scale)

    img = standardize_size(img, target_size=target_size)
    return img


# -----------------------------------------------------------------------------
# No-reference underwater quality metrics
# -----------------------------------------------------------------------------

def compute_uiqm(img_rgb: np.ndarray) -> float:
    """Compute the Underwater Image Quality Measure (UIQM)."""
    r = img_rgb[:, :, 0].astype(np.float64)
    g = img_rgb[:, :, 1].astype(np.float64)
    b = img_rgb[:, :, 2].astype(np.float64)

    rg = r - g
    yb = 0.5 * (r + g) - b
    uicm = (
        -0.0268 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
        + 0.1586 * np.sqrt(rg.var() + yb.var())
    )

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_mag = np.hypot(sobelx, sobely)
    uism = np.log(edge_mag.mean() + 1e-8)

    gray_float = gray.astype(np.float64)
    local_mean = uniform_filter(gray_float, size=11)
    local_var = uniform_filter((gray_float - local_mean) ** 2, size=11)
    uiconm = np.log(np.sqrt(local_var.mean()) + 1e-8)

    c1, c2, c3 = 0.0282, 0.2953, 3.5753
    return float(c1 * uicm + c2 * uism + c3 * uiconm)


def compute_uciqe(img_bgr: np.ndarray) -> float:
    """Compute the Underwater Color Image Quality Evaluation (UCIQE)."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    l_channel, a_channel, b_channel = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    chroma = np.sqrt(a_channel ** 2 + b_channel ** 2)
    sigma_c = chroma.std()

    l_norm = l_channel / 255.0
    con_l = l_norm.max() - l_norm.min()

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float64)
    mu_s = (hsv[:, :, 1] / 255.0).mean()

    c1, c2, c3 = 0.4680, 0.2745, 0.2576
    return float(c1 * sigma_c + c2 * con_l + c3 * mu_s)


# -----------------------------------------------------------------------------
# Preprocessing orchestration
# -----------------------------------------------------------------------------

def copy_label_for_enhanced_image(dataset_root: Path, split: str, image_stem: str, output_dir: Path) -> None:
    """
    Copy a matching YOLO label into images_enhanced.

    Ultralytics usually infers label paths by replacing an /images/ path segment with
    /labels/. Because the original notebook writes enhanced images to images_enhanced,
    this helper also places matching .txt labels beside the enhanced images so training
    from data_enhanced.yaml can find them reliably.
    """
    label_path = dataset_root / split / "labels" / f"{image_stem}.txt"
    if label_path.exists():
        shutil.copy2(label_path, output_dir / f"{image_stem}.txt")


def write_enhanced_data_yaml(config: PreprocessConfig) -> Path:
    output_yaml = config.dataset_root / "data_enhanced.yaml"
    names_repr = "[" + ", ".join(repr(name) for name in config.names) + "]"
    content = (
        f"path: {absolute(config.dataset_root)}\n"
        "train: train/images_enhanced\n"
        "val: valid/images_enhanced\n"
        "test: test/images_enhanced\n\n"
        f"nc: {config.nc}\n"
        f"names: {names_repr}\n"
    )
    output_yaml.write_text(content, encoding="utf-8")
    print(f"Created enhanced YOLO data file: {absolute(output_yaml)}")
    return output_yaml


def summarize_metrics(metrics: list[dict], dataset_root: Path, csv_name: str) -> Optional["object"]:
    if not metrics:
        print("No metrics collected. Check that input image folders exist and contain images.")
        return None

    import pandas as pd

    df = pd.DataFrame(metrics)
    csv_path = dataset_root / csv_name
    df.to_csv(csv_path, index=False)

    summary = df.groupby("split").agg(
        {
            "filename": "count",
            "uiqm_before": "mean",
            "uiqm_after": "mean",
            "uiqm_delta": "mean",
            "uciqe_before": "mean",
            "uciqe_after": "mean",
            "uciqe_delta": "mean",
        }
    ).rename(columns={"filename": "num_images"}).round(4)

    print("\n=== Quality Metrics Summary (Per Split) ===")
    print(summary.to_string())

    print("\n=== Overall ===")
    print(f"  Images processed : {len(df)}")
    print(f"  UIQM  (before)   : {df['uiqm_before'].mean():.4f} ± {df['uiqm_before'].std():.4f}")
    print(f"  UIQM  (after)    : {df['uiqm_after'].mean():.4f} ± {df['uiqm_after'].std():.4f}")
    print(f"  UIQM  (Δ)        : {df['uiqm_delta'].mean():+.4f}")
    print(f"  UCIQE (before)   : {df['uciqe_before'].mean():.4f} ± {df['uciqe_before'].std():.4f}")
    print(f"  UCIQE (after)    : {df['uciqe_after'].mean():.4f} ± {df['uciqe_after'].std():.4f}")
    print(f"  UCIQE (Δ)        : {df['uciqe_delta'].mean():+.4f}")
    print(f"\nMetrics saved to: {absolute(csv_path)}")
    return df


def save_metric_plots(df, dataset_root: Path, show: bool = False) -> None:
    import matplotlib.pyplot as plt

    plots_dir = dataset_root / "enhancement_plots"
    ensure_directory(plots_dir)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df["uiqm_before"], bins=20, alpha=0.6, label="Before", edgecolor="white")
    axes[0].hist(df["uiqm_after"], bins=20, alpha=0.6, label="After", edgecolor="white")
    axes[0].axvline(df["uiqm_before"].mean(), linestyle="--", linewidth=2)
    axes[0].axvline(df["uiqm_after"].mean(), linestyle="--", linewidth=2)
    axes[0].set_title("UIQM Distribution", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("UIQM Score")
    axes[0].set_ylabel("Count")
    axes[0].legend(fontsize=11)

    axes[1].hist(df["uciqe_before"], bins=20, alpha=0.6, label="Before", edgecolor="white")
    axes[1].hist(df["uciqe_after"], bins=20, alpha=0.6, label="After", edgecolor="white")
    axes[1].axvline(df["uciqe_before"].mean(), linestyle="--", linewidth=2)
    axes[1].axvline(df["uciqe_after"].mean(), linestyle="--", linewidth=2)
    axes[1].set_title("UCIQE Distribution", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("UCIQE Score")
    axes[1].set_ylabel("Count")
    axes[1].legend(fontsize=11)

    plt.suptitle("Image Quality Improvement – Before vs After Enhancement", fontsize=14, fontweight="bold")
    plt.tight_layout()
    dist_path = plots_dir / "quality_metric_distributions.png"
    fig.savefig(dist_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for split_name in sorted(df["split"].unique()):
        mask = df["split"] == split_name
        ax.scatter(df.loc[mask, "uiqm_delta"], df.loc[mask, "uciqe_delta"], label=split_name, alpha=0.7, s=50)

    ax.axhline(0, linestyle="--", alpha=0.5)
    ax.axvline(0, linestyle="--", alpha=0.5)
    ax.set_xlabel("ΔUIQM (After − Before)", fontsize=12)
    ax.set_ylabel("ΔUCIQE (After − Before)", fontsize=12)
    ax.set_title("Per-Image Quality Improvement", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    plt.tight_layout()
    delta_path = plots_dir / "quality_metric_deltas.png"
    fig.savefig(delta_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    print(f"Saved metric plots to: {absolute(plots_dir)}")


def preprocess_dataset(config: PreprocessConfig):
    dataset_root = config.dataset_root.expanduser()
    print(f"Dataset root : {absolute(dataset_root)}")
    print(f"Splits       : {config.splits}")
    print(f"Target size  : {config.target_size}")
    print(f"Super-Res    : {'ON (x' + str(config.sr_scale) + ')' if config.use_super_resolution else 'OFF'}")

    upsampler = setup_realesrgan(scale=config.sr_scale, half=config.sr_half) if config.use_super_resolution else None
    if upsampler is None:
        print("Super-Resolution: DISABLED")

    all_metrics: list[dict] = []

    for split in config.splits:
        input_dir = dataset_root / split / "images"
        output_dir = input_dir if config.overwrite_originals else dataset_root / split / "images_enhanced"
        ensure_directory(output_dir)

        image_files = list_images(input_dir)
        print("\n" + "=" * 60)
        print(f"Processing split: {split} ({len(image_files)} images)")
        print(f"  Input : {absolute(input_dir)}")
        print(f"  Output: {absolute(output_dir)}")
        print("=" * 60)

        if not input_dir.exists():
            print(f"WARNING: Input directory does not exist: {absolute(input_dir)}")
            continue

        for image_path in tqdm(image_files, desc=split, unit="img"):
            img_bgr = cv2.imread(str(image_path))
            if img_bgr is None:
                print(f"WARNING: Could not read {absolute(image_path)}, skipping.")
                continue

            img_rgb_orig = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            uiqm_before = compute_uiqm(img_rgb_orig)
            uciqe_before = compute_uciqe(img_bgr)
            h_orig, w_orig = img_bgr.shape[:2]

            sr_target = config.target_size if not config.use_super_resolution else None
            enhanced_bgr = enhance_image(
                img_bgr,
                upsampler=upsampler,
                target_size=sr_target,
                clip_limit=config.clahe_clip_limit,
                tile_grid=config.clahe_tile_grid,
                sr_scale=config.sr_scale if config.use_super_resolution else None,
            )

            if config.use_super_resolution and config.target_size is not None:
                enhanced_bgr = standardize_size(enhanced_bgr, config.target_size)

            enh_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
            uiqm_after = compute_uiqm(enh_rgb)
            uciqe_after = compute_uciqe(enhanced_bgr)
            h_out, w_out = enhanced_bgr.shape[:2]

            out_name = f"{image_path.stem}.{config.output_format.lower()}"
            out_path = output_dir / out_name
            cv2.imwrite(str(out_path), enhanced_bgr)

            if (not config.overwrite_originals) and config.copy_labels_for_enhanced:
                copy_label_for_enhanced_image(dataset_root, split, image_path.stem, output_dir)

            all_metrics.append(
                {
                    "split": split,
                    "filename": image_path.name,
                    "size_original": f"{w_orig}x{h_orig}",
                    "size_output": f"{w_out}x{h_out}",
                    "uiqm_before": uiqm_before,
                    "uiqm_after": uiqm_after,
                    "uiqm_delta": uiqm_after - uiqm_before,
                    "uciqe_before": uciqe_before,
                    "uciqe_after": uciqe_after,
                    "uciqe_delta": uciqe_after - uciqe_before,
                }
            )

    print("\n" + "=" * 60)
    print(f"DONE – Processed {len(all_metrics)} images across {len(config.splits)} splits.")
    print("=" * 60)

    df = summarize_metrics(all_metrics, dataset_root, config.metrics_csv_name)

    if config.create_enhanced_yaml and not config.overwrite_originals:
        write_enhanced_data_yaml(config)
    elif config.overwrite_originals:
        print("OVERWRITE_ORIGINALS=True -> original data.yaml remains valid.")

    if df is not None and config.save_plots:
        save_metric_plots(df, dataset_root, show=config.show_plots)

    return df


# -----------------------------------------------------------------------------
# YOLO training and inference
# -----------------------------------------------------------------------------

def train_yolo(config: TrainConfig):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Ultralytics is not installed. Install it with: pip install ultralytics") from exc

    model = YOLO(config.model)
    results = model.train(
        data=str(config.data),
        epochs=config.epochs,
        imgsz=config.imgsz,
        device=config.device,
        batch=config.batch,
        workers=config.workers,
        amp=config.amp,
        cache=config.cache,
        pretrained=config.pretrained,
        optimizer=config.optimizer,
        cos_lr=config.cos_lr,
        seed=config.seed,
        patience=config.patience,
        weight_decay=config.weight_decay,
        val=config.val,
        plots=config.plots,
        save_period=config.save_period,
        verbose=config.verbose,
        augment=config.augment,
        rect=config.rect,
        bgr=config.bgr,
        translate=config.translate,
        scale=config.scale,
        dropout=config.dropout,
        mosaic=config.mosaic,
        hsv_h=config.hsv_h,
        hsv_s=config.hsv_s,
        hsv_v=config.hsv_v,
        flipud=config.flipud,
        fliplr=config.fliplr,
        cutmix=config.cutmix,
        copy_paste=config.copy_paste,
        shear=config.shear,
        degrees=config.degrees,
    )
    return results


def infer_yolo(config: InferenceConfig):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Ultralytics is not installed. Install it with: pip install ultralytics") from exc

    print(f"Working directory: {os.getcwd()}")
    model = YOLO(str(config.weights))

    predict_kwargs = {
        "source": str(config.source),
        "conf": config.conf,
        "save": config.save,
        "save_conf": config.save_conf,
        "save_txt": config.save_txt,
        "show_conf": config.show_conf,
        "show_labels": config.show_labels,
        "show_boxes": config.show_boxes,
    }
    if config.project:
        predict_kwargs["project"] = config.project
    if config.name:
        predict_kwargs["name"] = config.name

    return model.predict(**predict_kwargs)


def best_weights_from_train_results(results) -> Optional[Path]:
    save_dir = getattr(results, "save_dir", None)
    if save_dir is None:
        return None
    candidate = Path(save_dir) / "weights" / "best.pt"
    return candidate if candidate.exists() else None


# -----------------------------------------------------------------------------
# Post-processing / crop extraction
# -----------------------------------------------------------------------------

def crop_yolo_predictions(config: CropConfig) -> int:
    labels_dir = config.labels_dir.expanduser()
    images_dir = config.images_dir.expanduser()
    output_dir = config.output_dir.expanduser()
    ensure_directory(output_dir)

    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {absolute(labels_dir)}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {absolute(images_dir)}")

    label_files = sorted(labels_dir.glob("*.txt"))
    crop_count = 0

    for label_path in tqdm(label_files, desc="cropping", unit="label"):
        image_path = find_image_by_stem(images_dir, label_path.stem)
        if image_path is None:
            print(f"WARNING: No image found for label file {label_path.name}, skipping.")
            continue

        image = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size
        lines = label_path.read_text(encoding="utf-8").splitlines()

        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            # YOLO format: class x_center y_center width height [confidence]
            x_center, y_center, width, height = map(float, parts[1:5])
            x_center *= img_w
            y_center *= img_h
            width *= img_w
            height *= img_h

            left = max(0, int(x_center - width / 2))
            top = max(0, int(y_center - height / 2))
            right = min(img_w, int(x_center + width / 2))
            bottom = min(img_h, int(y_center + height / 2))

            if right <= left or bottom <= top:
                continue

            cropped_img = image.crop((left, top, right, bottom))
            output_path = output_dir / f"{image_path.stem}_crop_{i}.{config.output_format.lower()}"
            cropped_img.save(output_path)
            crop_count += 1

    print(f"Saved {crop_count} crops to: {absolute(output_dir)}")
    return crop_count


# -----------------------------------------------------------------------------
# CLI parsing
# -----------------------------------------------------------------------------

def parse_target_size(values: Optional[list[int]]) -> Optional[tuple[int, int]]:
    if values is None:
        return (1920, 1080)
    if len(values) == 0:
        return None
    if len(values) != 2:
        raise argparse.ArgumentTypeError("Target size must be two integers: WIDTH HEIGHT")
    return (values[0], values[1])


def add_preprocess_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--target-size", nargs="*", type=int, default=[1920, 1080], help="WIDTH HEIGHT. Use --target-size with no values to keep original sizes.")
    parser.add_argument("--clahe-clip-limit", type=float, default=3.0)
    parser.add_argument("--clahe-tile-grid", nargs=2, type=int, default=[8, 8])
    parser.add_argument("--use-super-resolution", action="store_true")
    parser.add_argument("--sr-scale", type=int, choices=[2, 4], default=2)
    parser.add_argument("--no-sr-half", action="store_true")
    parser.add_argument("--output-format", choices=["png", "jpg", "jpeg"], default="png")
    parser.add_argument("--overwrite-originals", action="store_true")
    parser.add_argument("--metrics-csv-name", default="enhancement_metrics.csv")
    parser.add_argument("--no-enhanced-yaml", action="store_true")
    parser.add_argument("--no-copy-labels-for-enhanced", action="store_true")
    parser.add_argument("--nc", type=int, default=1)
    parser.add_argument("--names", nargs="+", default=["fish"])
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument("--show-plots", action="store_true")


def add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="yolo26x.pt")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET_ROOT / "data.yaml")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--imgsz", nargs=2, type=int, default=[960, 540], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--no-cos-lr", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--weight-decay", type=float, default=0.005)
    parser.add_argument("--no-val", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--save-period", type=int, default=-1)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-rect", action="store_true")
    parser.add_argument("--bgr", type=float, default=0.3)
    parser.add_argument("--translate", type=float, default=0.05)
    parser.add_argument("--scale", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--mosaic", type=float, default=0.7)
    parser.add_argument("--hsv-h", type=float, default=0.01)
    parser.add_argument("--hsv-s", type=float, default=0.5)
    parser.add_argument("--hsv-v", type=float, default=0.4)
    parser.add_argument("--flipud", type=float, default=0.1)
    parser.add_argument("--fliplr", type=float, default=0.7)
    parser.add_argument("--cutmix", type=float, default=0.0)
    parser.add_argument("--copy-paste", type=float, default=0.0)
    parser.add_argument("--shear", type=float, default=3.0)
    parser.add_argument("--degrees", type=float, default=5.0)


def add_infer_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--weights", type=Path, default=Path("runs/detect/train-1/weights/best.pt"))
    parser.add_argument("--source", type=Path, default=DEFAULT_DATASET_ROOT / "test/images")
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--no-save-conf", action="store_true")
    parser.add_argument("--no-save-txt", action="store_true")
    parser.add_argument("--no-show-conf", action="store_true")
    parser.add_argument("--no-show-labels", action="store_true")
    parser.add_argument("--no-show-boxes", action="store_true")
    parser.add_argument("--project", default=None)
    parser.add_argument("--name", default=None)


def add_crop_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--labels-dir", type=Path, default=Path("runs/detect/predict/labels"))
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_DATASET_ROOT / "test/images")
    parser.add_argument("--output-dir", type=Path, default=Path("crops"))
    parser.add_argument("--crop-output-format", choices=["png", "jpg", "jpeg"], default="png")


def preprocess_config_from_args(args: argparse.Namespace) -> PreprocessConfig:
    return PreprocessConfig(
        dataset_root=args.dataset_root,
        splits=list(args.splits),
        target_size=parse_target_size(args.target_size),
        clahe_clip_limit=args.clahe_clip_limit,
        clahe_tile_grid=tuple(args.clahe_tile_grid),
        use_super_resolution=args.use_super_resolution,
        sr_scale=args.sr_scale,
        sr_half=not args.no_sr_half,
        output_format=args.output_format,
        overwrite_originals=args.overwrite_originals,
        metrics_csv_name=args.metrics_csv_name,
        create_enhanced_yaml=not args.no_enhanced_yaml,
        copy_labels_for_enhanced=not args.no_copy_labels_for_enhanced,
        nc=args.nc,
        names=list(args.names),
        save_plots=args.save_plots,
        show_plots=args.show_plots,
    )


def train_config_from_args(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        model=args.model,
        data=args.data,
        epochs=args.epochs,
        imgsz=tuple(args.imgsz),
        device=args.device,
        batch=args.batch,
        workers=args.workers,
        amp=not args.no_amp,
        cache=not args.no_cache,
        pretrained=not args.no_pretrained,
        optimizer=args.optimizer,
        cos_lr=not args.no_cos_lr,
        seed=args.seed,
        patience=args.patience,
        weight_decay=args.weight_decay,
        val=not args.no_val,
        plots=not args.no_plots,
        save_period=args.save_period,
        verbose=not args.quiet,
        augment=not args.no_augment,
        rect=not args.no_rect,
        bgr=args.bgr,
        translate=args.translate,
        scale=args.scale,
        dropout=args.dropout,
        mosaic=args.mosaic,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        flipud=args.flipud,
        fliplr=args.fliplr,
        cutmix=args.cutmix,
        copy_paste=args.copy_paste,
        shear=args.shear,
        degrees=args.degrees,
    )


def inference_config_from_args(args: argparse.Namespace) -> InferenceConfig:
    return InferenceConfig(
        weights=args.weights,
        source=args.source,
        conf=args.conf,
        save=not args.no_save,
        save_conf=not args.no_save_conf,
        save_txt=not args.no_save_txt,
        show_conf=not args.no_show_conf,
        show_labels=not args.no_show_labels,
        show_boxes=not args.no_show_boxes,
        project=args.project,
        name=args.name,
    )


def crop_config_from_args(args: argparse.Namespace) -> CropConfig:
    return CropConfig(
        labels_dir=args.labels_dir,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        output_format=args.crop_output_format,
    )


def run_all(args: argparse.Namespace) -> None:
    preprocess_dataset(preprocess_config_from_args(args))

    train_config = train_config_from_args(args)
    if getattr(args, "train_on_enhanced", False):
        train_config.data = args.dataset_root / "data_enhanced.yaml"

    results = train_yolo(train_config)
    trained_weights = best_weights_from_train_results(results)

    inference_config = inference_config_from_args(args)
    if trained_weights is not None:
        inference_config.weights = trained_weights

    infer_yolo(inference_config)
    crop_yolo_predictions(crop_config_from_args(args))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Consolidated preprocessing, YOLO training, inference, and crop extraction pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess_parser = subparsers.add_parser("preprocess", help="Enhance images and write quality metrics.")
    add_preprocess_args(preprocess_parser)

    train_parser = subparsers.add_parser("train", help="Train YOLO.")
    add_train_args(train_parser)

    infer_parser = subparsers.add_parser("infer", help="Run YOLO prediction.")
    add_infer_args(infer_parser)

    crop_parser = subparsers.add_parser("crop", help="Crop detected ROIs from YOLO label files.")
    add_crop_args(crop_parser)

    all_parser = subparsers.add_parser("all", help="Run preprocess -> train -> infer -> crop.")
    add_preprocess_args(all_parser)
    add_train_args(all_parser)
    add_infer_args(all_parser)
    add_crop_args(all_parser)
    all_parser.add_argument("--train-on-enhanced", action="store_true", help="After preprocessing, train using data_enhanced.yaml instead of data.yaml.")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "preprocess":
        preprocess_dataset(preprocess_config_from_args(args))
    elif args.command == "train":
        train_yolo(train_config_from_args(args))
    elif args.command == "infer":
        infer_yolo(inference_config_from_args(args))
    elif args.command == "crop":
        crop_yolo_predictions(crop_config_from_args(args))
    elif args.command == "all":
        run_all(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    freeze_support()
    main()
