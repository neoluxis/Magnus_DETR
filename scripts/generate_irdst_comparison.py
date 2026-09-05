#!/usr/bin/env python3
"""
Run inference for IRDST-trained models on 3 randomly-selected test images.
Save individual subplot PNGs + original images, then assemble a comparison grid.
"""

import random
import shutil
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_project_root = Path(__file__).resolve().parent.parent
_ugit = _project_root / "ultralytics-git"
if str(_ugit) not in sys.path:
    sys.path.insert(0, str(_ugit))

from ultralytics import YOLO, RTDETR

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT = _project_root

# IRDST-trained models (b3 has no weights, skip it)
MODELS: list[dict] = [
    {"name": "YOLOv5s",       "cls": "yolo",   "ckpt": "runs/detect/irdst20k/yolov5s/weights/best.pt"},
    {"name": "YOLOv5s(P2)",   "cls": "yolo",   "ckpt": "runs/detect/irdst20k/yolov5s-p2/weights/best.pt"},
    {"name": "YOLOv8s",       "cls": "yolo",   "ckpt": "runs/detect/irdst20k/yolov8s/weights/best.pt"},
    {"name": "YOLOv8s(P2)",   "cls": "yolo",   "ckpt": "runs/detect/irdst20k/yolov8s-p2/weights/best.pt"},
    {"name": "YOLO11s",       "cls": "yolo",   "ckpt": "runs/detect/irdst20k/yolo11s/weights/best.pt"},
    {"name": "YOLO11s(P2)",   "cls": "yolo",   "ckpt": "runs/detect/irdst20k/yolo11s-p2/weights/best.pt"},
    {"name": "YOLO26s",       "cls": "yolo",   "ckpt": "runs/detect/irdst20k/yolo26s/weights/best.pt"},
    {"name": "YOLO26s(P2)",   "cls": "yolo",   "ckpt": "runs/detect/irdst20k/yolo26s-p2/weights/best.pt"},
    {"name": "RT-DETR-L",     "cls": "rtdetr", "ckpt": "runs/detect/irdst20k/rtdetr-l/weights/best.pt"},
    {"name": "RT-DETR-R18",   "cls": "rtdetr", "ckpt": "runs/detect/irdst20k/rtdetr-r18/weights/best.pt"},
    {"name": "RT-DETR-R18(P2)","cls": "rtdetr", "ckpt": "runs/detect/irdst20k/rtdetr-r18-p2/weights/best.pt"},
    {"name": "RT-DETR-R50",   "cls": "rtdetr", "ckpt": "runs/detect/irdst20k/rtdetr-r50/weights/best.pt"},
    {"name": "RT-DETR-R101",  "cls": "rtdetr", "ckpt": "runs/detect/irdst20k/rtdetr-r101/weights/best.pt"},
    {"name": "B5",            "cls": "rtdetr", "ckpt": "runs/detect/irdst20k/b5/weights/best.pt"},
]

VAL_IMG_DIR = ROOT / "datasets" / "IRDST_real_yolo" / "images" / "val"
OUT_DIR = ROOT / "assets" / "figures" / "irdst_subplots"
GRID_OUT = ROOT / "assets" / "figures" / "irdst_comparison_grid.png"

IMGSZ = 640
CONF = 0.25
DEVICE = "0"
SEED = 42
NUM_IMAGES = 3

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def pick_random_images(n: int) -> list[Path]:
    random.seed(SEED)
    all_imgs = sorted(VAL_IMG_DIR.glob("*.png"))
    if len(all_imgs) < n:
        raise RuntimeError(f"Only {len(all_imgs)} images available, need {n}")
    chosen = random.sample(all_imgs, n)
    print(f"Randomly selected (seed={SEED}):")
    for p in chosen:
        print(f"  {p.name}")
    return chosen


def load_model(entry: dict):
    ckpt = ROOT / entry["ckpt"]
    if entry["cls"] == "yolo":
        return YOLO(str(ckpt))
    else:
        return RTDETR(str(ckpt))


def run_inference(model, img_path: Path) -> np.ndarray:
    results = model(str(img_path), imgsz=IMGSZ, conf=CONF, device=DEVICE, verbose=False)
    annotated = results[0].plot(conf=True, labels=True, boxes=True)
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Figure assembly (2 rows × 7 cols)
# ---------------------------------------------------------------------------
def draw_block(gs_slice, models: list[dict], images: list[Path],
               grid: list[list[np.ndarray]], cell_w: int, thumb_h: int):
    n_cols = len(models)
    n_rows = len(images)
    label_w = 100

    inner_gs = gs_slice.subgridspec(
        nrows=1 + n_rows, ncols=1 + n_cols,
        height_ratios=[0.08] + [1] * n_rows,
        width_ratios=[label_w / (label_w + n_cols * cell_w)] + [1] * n_cols,
        hspace=0.02, wspace=0.02,
    )

    for j, entry in enumerate(models):
        ax = plt.subplot(inner_gs[0, 1 + j])
        ax.axis("off")
        ax.text(0.5, 0.5, entry["name"], transform=ax.transAxes,
                ha="center", va="center", fontsize=6.5, fontweight="bold", linespacing=1.1)
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_color("#aaaaaa"); spine.set_linewidth(0.5)

    for i, img_path in enumerate(images):
        ax = plt.subplot(inner_gs[1 + i, 0])
        ax.axis("off")
        ax.text(0.5, 0.5, img_path.name, transform=ax.transAxes,
                ha="center", va="center", fontsize=7, fontstyle="italic")
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_color("#aaaaaa"); spine.set_linewidth(0.5)

    for i in range(n_rows):
        for j in range(n_cols):
            ax = plt.subplot(inner_gs[1 + i, 1 + j])
            ax.axis("off")
            img = grid[i][j]
            if img is not None:
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "ERROR", transform=ax.transAxes,
                        ha="center", va="center", fontsize=5, color="red")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    images = pick_random_images(NUM_IMAGES)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    thumb_h = 450
    n_models = len(MODELS)
    n_images = len(images)

    # Pre-allocate grid
    grid = [[None for _ in range(n_models)] for _ in range(n_images)]
    cell_w = int(640 * thumb_h / 640)  # default aspect ratio

    # ---- Save original images ----
    print("\n--- Saving original images ---")
    for img_path in images:
        img_dir = OUT_DIR / img_path.stem
        img_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, img_dir / f"original{img_path.suffix}")
        print(f"  {img_path.stem}/original{img_path.suffix}")

    # ---- Run inference ----
    for j, entry in enumerate(MODELS):
        name_flat = entry["name"].replace("\n", " ")
        print(f"\n[{j+1}/{n_models}] {name_flat} ...")
        model = load_model(entry)

        for i, img_path in enumerate(images):
            stem = img_path.stem
            print(f"  {stem} inferring...")

            try:
                annotated = run_inference(model, img_path)
                grid[i][j] = annotated
                if i == 0 and j == 0:
                    h, w = annotated.shape[:2]
                    cell_w = int(w * thumb_h / h)

                # Save individual subplot
                img_dir = OUT_DIR / stem
                safe_name = entry["name"].replace("\n", "").replace("/", "_")
                out_path = img_dir / f"{safe_name}.png"
                cv2.imwrite(str(out_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
            except Exception as e:
                print(f"    ERROR: {e}")
                grid[i][j] = None

    # ---- Assemble comparison grid ----
    print("\n--- Assembling comparison grid ---")
    COLS_PER_ROW = 7
    row1_models = MODELS[:COLS_PER_ROW]
    row2_models = MODELS[COLS_PER_ROW:]
    row1_grid = [row[:COLS_PER_ROW] for row in grid]
    row2_grid = [row[COLS_PER_ROW:] for row in grid]

    label_w = 100
    dpi = 150
    header_ratio = 0.08
    img_ratio = 1.0
    spacer_ratio = 0.05

    fig_w_px = label_w + COLS_PER_ROW * cell_w
    fig_h_px = (2 * header_ratio + 2 * n_images * img_ratio + spacer_ratio) * thumb_h

    fig = plt.figure(figsize=(fig_w_px / dpi, fig_h_px / dpi), dpi=dpi)
    gs = fig.add_gridspec(
        nrows=2, ncols=1,
        height_ratios=[header_ratio + n_images * img_ratio] * 2,
        hspace=spacer_ratio / (2 * header_ratio + 2 * n_images * img_ratio) * 2,
        top=0.98, bottom=0.02, left=0.03, right=0.99,
    )

    draw_block(gs[0], row1_models, images, row1_grid, cell_w, thumb_h)
    draw_block(gs[1], row2_models, images, row2_grid, cell_w, thumb_h)

    fig.suptitle("Model Comparison — IRDST-20k Test Images",
                 fontsize=13, fontweight="bold", y=0.998)

    GRID_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(GRID_OUT, dpi=dpi, bbox_inches="tight", facecolor="white",
                edgecolor="none", pad_inches=0.3)
    plt.close(fig)

    # ---- Summary ----
    print(f"\n=== Done ===")
    print(f"Subplots: {OUT_DIR}/")
    for img_path in images:
        n_files = len(list((OUT_DIR / img_path.stem).glob("*.png")))
        print(f"  {img_path.stem}/  ({n_files} files: original + {n_models} models)")
    print(f"Grid: {GRID_OUT}")


if __name__ == "__main__":
    main()
