#!/usr/bin/env python3
"""Evaluate size-stratified AP50 for tiny-object Anti-DETR experiments.

Bins default to maximum box side length in pixels:
- 1-4 px
- 5-8 px
- 9-16 px
- >16 px
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import numpy as np
from ultralytics import RTDETR, YOLO


IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}
CST_DATASET_ROOTS = [
    Path("../datasets/CST_AntiUAV/CST-AntiUAV"),
    Path("datasets/CST_AntiUAV/CST-AntiUAV"),
]


@dataclass
class GTBox:
    image_id: str
    box: np.ndarray
    bin_name: str


@dataclass
class PredBox:
    image_id: str
    box: np.ndarray
    score: float


def parse_args():
    parser = argparse.ArgumentParser(description="Compute size-stratified AP50.")
    parser.add_argument("--model", required=True, help="Model weights or run directory.")
    parser.add_argument("--source", required=True, help="Image file or directory.")
    parser.add_argument("--arch", default="auto", choices=("auto", "yolo", "detr"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--outdir", default="runs/size_stratified_eval")
    parser.add_argument("--max-images", type=int, default=0, help="0 means use all images.")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.5)
    return parser.parse_args()


def infer_arch(model_path: str, arch: str) -> str:
    if arch != "auto":
        return arch
    return "detr" if "detr" in model_path.lower() else "yolo"


def resolve_model_path(model_path: str) -> str:
    path = Path(model_path)
    if path.is_file():
        return str(path)
    if path.is_dir():
        for candidate in (path / "weights" / "best.pt", path / "best.pt"):
            if candidate.is_file():
                return str(candidate)
    return model_path


def resolve_source_path(source: str) -> Path:
    path = Path(source)
    if path.exists():
        return path
    normalized = source.replace("\\", "/")
    marker = "CST-AntiUAV/"
    if marker in normalized:
        rel = normalized.split(marker, 1)[1]
        for root in CST_DATASET_ROOTS:
            candidate = (root / rel).resolve()
            if candidate.exists():
                return candidate
    return path


def collect_images(source: str, max_images: int) -> list[Path]:
    path = resolve_source_path(source)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Source not found: {source}")
    images = [p for p in sorted(path.rglob("*")) if p.suffix.lower() in IMAGE_SUFFIXES]
    return images if max_images <= 0 else images[:max_images]


def infer_label_path(image_path: Path) -> Path | None:
    if (image_path.parent / "gt.txt").exists():
        return image_path.parent / "gt.txt"
    parts = list(image_path.parts)
    if "images" not in parts:
        return None
    idx = parts.index("images")
    label_parts = parts.copy()
    label_parts[idx] = "labels"
    return Path(*label_parts).with_suffix(".txt")


def load_cst_gt_boxes(image_path: Path) -> list[np.ndarray]:
    gt_path = image_path.parent / "gt.txt"
    if not gt_path.exists() or not image_path.stem.isdigit():
        return []
    frame_idx = int(image_path.stem)
    lines = gt_path.read_text(encoding="utf-8").splitlines()
    if frame_idx >= len(lines):
        return []
    parts = [p.strip() for p in lines[frame_idx].split(",")]
    if len(parts) < 4:
        return []
    x, y, w, h = map(float, parts[:4])
    if x == 0 and y == 0 and w == 0 and h == 0:
        return []
    return [np.array([x, y, x + w, y + h], dtype=np.float32)]


def load_gt_boxes(image_path: Path, image_shape: tuple[int, int]) -> list[np.ndarray]:
    label_path = infer_label_path(image_path)
    if label_path is None or not label_path.exists():
        return []
    if label_path.name == "gt.txt":
        return load_cst_gt_boxes(image_path)
    h, w = image_shape[:2]
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cls_id, xc, yc, bw, bh = map(float, line.split())
        del cls_id
        x1 = (xc - bw / 2.0) * w
        y1 = (yc - bh / 2.0) * h
        x2 = (xc + bw / 2.0) * w
        y2 = (yc + bh / 2.0) * h
        boxes.append(np.array([x1, y1, x2, y2], dtype=np.float32))
    return boxes


def max_side_bin(box: np.ndarray) -> str:
    w = max(float(box[2] - box[0]), 0.0)
    h = max(float(box[3] - box[1]), 0.0)
    side = max(w, h)
    if side <= 4:
        return "1-4"
    if side <= 8:
        return "5-8"
    if side <= 16:
        return "9-16"
    return ">16"


def box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, float(box1[2] - box1[0])) * max(0.0, float(box1[3] - box1[1]))
    area2 = max(0.0, float(box2[2] - box2[0])) * max(0.0, float(box2[3] - box2[1]))
    return inter / (area1 + area2 - inter + 1e-6)


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def main():
    args = parse_args()
    arch = infer_arch(args.model, args.arch)
    model_path = resolve_model_path(args.model)
    wrapper = RTDETR(model_path) if arch == "detr" else YOLO(model_path)
    if args.device is not None:
        wrapper.to(args.device)

    bins = ["1-4", "5-8", "9-16", ">16"]
    gt_by_bin: dict[str, list[GTBox]] = {k: [] for k in bins}
    pred_by_bin: dict[str, list[PredBox]] = {k: [] for k in bins}

    images = collect_images(args.source, args.max_images)
    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        image_id = str(image_path)
        gt_boxes = load_gt_boxes(image_path, image.shape)
        gt_bins = [max_side_bin(box) for box in gt_boxes]
        for box, bin_name in zip(gt_boxes, gt_bins):
            gt_by_bin[bin_name].append(GTBox(image_id=image_id, box=box, bin_name=bin_name))

        results = wrapper.predict(
            source=image,
            imgsz=args.imgsz,
            conf=args.conf,
            verbose=False,
            device=args.device,
        )
        if not results:
            continue
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.detach().cpu().numpy()
        scores = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy), dtype=np.float32)
        for pred_box, score in zip(xyxy, scores):
            if gt_boxes:
                best_gt_idx = int(np.argmax([box_iou(pred_box, gt_box) for gt_box in gt_boxes]))
                pred_bin = gt_bins[best_gt_idx]
            else:
                pred_bin = ">16"
            pred_by_bin[pred_bin].append(PredBox(image_id=image_id, box=pred_box, score=float(score)))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for bin_name in bins:
        gts = gt_by_bin[bin_name]
        preds = sorted(pred_by_bin[bin_name], key=lambda x: x.score, reverse=True)
        matched: dict[str, set[int]] = {}
        tps = []
        fps = []

        gt_lookup: dict[str, list[np.ndarray]] = {}
        for gt in gts:
            gt_lookup.setdefault(gt.image_id, []).append(gt.box)

        for pred in preds:
            image_gts = gt_lookup.get(pred.image_id, [])
            best_iou = 0.0
            best_idx = -1
            for idx, gt_box in enumerate(image_gts):
                iou = box_iou(pred.box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            used = matched.setdefault(pred.image_id, set())
            if best_iou >= args.iou and best_idx >= 0 and best_idx not in used:
                used.add(best_idx)
                tps.append(1.0)
                fps.append(0.0)
            else:
                tps.append(0.0)
                fps.append(1.0)

        if preds and gts:
            tps_cum = np.cumsum(np.array(tps, dtype=np.float32))
            fps_cum = np.cumsum(np.array(fps, dtype=np.float32))
            recalls = tps_cum / max(len(gts), 1)
            precisions = tps_cum / np.maximum(tps_cum + fps_cum, 1e-6)
            ap50 = compute_ap(recalls, precisions)
            recall = float(recalls[-1])
            precision = float(precisions[-1])
        else:
            ap50 = 0.0
            recall = 0.0
            precision = 0.0

        rows.append(
            {
                "size_bin_px": bin_name,
                "gt_count": len(gts),
                "pred_count": len(preds),
                "precision_final": precision,
                "recall_final": recall,
                "ap50": ap50,
                "side_rule": "max(width,height)",
            }
        )

    csv_path = outdir / "size_stratified_ap50.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["size_bin_px", "gt_count", "pred_count", "precision_final", "recall_final", "ap50", "side_rule"],
        )
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "# Size-Stratified AP50",
        "",
        "| Size bin (px) | GT count | Pred count | Precision | Recall | AP50 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['size_bin_px']} | {row['gt_count']} | {row['pred_count']} | "
            f"{row['precision_final']:.4f} | {row['recall_final']:.4f} | {row['ap50']:.4f} |"
        )
    md_lines += [
        "",
        "Side-length rule: `max(width, height)` in original-image pixels.",
        f"IoU threshold: `{args.iou:.2f}`.",
    ]
    (outdir / "size_stratified_ap50.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"saved size-stratified metrics to {csv_path}")


if __name__ == "__main__":
    main()
