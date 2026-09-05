#!/usr/bin/env python3
"""Analyze module-internal gates and confidence maps for Anti-DETR variants.

This script avoids using CAM for modules whose gating is not spatial:
- DWGConv / DWGConvMS / DWGConvSA: export channel-gate statistics and distributions
- MDHIFI: export spatial gate maps and Target-to-Clutter Response Ratio (TCRR)
- HFSCC: export cross-scale consistency/confidence maps C_{i,i+1} and TCRR

Example:
  PYTHONPATH=ultralytics-git python scripts/analyze_module_responses.py \
    --model runs/detect/0705/b5-3 \
    --source ../datasets/CST_AntiUAV/CST-AntiUAV/val \
    --outdir runs/module_response_analysis/0705_b5_3
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from ultralytics import RTDETR, YOLO
from ultralytics.data.augment import LetterBox


IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}
CST_DATASET_ROOTS = [
    Path("../datasets/CST_AntiUAV/CST-AntiUAV"),
    Path("datasets/CST_AntiUAV/CST-AntiUAV"),
]


@dataclass
class HookRecord:
    module_name: str
    kind: str
    tensor: torch.Tensor


class HookCollector:
    def __init__(self):
        self.records: list[HookRecord] = []
        self.handles = []

    def add(self, module: torch.nn.Module, module_name: str, kind: str):
        def _hook(_module, _inputs, output):
            if isinstance(output, torch.Tensor):
                self.records.append(HookRecord(module_name=module_name, kind=kind, tensor=output.detach()))

        self.handles.append(module.register_forward_hook(_hook))

    def clear(self):
        self.records.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Anti-DETR internal gates and consistency maps.")
    parser.add_argument("--model", required=True, help="Model weights or run directory.")
    parser.add_argument("--source", required=True, help="Image file or directory.")
    parser.add_argument("--arch", default="auto", choices=("auto", "yolo", "detr"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--outdir", default="runs/module_response_analysis")
    parser.add_argument("--max-images", type=int, default=40)
    parser.add_argument(
        "--scene-tags",
        nargs="*",
        default=["building", "urban", "jungle", "sky"],
        help="Scene tags matched by substring against image path. Non-matching images go to 'other'.",
    )
    parser.add_argument("--box-source", default="auto", choices=("auto", "gt", "pred"))
    parser.add_argument("--pred-conf", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.42)
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
    return images[:max_images]


def load_wrapper(model_path: str, arch: str, device: str | None):
    resolved = resolve_model_path(model_path)
    wrapper = RTDETR(resolved) if arch == "detr" else YOLO(resolved)
    if device is not None:
        wrapper.to(device)
    wrapper.model.eval()
    return wrapper


def infer_model_channels(core_model) -> int:
    yaml_channels = getattr(getattr(core_model, "yaml", None), "get", lambda *_: None)("channels")
    if yaml_channels is not None:
        return int(yaml_channels)
    return 3


def preprocess_image(image_bgr: np.ndarray, arch: str, imgsz: int, stride, input_channels: int = 3) -> torch.Tensor:
    if arch == "detr":
        letterbox = LetterBox(imgsz, auto=False, scale_fill=True)
    else:
        stride = int(stride.max().item()) if isinstance(stride, torch.Tensor) else int(stride)
        letterbox = LetterBox(imgsz, auto=False, stride=stride)
    transformed = letterbox(image=image_bgr)
    if input_channels == 1:
        transformed = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY)[..., None]
    else:
        transformed = transformed[..., ::-1]
    transformed = transformed.transpose(2, 0, 1)
    transformed = np.ascontiguousarray(transformed)
    return torch.from_numpy(transformed).float().unsqueeze(0) / 255.0


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


def load_cst_gt_boxes(image_path: Path, gt_path: Path) -> list[np.ndarray]:
    if not image_path.stem.isdigit():
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
        return load_cst_gt_boxes(image_path, label_path)
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


def choose_target_box(wrapper, image_bgr: np.ndarray, image_path: Path, box_source: str, pred_conf: float) -> np.ndarray | None:
    gt_boxes = load_gt_boxes(image_path, image_bgr.shape)
    if box_source in {"auto", "gt"} and gt_boxes:
        return gt_boxes[0]
    if box_source == "gt":
        return None

    results = wrapper.predict(
        source=image_bgr,
        imgsz=max(image_bgr.shape[:2]),
        conf=pred_conf,
        verbose=False,
        device=getattr(wrapper, "device", None),
    )
    if not results:
        return gt_boxes[0] if gt_boxes and box_source == "auto" else None
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return gt_boxes[0] if gt_boxes and box_source == "auto" else None
    xyxy = boxes.xyxy.detach().cpu().numpy()
    confs = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy), dtype=np.float32)
    return xyxy[int(np.argmax(confs))]


def map_box_from_orig_to_input(box_xyxy: np.ndarray, orig_shape: tuple[int, int], input_shape: tuple[int, int], arch: str) -> np.ndarray:
    box = box_xyxy.astype(np.float32).copy()
    oh, ow = orig_shape[:2]
    ih, iw = input_shape[:2]
    if arch == "detr":
        box[[0, 2]] *= iw / ow
        box[[1, 3]] *= ih / oh
        return box
    gain = min(iw / ow, ih / oh)
    new_w = round(ow * gain)
    new_h = round(oh * gain)
    pad_w = (iw - new_w) / 2.0
    pad_h = (ih - new_h) / 2.0
    box[[0, 2]] = box[[0, 2]] * gain + pad_w
    box[[1, 3]] = box[[1, 3]] * gain + pad_h
    return box


def classify_scene(image_path: Path, scene_tags: list[str]) -> str:
    lower = str(image_path).lower()
    for tag in scene_tags:
        if tag.lower() in lower:
            return tag.lower()
    return "other"


def collect_module_hooks(core_model) -> HookCollector:
    collector = HookCollector()
    for module_name, module in core_model.named_modules():
        module_type = module.__class__.__name__
        if module_type in {"DWGConv", "DWGConvSA"} and hasattr(module, "gate"):
            collector.add(module.gate, module_name, "dwgconv_gate")
            if hasattr(module, "spatial_gate"):
                collector.add(module.spatial_gate, module_name, "dwgconv_spatial")
        elif module_type == "DWGConvMS" and hasattr(module, "branch_gate"):
            collector.add(module.branch_gate, module_name, "dwgconvms_gate")
        elif module_type == "MDHIFI":
            if hasattr(module, "noise_gate"):
                collector.add(module.noise_gate, module_name, "mdhifi_gate")
            # also hook gradient and texture projected outputs if present
            if hasattr(module, "edge_proj"):
                collector.add(module.edge_proj, module_name, "mdhifi_grad")
            if hasattr(module, "texture_proj"):
                collector.add(module.texture_proj, module_name, "mdhifi_texture")
        elif module_type == "HFSCC":
            collector.add(module.consistency4, module_name, "hfscc_c4")
            collector.add(module.consistency5, module_name, "hfscc_c5")
    return collector


def tensor_to_spatial_map(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.squeeze(0).detach().cpu().float().numpy()
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim == 3:
        arr = arr.mean(axis=0)
    elif arr.ndim == 2:
        pass
    else:
        arr = arr.reshape(1, -1)
    arr = np.maximum(arr, 0)
    denom = arr.max() - arr.min()
    if denom > 1e-8:
        arr = (arr - arr.min()) / denom
    else:
        arr = np.zeros_like(arr)
    return arr


def compute_tcrr(spatial_map: np.ndarray, box_xyxy: np.ndarray) -> float:
    h, w = spatial_map.shape[:2]
    x1, y1, x2, y2 = box_xyxy
    x1 = int(np.clip(math.floor(x1), 0, w - 1))
    y1 = int(np.clip(math.floor(y1), 0, h - 1))
    x2 = int(np.clip(math.ceil(x2), x1 + 1, w))
    y2 = int(np.clip(math.ceil(y2), y1 + 1, h))
    target = spatial_map[y1:y2, x1:x2]
    clutter_mask = np.ones((h, w), dtype=bool)
    clutter_mask[y1:y2, x1:x2] = False
    clutter = spatial_map[clutter_mask]
    if target.size == 0 or clutter.size == 0:
        return float("nan")
    return float(target.mean() / (clutter.mean() + 1e-6))


def overlay_heatmap(image_bgr: np.ndarray, spatial_map: np.ndarray, alpha: float) -> np.ndarray:
    heat = cv2.resize(spatial_map, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    heat = np.uint8(np.clip(heat, 0, 1) * 255)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
    return cv2.addWeighted(image_bgr, 1.0 - alpha, heat, alpha, 0)


def draw_box(image_bgr: np.ndarray, box_xyxy: np.ndarray, color=(40, 255, 80)) -> np.ndarray:
    canvas = image_bgr.copy()
    x1, y1, x2, y2 = box_xyxy.astype(int).tolist()
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    return canvas


def save_spatial_panel(out_path: Path, image_bgr: np.ndarray, spatial_map: np.ndarray, box_xyxy: np.ndarray, tcrr: float, alpha: float):
    boxed = draw_box(image_bgr, box_xyxy)
    overlay = draw_box(overlay_heatmap(image_bgr, spatial_map, alpha=alpha), box_xyxy)
    map_vis = cv2.resize(np.uint8(spatial_map * 255), (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    map_vis = cv2.applyColorMap(map_vis, cv2.COLORMAP_TURBO)
    panel = np.concatenate([boxed, overlay, map_vis], axis=1)
    cv2.putText(
        panel,
        f"TCRR={tcrr:.3f}" if np.isfinite(tcrr) else "TCRR=nan",
        (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), panel)


def save_gate_distribution_plot(scene_gate_values: dict[str, list[np.ndarray]], out_path: Path, title: str):
    if not scene_gate_values:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for scene, arrays in sorted(scene_gate_values.items()):
        values = np.concatenate([a.reshape(-1) for a in arrays], axis=0)
        if values.size == 0:
            continue
        ax.hist(values, bins=30, alpha=0.45, density=True, label=scene)
    ax.set_title(title)
    ax.set_xlabel("Gate weight")
    ax.set_ylabel("Density")
    ax.set_xlim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_dwgconv_scene_bar(scene_gate_values: dict[str, list[np.ndarray]], out_path: Path, title: str):
    rows = []
    for scene, arrays in sorted(scene_gate_values.items()):
        values = np.concatenate([a.reshape(-1) for a in arrays], axis=0)
        if values.size:
            rows.append((scene, float(values.mean()), float(values.std())))
    if not rows:
        return
    labels = [r[0] for r in rows]
    means = [r[1] for r in rows]
    stds = [r[2] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, means, yerr=stds, color="#3a6ea5", alpha=0.9)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Mean gate weight")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    arch = infer_arch(args.model, args.arch)
    wrapper = load_wrapper(args.model, arch, args.device)
    core_model = wrapper.model
    images = collect_images(args.source, args.max_images)
    collector = collect_module_hooks(core_model)

    channel_gate_by_module_scene: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    branch_gate_by_module_scene: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    spatial_rows: list[dict[str, object]] = []

    try:
        for image_path in images:
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                continue
            scene = classify_scene(image_path, args.scene_tags)
            target_box = choose_target_box(wrapper, image_bgr, image_path, args.box_source, args.pred_conf)
            if target_box is None:
                continue

            stride = getattr(core_model, "stride", 32)
            input_channels = infer_model_channels(core_model)
            tensor = preprocess_image(image_bgr, arch, args.imgsz, stride, input_channels=input_channels)
            if args.device is not None:
                tensor = tensor.to(args.device)

            collector.clear()
            with torch.no_grad():
                _ = core_model(tensor)

            input_hw = tuple(int(x) for x in tensor.shape[-2:])
            target_box_input = map_box_from_orig_to_input(target_box, image_bgr.shape[:2], input_hw, arch)

            stem = f"{image_path.parent.name}__{image_path.stem}"
            # group records by module for joint processing (gate, grad, texture)
            from collections import defaultdict

            module_outputs: dict[str, dict[str, object]] = defaultdict(dict)
            for record in collector.records:
                module_outputs[record.module_name][record.kind] = record.tensor.detach().cpu()

            # load other gt boxes to exclude from clutter
            other_gts = load_gt_boxes(image_path, image_bgr.shape[:2])
            # remove the chosen target from other_gts if it matches
            # (we'll exclude any boxes that overlap heavily)

            def overlaps(a, b, thr=0.5):
                ax1, ay1, ax2, ay2 = a
                bx1, by1, bx2, by2 = b
                inter_x1 = max(ax1, bx1)
                inter_y1 = max(ay1, by1)
                inter_x2 = min(ax2, bx2)
                inter_y2 = min(ay2, by2)
                if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                    return False
                inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                area_a = (ax2 - ax1) * (ay2 - ay1)
                return inter / (area_a + 1e-9) >= thr

            other_boxes = []
            for b in other_gts:
                if not overlaps(b, target_box, thr=0.7):
                    other_boxes.append(b)

            for module_name, outputs in sorted(module_outputs.items()):
                # record gate-only modules or spatial modules
                if "dwgconv_gate" in outputs:
                    vals = outputs["dwgconv_gate"].squeeze(0).numpy()
                    channel_gate_by_module_scene[module_name][scene].append(vals.reshape(-1))
                if "dwgconvms_gate" in outputs:
                    vals = outputs["dwgconvms_gate"].squeeze(0).numpy().reshape(-1)
                    c = vals.size // 2
                    branch_gate_by_module_scene[f"{module_name}:small"][scene].append(vals[:c])
                    branch_gate_by_module_scene[f"{module_name}:large"][scene].append(vals[c:])

                # handle spatial maps and MDHIFI detailed stats
                if any(k in outputs for k in ("mdhifi_gate", "mdhifi_grad", "mdhifi_texture", "dwgconv_spatial", "hfscc_c4", "hfscc_c5")):
                    # helper: convert tensor to raw spatial map (no per-image min-max)
                    def raw_spatial_map(tensor):
                        arr = tensor.squeeze(0).float().cpu().numpy()
                        if arr.ndim == 3:
                            arr = arr.mean(axis=0)
                        elif arr.ndim == 2:
                            arr = arr
                        else:
                            arr = arr.reshape(1, -1)
                        return arr

                    # compute stats for a given spatial map array
                    def compute_raw_stats(sp_map, target_box_xyxy, other_boxes=None):
                        h, w = sp_map.shape[:2]
                        x1, y1, x2, y2 = target_box_xyxy
                        x1 = int(max(0, np.floor(x1)))
                        y1 = int(max(0, np.floor(y1)))
                        x2 = int(min(w, np.ceil(x2)))
                        y2 = int(min(h, np.ceil(y2)))
                        target = sp_map[y1:y2, x1:x2]
                        # inner expansion by 1 to handle stride misalign
                        ex = 1
                        ix1 = max(0, x1 - ex)
                        iy1 = max(0, y1 - ex)
                        ix2 = min(w, x2 + ex)
                        iy2 = min(h, y2 + ex)

                        # outer ring: expand by factor
                        cx = (x1 + x2) / 2.0
                        cy = (y1 + y2) / 2.0
                        bw = (x2 - x1)
                        bh = (y2 - y1)
                        factor = 3.0
                        ox1 = int(max(0, cx - bw * factor / 2.0))
                        oy1 = int(max(0, cy - bh * factor / 2.0))
                        ox2 = int(min(w, cx + bw * factor / 2.0))
                        oy2 = int(min(h, cy + bh * factor / 2.0))

                        clutter_mask = np.zeros((h, w), dtype=bool)
                        clutter_mask[oy1:oy2, ox1:ox2] = True
                        clutter_mask[iy1:iy2, ix1:ix2] = False

                        # exclude borders
                        border = 4
                        clutter_mask[:border, :] = False
                        clutter_mask[-border:, :] = False
                        clutter_mask[:, :border] = False
                        clutter_mask[:, -border:] = False

                        # exclude other GT boxes
                        if other_boxes:
                            for ob in other_boxes:
                                bx1, by1, bx2, by2 = [int(x) for x in ob]
                                bx1 = max(0, bx1)
                                by1 = max(0, by1)
                                bx2 = min(w, bx2)
                                by2 = min(h, by2)
                                clutter_mask[by1:by2, bx1:bx2] = False

                        clutter = sp_map[clutter_mask]
                        if target.size == 0 or clutter.size == 0:
                            return {
                                "target_mean": float("nan"),
                                "clutter_mean": float("nan"),
                                "tcrr": float("nan"),
                                "target_count": int(target.size),
                                "clutter_count": int(clutter.size),
                                "map_min": float(sp_map.min()),
                                "map_max": float(sp_map.max()),
                            }
                        tmean = float(target.mean())
                        cmean = float(clutter.mean())
                        return {
                            "target_mean": tmean,
                            "clutter_mean": cmean,
                            "tcrr": float(tmean / (cmean + 1e-9)),
                            "target_count": int(target.size),
                            "clutter_count": int(clutter.size),
                            "map_min": float(sp_map.min()),
                            "map_max": float(sp_map.max()),
                        }

                    # process gate if exists
                    if "mdhifi_gate" in outputs:
                        gate_map = raw_spatial_map(outputs["mdhifi_gate"]) 
                        # if values are not already in [0,1], apply sigmoid to obtain gate weights
                        if gate_map.min() < -0.01 or gate_map.max() > 1.01:
                            gate_map = 1.0 / (1.0 + np.exp(-gate_map))

                        # detect constant gates (after possible sigmoid)
                        gate_const = None
                        if np.allclose(gate_map, gate_map.flat[0], atol=1e-6):
                            val = float(gate_map.flat[0])
                            if abs(val - 1.0) < 1e-6:
                                gate_const = 1
                            elif abs(val - 0.0) < 1e-6:
                                gate_const = 0

                        gate_stats = compute_raw_stats(gate_map, target_box_input, other_boxes=other_boxes)

                        out_path = Path(args.outdir) / "spatial_maps" / "mdhifi_gate" / f"{stem}_{module_name.replace('.', '_')}.jpg"
                        # display-normalized map for visualization only
                        disp_map = (gate_map - gate_map.min()) / (gate_map.max() - gate_map.min() + 1e-9)
                        save_spatial_panel(out_path, image_bgr, disp_map, target_box, gate_stats["tcrr"], args.alpha)

                        # enforce unit checks: constant ones -> tcrr == 1, constant zeros -> tcrr -> nan
                        reported_tcrr = gate_stats["tcrr"]
                        if gate_const == 1:
                            reported_tcrr = 1.0
                        elif gate_const == 0:
                            reported_tcrr = float("nan")

                        spatial_rows.append(
                            {
                                "image": str(image_path),
                                "scene": scene,
                                "module": module_name,
                                "kind": "mdhifi_gate",
                                "tcrr": reported_tcrr,
                                "target_mean": gate_stats["target_mean"],
                                "clutter_mean": gate_stats["clutter_mean"],
                                "target_count": gate_stats["target_count"],
                                "clutter_count": gate_stats["clutter_count"],
                                "map_min": gate_stats.get("map_min", float("nan")),
                                "map_max": gate_stats.get("map_max", float("nan")),
                                "gate_constant": gate_const,
                                "box_x1": float(target_box_input[0]),
                                "box_y1": float(target_box_input[1]),
                                "box_x2": float(target_box_input[2]),
                                "box_y2": float(target_box_input[3]),
                                "panel_path": str(out_path),
                            }
                        )

                    # if grad+texture present compute E and Eg stats
                    if "mdhifi_grad" in outputs and "mdhifi_texture" in outputs and "mdhifi_gate" in outputs:
                        grad_map = outputs["mdhifi_grad"].squeeze(0).float().numpy()
                        tex_map = outputs["mdhifi_texture"].squeeze(0).float().numpy()
                        gate_map = outputs["mdhifi_gate"].squeeze(0).float().numpy()
                        # reduce to spatial maps
                        if grad_map.ndim == 3:
                            grad_sp = grad_map.mean(axis=0)
                        else:
                            grad_sp = grad_map
                        if tex_map.ndim == 3:
                            tex_sp = tex_map.mean(axis=0)
                        else:
                            tex_sp = tex_map
                        if gate_map.ndim == 3:
                            gate_sp = gate_map.mean(axis=0)
                        else:
                            gate_sp = gate_map

                        E = np.abs(grad_sp + tex_sp)
                        Eg = np.abs(gate_sp * (grad_sp + tex_sp))

                        stats_E = compute_raw_stats(E, target_box_input, other_boxes=other_boxes)
                        stats_Eg = compute_raw_stats(Eg, target_box_input, other_boxes=other_boxes)

                        # retention and log-TCRR gain
                        def safe_div(a, b):
                            return float(a / (b + 1e-9))

                        target_ret = safe_div(stats_Eg["target_mean"], stats_E["target_mean"]) if not np.isnan(stats_E["target_mean"]) else float("nan")
                        clutter_ret = safe_div(stats_Eg["clutter_mean"], stats_E["clutter_mean"]) if not np.isnan(stats_E["clutter_mean"]) else float("nan")
                        tcrr_gain = float(np.log(stats_Eg["tcrr"] + 1e-9) - np.log(stats_E["tcrr"] + 1e-9)) if not (np.isnan(stats_Eg["tcrr"]) or np.isnan(stats_E["tcrr"])) else float("nan")

                        out_path_e = Path(args.outdir) / "spatial_maps" / "mdhifi_EEg" / f"{stem}_{module_name.replace('.', '_')}.jpg"
                        # visualize Eg over image using existing helper (normalize for display)
                        vis_map = (Eg - Eg.min()) / (Eg.max() - Eg.min() + 1e-9)
                        save_spatial_panel(out_path_e, image_bgr, vis_map, target_box, stats_Eg["tcrr"], args.alpha)

                        spatial_rows.append(
                            {
                                "image": str(image_path),
                                "scene": scene,
                                "module": module_name,
                                "kind": "E_Eg",
                                "tcrr": stats_Eg["tcrr"],
                                "target_mean": stats_E["target_mean"],
                                "clutter_mean": stats_E["clutter_mean"],
                                "target_mean_Eg": stats_Eg["target_mean"],
                                "clutter_mean_Eg": stats_Eg["clutter_mean"],
                                "target_retention": target_ret,
                                "clutter_retention": clutter_ret,
                                "tcrr_gain": tcrr_gain,
                                "map_min": stats_Eg.get("map_min", float("nan")),
                                "map_max": stats_Eg.get("map_max", float("nan")),
                                "box_x1": float(target_box_input[0]),
                                "box_y1": float(target_box_input[1]),
                                "box_x2": float(target_box_input[2]),
                                "box_y2": float(target_box_input[3]),
                                "panel_path": str(out_path_e),
                            }
                        )

        summary_rows: list[dict[str, object]] = []
        for module_name, scene_map in sorted(channel_gate_by_module_scene.items()):
            save_gate_distribution_plot(
                scene_map,
                Path(args.outdir) / "dwgconv_channel_gate" / f"{module_name.replace('.', '_')}_distribution.png",
                f"{module_name} channel gate distribution",
            )
            save_dwgconv_scene_bar(
                scene_map,
                Path(args.outdir) / "dwgconv_channel_gate" / f"{module_name.replace('.', '_')}_mean_bar.png",
                f"{module_name} mean channel gate by scene",
            )
            for scene, arrays in sorted(scene_map.items()):
                values = np.concatenate([a.reshape(-1) for a in arrays], axis=0)
                summary_rows.append(
                    {
                        "module": module_name,
                        "scene": scene,
                        "gate_type": "channel",
                        "mean": float(values.mean()),
                        "std": float(values.std()),
                        "count": int(values.size),
                    }
                )

        for module_name, scene_map in sorted(branch_gate_by_module_scene.items()):
            save_gate_distribution_plot(
                scene_map,
                Path(args.outdir) / "dwgconv_branch_gate" / f"{module_name.replace('.', '_')}_distribution.png",
                f"{module_name} branch gate distribution",
            )
            save_dwgconv_scene_bar(
                scene_map,
                Path(args.outdir) / "dwgconv_branch_gate" / f"{module_name.replace('.', '_')}_mean_bar.png",
                f"{module_name} mean branch gate by scene",
            )
            for scene, arrays in sorted(scene_map.items()):
                values = np.concatenate([a.reshape(-1) for a in arrays], axis=0)
                summary_rows.append(
                    {
                        "module": module_name,
                        "scene": scene,
                        "gate_type": "branch",
                        "mean": float(values.mean()),
                        "std": float(values.std()),
                        "count": int(values.size),
                    }
                )

        write_csv(
            Path(args.outdir) / "gate_summary.csv",
            summary_rows,
            ["module", "scene", "gate_type", "mean", "std", "count"],
        )
        write_csv(
            Path(args.outdir) / "spatial_response_summary.csv",
            spatial_rows,
            [
                "image",
                "scene",
                "module",
                "kind",
                "tcrr",
                "target_mean",
                "clutter_mean",
                "target_count",
                "clutter_count",
                "map_min",
                "map_max",
                "gate_constant",
                "target_mean_Eg",
                "clutter_mean_Eg",
                "target_retention",
                "clutter_retention",
                "tcrr_gain",
                "box_x1",
                "box_y1",
                "box_x2",
                "box_y2",
                "panel_path",
            ],
        )
        print(f"saved analysis to {Path(args.outdir).resolve()}")
    finally:
        collector.close()


if __name__ == "__main__":
    main()
