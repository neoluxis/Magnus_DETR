#!/usr/bin/env python3
"""Run MDHIFI noise_gate analysis in three modes: orig / ones / zeros.

This script reuses helpers from `scripts/analyze_module_responses.py` and
replaces `MDHIFI.noise_gate` with a constant gate when requested, then
runs the same spatial map + TCRR export pipeline.

Example:
  PYTHONPATH=ultralytics-git python scripts/run_mdhifi_gate_analysis.py \
    --model runs/detect/0705/b5-3 --source ../datasets/CST_AntiUAV/CST-AntiUAV/val \
    --outdir runs/module_response_analysis/b5-3_compare --mode ones --max-images 40
"""

from __future__ import annotations

import argparse
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

import scripts.analyze_module_responses as amr


class ConstantGate(nn.Module):
    def __init__(self, out_channels: int, value: float = 1.0):
        super().__init__()
        self.out_channels = int(out_channels)
        self.value = float(value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        h = x.shape[2]
        w = x.shape[3]
        return torch.full((b, self.out_channels, h, w), fill_value=self.value, dtype=x.dtype, device=x.device)


def patch_model_for_mode(core_model, mode: str):
    if mode == "orig":
        return
    for name, module in core_model.named_modules():
        if module.__class__.__name__ == "MDHIFI" and hasattr(module, "noise_gate"):
            c = getattr(module, "edge_proj").in_channels if hasattr(module, "edge_proj") else None
            if c is None:
                continue
            val = 1.0 if mode == "ones" else 0.0
            module.noise_gate = ConstantGate(c, value=val)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--outdir", default="runs/module_response_analysis/compare")
    p.add_argument("--mode", choices=("orig", "ones", "zeros"), default="orig")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None)
    p.add_argument("--max-images", type=int, default=40)
    p.add_argument("--box-source", default="auto", choices=("auto", "gt", "pred"))
    p.add_argument("--pred-conf", type=float, default=0.05)
    p.add_argument("--alpha", type=float, default=0.42)
    return p.parse_args()


def run_analysis(args):
    arch = amr.infer_arch(args.model, "auto")
    wrapper = amr.load_wrapper(args.model, arch, args.device)
    core_model = wrapper.model

    # patch model if requested
    patch_model_for_mode(core_model, args.mode)

    images = amr.collect_images(args.source, args.max_images)
    collector = amr.collect_module_hooks(core_model)

    spatial_rows = []

    try:
        for image_path in images:
            image_bgr = amr.cv2.imread(str(image_path))
            if image_bgr is None:
                continue
            scene = amr.classify_scene(image_path, ["cloud", "building", "vegetation", "sky"])
            target_box = amr.choose_target_box(wrapper, image_bgr, image_path, args.box_source, args.pred_conf)
            if target_box is None:
                continue

            stride = getattr(core_model, "stride", 32)
            input_channels = amr.infer_model_channels(core_model)
            tensor = amr.preprocess_image(image_bgr, arch, args.imgsz, stride, input_channels=input_channels)
            if args.device is not None:
                tensor = tensor.to(args.device)

            collector.clear()
            with torch.no_grad():
                _ = core_model(tensor)

            input_hw = tuple(int(x) for x in tensor.shape[-2:])
            target_box_input = amr.map_box_from_orig_to_input(target_box, image_bgr.shape[:2], input_hw, arch)

            stem = f"{image_path.parent.name}__{image_path.stem}"
            # group records by module for joint processing (gate, grad, texture)
            from collections import defaultdict

            module_outputs: dict[str, dict[str, object]] = defaultdict(dict)
            for record in collector.records:
                module_outputs[record.module_name][record.kind] = record.tensor.detach().cpu()

            for module_name, outputs in sorted(module_outputs.items()):
                # handle spatial modules and MDHIFI detailed stats
                if any(k in outputs for k in ("mdhifi_gate", "mdhifi_grad", "mdhifi_texture", "dwgconv_spatial", "hfscc_c4", "hfscc_c5")):
                    def raw_spatial_map(tensor):
                        arr = tensor.squeeze(0).float().numpy()
                        if arr.ndim == 3:
                            arr = arr.mean(axis=0)
                        elif arr.ndim == 2:
                            arr = arr
                        else:
                            arr = arr.reshape(1, -1)
                        return arr

                    def compute_raw_stats(sp_map, target_box_xyxy):
                        h, w = sp_map.shape[:2]
                        x1, y1, x2, y2 = target_box_xyxy
                        x1 = int(max(0, np.floor(x1)))
                        y1 = int(max(0, np.floor(y1)))
                        x2 = int(min(w, np.ceil(x2)))
                        y2 = int(min(h, np.ceil(y2)))
                        target = sp_map[y1:y2, x1:x2]

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
                        clutter_mask[y1:y2, x1:x2] = False

                        border = 4
                        clutter_mask[:border, :] = False
                        clutter_mask[-border:, :] = False
                        clutter_mask[:, :border] = False
                        clutter_mask[:, -border:] = False

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
                        if gate_map.min() < -0.01 or gate_map.max() > 1.01:
                            gate_map = 1.0 / (1.0 + np.exp(-gate_map))

                        gate_stats = compute_raw_stats(gate_map, target_box_input)
                        gate_const = None
                        if np.allclose(gate_map, gate_map.flat[0], atol=1e-6):
                            val = float(gate_map.flat[0])
                            if abs(val - 1.0) < 1e-6:
                                gate_const = 1
                            elif abs(val - 0.0) < 1e-6:
                                gate_const = 0

                        out_path = Path(args.outdir) / args.mode / "spatial_maps" / "mdhifi_gate" / f"{stem}_{module_name.replace('.', '_')}.jpg"
                        disp_map = (gate_map - gate_map.min()) / (gate_map.max() - gate_map.min() + 1e-9)
                        amr.save_spatial_panel(out_path, image_bgr, disp_map, target_box, gate_stats["tcrr"], args.alpha)

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
                                "box_x1": float(target_box[0]),
                                "box_y1": float(target_box[1]),
                                "box_x2": float(target_box[2]),
                                "box_y2": float(target_box[3]),
                                "panel_path": str(out_path),
                            }
                        )

                    # if grad+texture present compute E and Eg stats
                    if "mdhifi_grad" in outputs and "mdhifi_texture" in outputs and "mdhifi_gate" in outputs:
                        grad_map = outputs["mdhifi_grad"].squeeze(0).float().numpy()
                        tex_map = outputs["mdhifi_texture"].squeeze(0).float().numpy()
                        gate_map = outputs["mdhifi_gate"].squeeze(0).float().numpy()
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

                        stats_E = compute_raw_stats(E, target_box_input)
                        stats_Eg = compute_raw_stats(Eg, target_box_input)

                        def safe_div(a, b):
                            return float(a / (b + 1e-9))

                        target_ret = safe_div(stats_Eg["target_mean"], stats_E["target_mean"]) if not np.isnan(stats_E["target_mean"]) else float("nan")
                        clutter_ret = safe_div(stats_Eg["clutter_mean"], stats_E["clutter_mean"]) if not np.isnan(stats_E["clutter_mean"]) else float("nan")
                        tcrr_gain = float(np.log(stats_Eg["tcrr"] + 1e-9) - np.log(stats_E["tcrr"] + 1e-9)) if not (np.isnan(stats_Eg["tcrr"]) or np.isnan(stats_E["tcrr"])) else float("nan")

                        out_path_e = Path(args.outdir) / args.mode / "spatial_maps" / "mdhifi_EEg" / f"{stem}_{module_name.replace('.', '_')}.jpg"
                        vis_map = (Eg - Eg.min()) / (Eg.max() - Eg.min() + 1e-9)
                        amr.save_spatial_panel(out_path_e, image_bgr, vis_map, target_box, stats_Eg["tcrr"], args.alpha)

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
                                "box_x1": float(target_box[0]),
                                "box_y1": float(target_box[1]),
                                "box_x2": float(target_box[2]),
                                "box_y2": float(target_box[3]),
                                "panel_path": str(out_path_e),
                            }
                        )

        amr.write_csv(Path(args.outdir) / args.mode / "spatial_response_summary.csv", spatial_rows, [
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
        ])
        print(f"saved analysis to {Path(args.outdir).resolve()}/{args.mode}")
    finally:
        collector.close()


def main():
    args = parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
