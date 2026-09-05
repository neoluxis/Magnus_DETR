#!/usr/bin/env python3
"""Create horizontal stitch panels comparing modes and write a short TCRR report.

Usage:
  .venv/bin/python scripts/create_mode_stitch_and_report.py runs/module_response_analysis/b5-3 --n 6
"""
from pathlib import Path
import argparse
import pandas as pd
import cv2
import numpy as np
import textwrap


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("base", help="Base analysis dir (contains mode subdirs)")
    p.add_argument("--modes", nargs="+", default=["orig", "ones", "zeros"])
    p.add_argument("--n", type=int, default=6, help="Number of example images to stitch")
    p.add_argument("--out", default=None)
    return p.parse_args()


def find_panel_for_stem(base: Path, mode: str, stem: str):
    search_dir = base / mode / "spatial_maps"
    if not search_dir.exists():
        return None
    matches = list(search_dir.rglob(f"{stem}_*.jpg"))
    if not matches:
        return None
    return matches[0]


def stitch_images(img_paths):
    imgs = []
    heights = []
    for p in img_paths:
        if p is None or not p.exists():
            imgs.append(None)
            heights.append(0)
            continue
        im = cv2.imread(str(p))
        if im is None:
            imgs.append(None)
            heights.append(0)
            continue
        imgs.append(im)
        heights.append(im.shape[0])
    if all(i is None for i in imgs):
        return None
    target_h = max(heights)
    resized = []
    for im in imgs:
        if im is None:
            # placeholder white image
            resized.append(255 * np.ones((target_h, target_h // 2, 3), dtype=np.uint8))
        else:
            h, w = im.shape[:2]
            new_w = int(w * target_h / h)
            resized.append(cv2.resize(im, (new_w, target_h), interpolation=cv2.INTER_CUBIC))
    spacer = 8
    total_w = sum(i.shape[1] for i in resized) + spacer * (len(resized) - 1)
    canvas = 255 * np.ones((target_h, total_w, 3), dtype=np.uint8)
    x = 0
    for im in resized:
        canvas[:, x : x + im.shape[1]] = im
        x += im.shape[1] + spacer
    return canvas


def main():
    args = parse_args()
    base = Path(args.base)
    out = Path(args.out) if args.out else base / "mode_comparisons"
    out.mkdir(parents=True, exist_ok=True)

    combined_csv = base / "spatial_response_summary_combined.csv"
    if not combined_csv.exists():
        # try to combine per-mode CSVs
        rows = []
        for m in args.modes:
            csvp = base / m / "spatial_response_summary.csv"
            if csvp.exists():
                df = pd.read_csv(csvp)
                df["mode"] = m
                rows.append(df)
        if not rows:
            print("No CSVs found to build examples")
            return 1
        all_df = pd.concat(rows, ignore_index=True)
    else:
        all_df = pd.read_csv(combined_csv)

    # select example stems (unique image groups)
    all_df["stem"] = all_df["image"].apply(lambda p: Path(p))
    all_df["stem"] = all_df["stem"].apply(lambda p: f"{p.parent.name}__{p.stem}")
    unique_stems = all_df["stem"].unique().tolist()
    selected = unique_stems[: args.n]

    for stem in selected:
        img_paths = []
        for m in args.modes:
            p = find_panel_for_stem(base, m, stem)
            img_paths.append(p)
        canvas = stitch_images(img_paths)
        if canvas is None:
            continue
        out_path = out / f"{stem}_modes.jpg"
        cv2.imwrite(str(out_path), canvas)

    # produce short report paragraph
    summary = all_df.groupby(["mode"])["tcrr"].mean()
    scene_agg = all_df.groupby(["mode", "scene"])["tcrr"].mean().unstack(fill_value=float("nan"))

    lines = []
    lines.append("MDHIFI noise_gate mode comparison — short report:\n")
    for mode, val in summary.items():
        lines.append(f"- Mode '{mode}': mean TCRR = {val:.3f}")
    best_mode = summary.idxmax()
    lines.append(f"\nOverall, mode '{best_mode}' achieves the highest mean TCRR.")
    lines.append("\nMean TCRR by scene:")
    lines.append(scene_agg.to_string())

    report = "\n".join(lines)
    (out / "report.txt").write_text(report, encoding="utf-8")
    print("wrote comparisons to", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
