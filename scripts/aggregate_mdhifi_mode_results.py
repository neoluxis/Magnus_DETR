#!/usr/bin/env python3
"""Aggregate spatial_response_summary.csv from multiple modes and plot TCRR comparisons."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import sys


def main(outdir: Path):
    modes = [p.name for p in outdir.iterdir() if p.is_dir()]
    rows = []
    for m in modes:
        csvp = outdir / m / "spatial_response_summary.csv"
        if not csvp.exists():
            continue
        df = pd.read_csv(csvp)
        df["mode"] = m
        rows.append(df)
    if not rows:
        print("no results found")
        return 1
    all_df = pd.concat(rows, ignore_index=True)
    all_df.to_csv(outdir / "spatial_response_summary_combined.csv", index=False)

    # plot mean TCRR by mode and scene
    agg = all_df.groupby(["mode", "scene"]) ["tcrr"].mean().unstack(fill_value=0)
    ax = agg.plot(kind="bar", figsize=(10,5))
    ax.set_ylabel("Mean TCRR")
    ax.set_title("MDHIFI noise_gate mode comparison: Mean TCRR by scene")
    plt.tight_layout()
    plt.savefig(outdir / "tcrr_mode_comparison.png", dpi=180)
    # per-module comparisons
    modules_specs = [
        ("model.17", lambda df: (df.module == "model.17") & (df.kind == "mdhifi_gate")),
        ("model.19", lambda df: (df.module == "model.19") & (df.kind == "mdhifi_gate")),
        ("hfscc_c4", lambda df: df.kind == "hfscc_c4"),
        ("hfscc_c5", lambda df: df.kind == "hfscc_c5"),
    ]
    for name, selector in modules_specs:
        sel_df = all_df[selector(all_df)].copy()
        if sel_df.empty:
            continue
        mean_by_mode = sel_df.groupby("mode")["tcrr"].mean()
        fig, ax = plt.subplots(figsize=(6,4))
        mean_by_mode.plot(kind="bar", ax=ax, color=["#2ca02c", "#1f77b4", "#d62728"]) if not mean_by_mode.empty else None
        ax.set_ylabel("Mean TCRR")
        ax.set_title(f"{name} TCRR by mode")
        plt.tight_layout()
        fig.savefig(outdir / f"tcrr_{name}_by_mode.png", dpi=180)
        plt.close(fig)
    print("wrote:", outdir / "spatial_response_summary_combined.csv", outdir / "tcrr_mode_comparison.png")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: aggregate_mdhifi_mode_results.py <outdir>")
        raise SystemExit(1)
    raise SystemExit(main(Path(sys.argv[1])))
