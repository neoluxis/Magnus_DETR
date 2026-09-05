import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultralytics-git"))

from ultralytics.utils import YAML


PLACEMENTS = ("receiver", "joint", "correction", "ungated")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the 0905 HFSCC placement ablation.")
    parser.add_argument("--base-yaml", type=Path, default=Path("exp_cfg/0905/rtdetr_swinv2_tiny_b5.yaml"))
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="datasets/CST_AntiUAV/cst-sample_train-5000_val-1000_test-1000_seq-100_id-0/data.yaml",
    )
    parser.add_argument("--project", type=Path, default=Path("runs/detect/0905"))
    parser.add_argument("--generated-config-dir", type=Path, default=Path("runs/detect/0905/_generated_cfg"))
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument(
        "--placement",
        choices=PLACEMENTS,
        default=None,
        help="Run only one placement. Omit to run all placements serially.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands and generated configs without training.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow Ultralytics to reuse existing run directories.")
    args = parser.parse_args()
    args.base_yaml = args.base_yaml if args.base_yaml.is_absolute() else ROOT / args.base_yaml
    args.project = args.project if args.project.is_absolute() else ROOT / args.project
    args.generated_config_dir = (
        args.generated_config_dir if args.generated_config_dir.is_absolute() else ROOT / args.generated_config_dir
    )
    return args


def write_variant_config(base_yaml, output_dir, placement):
    cfg = YAML.load(base_yaml)
    hfscc_rows = [row for row in cfg["head"] if row[2] == "HFSCCPlacement"]
    if len(hfscc_rows) != 1:
        raise ValueError(f"Expected exactly one HFSCCPlacement row in {base_yaml}, found {len(hfscc_rows)}.")

    hfscc_rows[0][3][5] = placement
    output_path = output_dir / f"rtdetr_swinv2_tiny_b5_{placement}.yaml"
    YAML.save(output_path, cfg, header="# Generated from exp_cfg/0905/rtdetr_swinv2_tiny_b5.yaml.\n")
    return output_path


def build_command(args, placement, config_path):
    command = [
        sys.executable,
        str(ROOT / "scripts/train_detr.py"),
        "--model",
        str(config_path),
        "--dataset_path",
        args.dataset_path,
        "--project",
        str(args.project),
        "--name",
        f"{placement}_s42",
        "--epochs",
        "100",
        "--patience",
        "10",
        "--batch_size",
        "1",
        "--imgsz",
        "640",
        "--device",
        args.device,
        "--workers",
        "0",
        "--optim",
        "AdamW",
        "--lr0",
        "0.001",
        "--lrf",
        "0.01",
        "--weight_decay",
        "0.0001",
        "--warmup_epochs",
        "3",
        "--seed",
        "42",
        "--amp",
        "--early-stop-metric",
        "val_loss",
    ]
    if args.exist_ok:
        command.append("--exist-ok")
    return command


def main():
    args = parse_args()
    args.generated_config_dir.mkdir(parents=True, exist_ok=True)

    commands = []
    placements = (args.placement,) if args.placement else PLACEMENTS
    for placement in placements:
        config_path = write_variant_config(args.base_yaml, args.generated_config_dir, placement)
        command = build_command(args, placement, config_path)
        commands.append((placement, config_path, command))

    for placement, config_path, command in commands:
        save_dir = args.project / f"{placement}_s42"
        print(f"[{placement}] config={config_path} save_dir={save_dir}")
        print(" ".join(command))

    if args.dry_run:
        return 0

    for placement, _, command in commands:
        print(f"Starting {placement}_s42", flush=True)
        subprocess.run(command, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
