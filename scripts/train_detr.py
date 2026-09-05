import argparse
from pathlib import Path

from ultralytics import RTDETR
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import LOGGER


def parse_pretrained_arg(value: str | None):
    """Normalize CLI pretrained values into bool/path/None for Ultralytics."""
    if value is None:
        return None
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    return value

def parse_resume_arg(value: str | None):
    """Normalize CLI resume values into bool/path/None for Ultralytics."""
    if value is None:
        return None
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    return value

def resolve_resume_strategy(model_path: str, resume: bool | str | None, target_epochs: int):
    """Convert finished-run resume requests into checkpoint fine-tuning."""
    if not isinstance(resume, str):
        return model_path, resume

    resume_path = Path(resume)
    if not resume_path.exists():
        return model_path, resume

    _, ckpt = load_checkpoint(resume_path, device="cpu", fuse=False)
    start_epoch = ckpt.get("epoch", -1) + 1
    completed_epochs = int(ckpt.get("train_args", {}).get("epochs", start_epoch) or start_epoch)
    if start_epoch >= completed_epochs and target_epochs > start_epoch:
        LOGGER.info(
            "Checkpoint %s already completed %s epochs; starting a new training run from its weights to %s epochs.",
            resume_path,
            start_epoch,
            target_epochs,
        )
        return str(resume_path), None

    return model_path, resume

def parse_args():
    parser = argparse.ArgumentParser(description="Train RT-DETR model")
    parser.add_argument("--project", type=str, default="cst-sample-5k1k1k-s100", help="Project name for saving results")
    parser.add_argument("--name", type=str, default="rtdetr-pretrained", help="Experiment name for saving results")
    parser.add_argument("--model", type=str, default="rtdetr-l.yaml", help="Pretrained model to use")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=100, help="Early stopping patience in epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training")
    parser.add_argument("--device", type=str, default=None, help="Training device, e.g. 0 or cpu")
    parser.add_argument("--dataset_path", type=str, default="datasets/CST_AntiUAV/cst-sample_train-5000_val-1000_test-1000_seq-100_id-0/data.yaml", help="Path to dataset YAML file")
    parser.add_argument("--optim", type=str, default="AdamW", help="Optimizer to use")
    parser.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final learning rate factor")
    parser.add_argument("--momentum", type=float, default=0.937, help="Momentum for optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay for optimizer")
    parser.add_argument("--warmup_bias_lr", type=float, default=0.0, help="Warmup learning rate for bias parameters")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False, help="Enable automatic mixed precision")
    parser.add_argument("--fraction", type=float, default=1.0, help="Dataset fraction to use for training")
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True, help="Cache images for faster training")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers for data loading")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--deterministic", action="store_true", default=False, help="Enable deterministic training for reproducibility")
    parser.add_argument("--iou_type", type=str, default="giou",
                        choices=["ciou", "diou", "giou", "siou", "inner_ciou", "inner_siou"],
                        help="IoU loss type for bounding box regression")
    parser.add_argument("--inner_ratio", type=float, default=0.75,
                        help="Inner box scale factor for inner_iou variants (0.0-1.0)")
    parser.add_argument("--freeze", type=int, default=0, help="Freeze the first N model layers")
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Optional checkpoint path or true/false. Leave unset to use model/YAML defaults.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional checkpoint path or true/false for resuming training.",
    )
    parser.add_argument("--exist-ok", action="store_true", default=False, help="Allow overwriting an existing run directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    resume = parse_resume_arg(args.resume)
    model_path, resume = resolve_resume_strategy(str(Path(args.model)), resume, args.epochs)
    yolo = RTDETR(model_path)
    train_kwargs = dict(
        data=args.dataset_path,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch_size,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        optimizer=args.optim,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_bias_lr=args.warmup_bias_lr,
        amp=args.amp,
        fraction=args.fraction,
        cache=args.cache,
        workers=args.workers,
        seed=args.seed,
        deterministic=args.deterministic,
        iou_type=args.iou_type,
        inner_ratio=args.inner_ratio,
        freeze=args.freeze,
        exist_ok=args.exist_ok,
        mosaic=False,  ## 加速训练 4.5min/epoch -> 1min/epoch
        mixup=False,   # Disable mixup augmentation for better small object detection
    )
    pretrained = parse_pretrained_arg(args.pretrained)
    if pretrained is not None:
        train_kwargs["pretrained"] = pretrained
    if resume is not None:
        train_kwargs["resume"] = resume
    yolo.train(**train_kwargs)

    # os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
