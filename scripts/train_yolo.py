from ultralytics import YOLO
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO model")
    parser.add_argument("--project", type=str, default="cst-sample-5k1k1k-s100", help="Project name for saving results")
    parser.add_argument("--name", type=str, default="yolov5s-pretrained", help="Experiment name for saving results")
    parser.add_argument("--model", type=str, default="yolov5s.pt", help="Pretrained model or config YAML to use")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience in epochs")
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
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True, help="Cache images for faster training")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers for data loading")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--deterministic", action="store_true", default=False, help="Enable deterministic training for reproducibility")
    parser.add_argument("--pretrained", action="store_true", default=True, help="Use pretrained weights")
    parser.add_argument("--iou_type", type=str, default="ciou",
                        choices=["ciou", "diou", "giou", "siou", "inner_ciou", "inner_siou"],
                        help="IoU loss type for bounding box regression")
    parser.add_argument("--inner_ratio", type=float, default=0.75,
                        help="Inner box scale factor for inner_iou variants (0.0-1.0)")
    parser.add_argument("--close_mosaic", type=int, default=10, help="Close mosaic augmentation N epochs before end")
    parser.add_argument("--exist-ok", action="store_true", default=False, help="Allow overwriting an existing run directory")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    yolo = YOLO(args.model)
    yolo.train(
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
        cache=args.cache,
        workers=args.workers,
        seed=args.seed,
        deterministic=args.deterministic,
        pretrained=args.pretrained,
        iou_type=args.iou_type,
        inner_ratio=args.inner_ratio,
        close_mosaic=args.close_mosaic,
        exist_ok=args.exist_ok,
        mosaic=False,  ## 加速训练 4.5min/epoch -> 1min/epoch
        mixup=False,   # Disable mixup augmentation for better small object detection
    )
