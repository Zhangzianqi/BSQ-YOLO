"""Evaluate a BSQ-YOLO checkpoint on the validation or test split."""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default=str(ROOT / "configs/data.yaml"))
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    from ultralytics import YOLO

    metrics = YOLO(args.weights, task="detect").val(
        data=args.data, split=args.split, device=args.device, imgsz=args.imgsz,
        batch=args.batch, workers=args.workers, plots=True,
        project=str(ROOT / "runs/val"), name=f"bsq-yolo-{args.split}", exist_ok=False,
    )
    print(f"Precision: {metrics.box.mp:.6f} | Recall: {metrics.box.mr:.6f}")
    print(f"mAP50: {metrics.box.map50:.6f} | mAP50-95: {metrics.box.map:.6f}")


if __name__ == "__main__":
    main()
