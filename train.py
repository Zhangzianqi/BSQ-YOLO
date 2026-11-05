"""Train BSQ-YOLO or the original YOLO11n baseline."""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "ultralytics/cfg/models/11"
MODELS = {
    "bsq-yolo": MODEL_DIR / "yolo11n-p2-full-bsq-spd-p2p3-rma-down-wfsa-p234-hrd-sgdfuse-p2.yaml",
    "yolo11n": MODEL_DIR / "yolo11n.yaml",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODELS), default="bsq-yolo")
    parser.add_argument("--data", default=str(ROOT / "configs/data.yaml"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=6)
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(str(MODELS[args.model]), task="detect")
    model.train(
        data=args.data, device=args.device, epochs=args.epochs, batch=args.batch,
        imgsz=args.imgsz, workers=args.workers, seed=args.seed,
        optimizer="auto", patience=50, pretrained=False, resume=False,
        project=str(ROOT / "runs/train"), name=args.model, exist_ok=False,
    )


if __name__ == "__main__":
    main()
