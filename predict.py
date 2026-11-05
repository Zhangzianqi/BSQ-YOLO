"""Run detection with a trained BSQ-YOLO checkpoint."""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True, help="Image, video, directory, or stream")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights, task="detect")
    for _ in model.predict(
        source=args.source, device=args.device, imgsz=args.imgsz, conf=args.conf,
        save=True, stream=True, project=str(ROOT / "runs/predict"),
        name="bsq-yolo", exist_ok=False,
    ):
        pass


if __name__ == "__main__":
    main()
