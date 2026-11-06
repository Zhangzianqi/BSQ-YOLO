# BSQ-YOLO

Official source code for **Progressive Semantic Feedback with Information Preservation for Small Floating Litter Detection**.

BSQ-YOLO is a YOLO11n-based progressive semantic feedback network for water-surface floating-litter detection.

## Model

The repository provides the final improved model and the original YOLO11n baseline:

| Model | Network configuration | Prediction scales |
| --- | --- | --- |
| BSQ-YOLO | [Final model YAML](ultralytics/cfg/models/11/yolo11n-p2-full-bsq-spd-p2p3-rma-down-wfsa-p234-hrd-sgdfuse-p2.yaml) | P2–P5 |
| YOLO11n | [Original baseline YAML](ultralytics/cfg/models/11/yolo11n.yaml) | P3–P5 |

The baseline configuration is copied unchanged from the original project's `yolo11.yaml` and named `yolo11n.yaml` to select the `n` scale. Its default 80 classes are overridden by the dataset configuration during training.

| Component | Implementation |
| --- | --- |
| Detail-preserving mixed-attention backbone (DMAB) | `SPDConv` in [conv.py](ultralytics/nn/modules/conv.py); `RMADown` in [block.py](ultralytics/nn/modules/block.py) |
| Blur-aware semantic query (BSQ) | `BSQGather`, `BSQFeedback` in [bsq.py](ultralytics/nn/modules/bsq.py) |
| Semantically guided detail-preserving fusion (SGDF) | `SGDFuseP2` in [block.py](ultralytics/nn/modules/block.py) |
| Wavelet-guided detail-reparameterization detection head (WDRH) | `WFSA` in [wsa.py](ultralytics/nn/modules/wsa.py); `HRDP2Detect` in [head.py](ultralytics/nn/modules/head.py) |
| Model parsing and module registration | [tasks.py](ultralytics/nn/tasks.py), [modules/__init__.py](ultralytics/nn/modules/__init__.py) |

The customized Ultralytics framework is retained with its original module paths and implementations to preserve internal imports and checkpoint compatibility. Other experimental network configurations and launchers are omitted.

## Repository structure

```text
BSQ-YOLO/
├── ultralytics/          # Customized framework and model implementations
│   └── cfg/models/11/   # Final BSQ-YOLO and original YOLO11n YAMLs
├── configs/data.yaml    # Dataset paths and class names
├── experiments/BSQ-YOLO/ # Released final-model weights and training results
├── train.py             # Training entry point
├── val.py               # Validation/test entry point
├── predict.py           # Inference entry point
├── pyproject.toml       # Package metadata and dependencies
├── requirements.txt
└── LICENSE
```

## Usage

Use Python 3.10 or newer. Install PyTorch and torchvision for your device, then install this customized source package:

```bash
pip install -r requirements.txt
```

Set the absolute dataset path in `configs/data.yaml`. Run commands from the repository root:

```bash
python train.py --model bsq-yolo --data configs/data.yaml --device 0
python train.py --model yolo11n --data configs/data.yaml --device 0
python val.py --weights /path/to/best.pt --data configs/data.yaml --split test --device 0
python predict.py --weights /path/to/best.pt --source /path/to/images --device 0
```

Training uses random initialization (`pretrained=False`). The training script defaults to seed 6; the released checkpoint's original settings are recorded in [args.yaml](experiments/BSQ-YOLO/args.yaml).

## Trained model and results

The final BSQ-YOLO run is available in [experiments/BSQ-YOLO](experiments/BSQ-YOLO). Original filenames and training records are retained.

| Artifact | File |
| --- | --- |
| Best checkpoint | [best.pt](experiments/BSQ-YOLO/weights/best.pt) |
| Last checkpoint | [last.pt](experiments/BSQ-YOLO/weights/last.pt) |
| Training metrics | [results.csv](experiments/BSQ-YOLO/results.csv) |
| Training curves | [results.png](experiments/BSQ-YOLO/results.png) |
| Recorded training settings | [args.yaml](experiments/BSQ-YOLO/args.yaml) |

Precision–recall curves, confusion matrices, and training/validation examples are included in the same folder.

## Dataset

The **Small Floating Litter Dataset (SFLD)** contains 2,939 images and 9,844 annotated instances across six categories; small objects account for over 74% of the instances.

SFLD is available separately through Baidu Netdisk:

- Download: [SFLD.zip (Baidu Netdisk)](https://pan.baidu.com/s/1Xx1QOHH-3BuOJUS4jZDGzg?pwd=5mfb)
- Extraction code: `5mfb`

The complete dataset is distributed through the link above. The repository contains the final-model weights and training results linked in the preceding section.

Class IDs: `0 bottle`, `1 can`, `2 cup`, `3 box`, `4 plastic-bag`, `5 other-waste`.

## Release status

- [x] BSQ-YOLO source code and final network configuration
- [x] Original YOLO11n baseline configuration
- [x] SFLD dataset download link
- [x] Final-model trained weights and training results
- [ ] TensorRT deployment scripts for Jetson Orin NX

## Acknowledgments and license

This code is based on [Ultralytics](https://github.com/ultralytics/ultralytics); the local framework version is `8.4.34`. The original AGPL-3.0 [LICENSE](LICENSE), source notices, and bundled third-party notices are retained. The additional CFPT notice is preserved in [cfpt_LICENSE.txt](ultralytics/nn/modules/cfpt_LICENSE.txt).
