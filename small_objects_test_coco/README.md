# COCO size-wise evaluation

This program evaluates the small- and medium-object bounding-box AP reported in the revised Table 5 using the official `pycocotools` evaluator. The original size criterion is retained separately in [`small_objects_test_legacy`](../small_objects_test_legacy/).

## Run

Use the repository's environment and install the additional evaluation dependencies:

```bash
pip install -r small_objects_test_coco/requirements-evaluation.txt
```

Set the absolute SFLD dataset root in `configs/data.yaml`. The dataset must contain `test/images` and `test/labels`, with `test: test/images` in the YAML. Run from the repository root:

```bash
python small_objects_test_coco/test_coco_sm_multiiou.py --data configs/data.yaml --weights experiments/BSQ-YOLO/weights/best.pt --device 0
```

To compare two checkpoints with the same settings, provide both paths, using distinct filename stems:

```bash
python small_objects_test_coco/test_coco_sm_multiiou.py --data configs/data.yaml --weights /path/to/ours.pt /path/to/yolo11n.pt --device 0
```

`--check-only` checks the data, checkpoint paths and evaluation dependencies without loading a model. Other options are `--imgsz`, `--batch`, `--workers` and `--output`. Defaults are imgsz=640, batch=16, workers=0 and FP32. Both models use rect=True, confidence=0.001, class-wise multi-label NMS at IoU=0.7, and at most 300 predictions per image. NMS has no time cutoff. Each model is inferred once, and its predictions are reused for all metrics.

## Area groups and metrics

Areas are annotated bounding-box width times height in original-image pixels. Negligible boundary overflow from rounded YOLO labels is clipped before conversion. The official inclusive ranges are Small `[0, 32^2]` and Medium `[32^2, 96^2]` square pixels.

The program reports `AP50_S`, `AP50_M`, `AP75_S`, `AP75_M`, `AP50_95_S` and `AP50_95_M`. The last pair averages IoU thresholds 0.50, 0.55, ..., 0.95; `AP50_95_S` is the standard COCO AP_S. AP uses the official 101 recall thresholds and `maxDets=100` per image/category. The inference cap of 300 is separate from this evaluation cap. Categories without ground truth in a size group are excluded from that group's AP average.

All test images and all annotations, including other object sizes, are retained. Official COCO matching gives in-range ground truths priority. Detections matched to out-of-range ground truths are ignored for that size-specific AP, contributing neither TP nor FP. An unmatched detection is ignored if its own box area is outside the evaluated range; otherwise it contributes FP. Since all annotations have `iscrowd=0`, each ground truth can be matched once per IoU threshold. Images with no in-range ground truth remain in the evaluation.

Only Small and Medium are evaluated. SFLD's test split contains 13 large objects among 974 annotations (1.33%), providing limited coverage for large-object performance. AP_L is therefore not reported in Table 5.

## Outputs and checks

Each run writes `coco_sm_results/<timestamp>/` with the summary CSV, a Markdown report, a LaTeX table, COCO annotations, dataset counts, checkpoint/source hashes and inference settings. Each model directory contains predictions, six AP values, per-class results, evaluation parameters and the official precision tensor. CSV/JSON metrics use values in `[0, 1]`; the report and table display percentages.

```bash
python -m unittest discover -s small_objects_test_coco -p test_coco_evaluation.py -v
```
