# Small-object evaluation

Evaluation code for the original manuscript's small-object definition.

For a bounding box with width `w` and height `h` in an original image of size `W x H`:

```text
small = ((w < 32) and (h < 32)) or (w * h / (W * H) < 0.001)
non-small = not small
```

The evaluator reports All, Small and Non-small groups using this criterion. These small-object metrics are not standard COCO AP_S.

## Run

Use the repository's existing environment and set the dataset path in `configs/data.yaml`. The evaluator reads its `test` split and class names.

From the repository root:

```bash
python small_objects_test_legacy/evaluate_small_objects.py --data configs/data.yaml --weights experiments/BSQ-YOLO/weights/best.pt --device 0 --save-cache
```

Multiple checkpoint paths can follow `--weights` to evaluate them with the same settings. Run `--help` for the available options.

Defaults are FP32, imgsz=640, batch=16, workers=6, rect=True, confidence >0.001, class-wise multi-label NMS at IoU=0.7, and max_det=300. Rectangular validation batches use a shared stride of 32. Recall is reported at confidence >=0.25 and IoU=0.5.

## Matching and AP

All test images and annotations are retained. Within each image and class, predictions are matched in descending confidence order, independently at each IoU threshold.

- Ground truths in the evaluated size group have priority.
- A detection matched to an out-of-group ground truth is ignored, contributing neither TP nor FP. Ground-truth matching is one-to-one, including ignored ground truths.
- An unmatched detection is counted as FP if its own original-image size belongs to the evaluated group; otherwise it is ignored.
- Unmatched in-group ground truths contribute FN. Images without in-group ground truths still contribute eligible false positives.

AP uses the precision envelope sampled at 101 recall thresholds. mAP50 is evaluated at IoU=0.50; mAP50-95 averages ten thresholds from 0.50 to 0.95 in steps of 0.05. Classes without ground truths in the evaluated group are excluded from its class average.

`native_comparison.csv` separately reports the repository's native all-object AP from the same predictions.

## Outputs

Each run creates `small_objects_test_legacy/evaluation_runs/run_.../` containing:

- `paper_wide.csv`: grouped model comparison.
- `metrics_long.csv` and `per_class.csv`: counts and detailed metrics.
- `native_comparison.csv`: native all-object comparison.
- `results.json`, `dataset_audit.json` and `test_images.txt`: protocol, dataset and source records.
- Original-image prediction caches when `--save-cache` is enabled.

## Checks

```bash
python -m unittest discover -s small_objects_test_legacy -p test_scale_evaluation.py -v
python small_objects_test_legacy/check_native_adapter.py
```

The first command checks matching, size boundaries and dataset/cache handling. The second checks native AP integration and coordinate restoration using synthetic inputs.
