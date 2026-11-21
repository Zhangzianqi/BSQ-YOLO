#!/usr/bin/env python3
"""Optional synthetic checks against the project's actual AP and preprocessing code.

Does not load weights or evaluate real images. Requires the project's existing environment.
"""
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    output = here / "local_checks" / ("native_adapter_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    (output / "runtime/Ultralytics").mkdir(parents=True)
    os.environ["YOLO_CONFIG_DIR"] = str(output / "runtime")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["MPLCONFIGDIR"] = str(output / "matplotlib")
    sys.path.insert(0, str(here.parent))
    sys.dont_write_bytecode = True
    import numpy as np
    import torch
    from types import SimpleNamespace
    from PIL import Image
    from ultralytics.engine.model import Model
    from ultralytics.models.yolo.detect.val import DetectionValidator
    from ultralytics.utils.metrics import ap_per_class
    from project_inference import ReadOnlyDataset, native_ap_from_stats
    from scale_metrics import restore_boxes
    from evaluation_data import write_json
    from evaluate_small_objects import parser

    captured = {}

    class CaptureValidator:
        def __init__(self, args, _callbacks):
            captured.update(args)
            self.metrics = None

        def __call__(self, model):
            pass

    # Exercise the real Model.val method, rather than assuming cfg/default.yaml
    # or checkpoint train_args are its final settings. No model is run here.
    dummy = SimpleNamespace(overrides={"rect": False}, callbacks={}, model=None)
    Model.val(dummy, validator=CaptureValidator)
    assert captured["rect"] is True
    assert parser().parse_args([]).rect == captured["rect"]

    native = DetectionValidator(save_dir=output / "native", args={"imgsz": 640, "plots": False})
    native.device, native.training = torch.device("cpu"), False
    rng = np.random.default_rng(513)
    # Native matching followed by AP-only glue must exactly equal ap_per_class's
    # AP outputs. Its threshold search is used here only as a SYNTHETIC oracle.
    native.metrics.names = {0: "a", 1: "b", 2: "c"}
    for _ in range(12):
        gt = rng.uniform(0, 300, (5, 4)).astype(np.float32)
        gt[:, 2:] = gt[:, :2] + rng.uniform(10, 100, (5, 2))
        pred = np.vstack([gt, gt[:2] + 3, [[0, 0, 8, 8]]])
        gt_cls = np.array([0, 0, 1, 1, 2], dtype=np.float32)
        p_cls = np.array([0, 0, 1, 1, 2, 0, 0, 2], dtype=np.float32)
        scores = rng.uniform(.05, .99, len(pred))
        matched = native._process_batch({"cls": torch.from_numpy(p_cls), "bboxes": torch.from_numpy(pred)},
                                        {"cls": torch.from_numpy(gt_cls), "bboxes": torch.from_numpy(gt)})
        native.metrics.update_stats({**matched, "conf": scores, "pred_cls": p_cls,
                                     "target_cls": gt_cls, "target_img": np.unique(gt_cls)})
    indices, aps = native_ap_from_stats(native)
    stats = {k: np.concatenate(v, 0) for k, v in native.metrics.stats.items()}
    reference = ap_per_class(stats["tp"], stats["conf"], stats["pred_cls"], stats["target_cls"])
    np.testing.assert_array_equal(indices, reference[6].astype(int))
    np.testing.assert_allclose(aps, reference[5], rtol=0, atol=0)

    records = []
    for i, (w, h) in enumerate(((1001, 501), (501, 1001), (320, 320))):
        image_path = output / f"synthetic_{i}.png"
        Image.new("RGB", (w, h), (23, 87, 131)).save(image_path)
        records.append({"id": str(i), "image_path": str(image_path), "width": w, "height": h,
                        "annotations": [{"class_id": 0, "xywh_normalized": [.5, .5, .04, .03]}]})
    maximum_error = 0.0
    tensor_shapes = {}
    for rect in (False, True):
        dataset = ReadOnlyDataset(records, data={"names": {0: "a"}, "nc": 1, "channels": 3},
                                  imgsz=640, batch_size=2, augment=False, hyp=native.args,
                                  rect=rect, stride=32, pad=.5, cache=False)
        loader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_fn)
        tensor_shapes[str(rect)] = []
        for batch in loader:
            tensor_shapes[str(rect)].append(list(batch["img"].shape))
            for i in range(len(batch["im_file"])):
                original = next(r for r in records if r["image_path"] == batch["im_file"][i])
                prepared = native._prepare_batch(i, batch)
                recovered = restore_boxes(prepared["bboxes"].numpy(), batch["ratio_pad"][i],
                                          original["width"], original["height"])
                expected = np.array([[.48 * original["width"], .485 * original["height"],
                                      .52 * original["width"], .515 * original["height"]]])
                np.testing.assert_allclose(recovered, expected, rtol=0, atol=.00015)
                maximum_error = max(maximum_error, float(np.abs(recovered - expected).max()))
    write_json(output / "checks.json", {"native_AP_matches_project_ap_per_class_exactly": True,
                                        "CLI_rect_matches_actual_Model_val_default": True,
                                        "synthetic_tensor_shapes_by_rect": tensor_shapes,
                                        "rectangular_and_square_coordinate_roundtrip_passed": True,
                                        "maximum_roundtrip_error_pixels": maximum_error,
                                        "real_dataset_evaluated": False})
    print(f"PASS: native AP parity and native transform coordinate roundtrip. Evidence: {output}")


if __name__ == "__main__":
    main()
