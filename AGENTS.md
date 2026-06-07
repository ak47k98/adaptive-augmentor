# AGENTS.md

## Overview

Python research tool for model-feedback-driven error repair in YOLO object detection. No build system, no tests, no CI, no package manager. Dependencies installed ad-hoc via pip.

## Dependencies

```
pip install ultralytics opencv-python pandas matplotlib tqdm pyyaml
```

SR module also needs `torch`. SR weights downloaded separately:

```bash
bash scripts/download_weights.sh
```

## Entry Points

| Script | Purpose |
|---|---|
| `closed_loop_framework.py` | **Primary entry.** 10-stage closed-loop repair pipeline (diagnosis → repair → validate → update policy) |
| `val_analyzer.py` | Validation analysis: standard val + dataset features + FN/FP defect analysis |
| `adaptive_augmentor.py` | Older v3 augmentation pipeline. Kept for compatibility |

## Running

```bash
# Diagnose + generate repair samples (round 1)
python closed_loop_framework.py --data data.yaml --weights best.pt --sr_mode realesrgan --imgsz 720 --out repair_round1

# Validate repair after retraining
python closed_loop_framework.py --data data.yaml --weights best.pt --repaired_weights runs/train/weights/best.pt --out repair_round1_validation

# Val analysis only
python val_analyzer.py --data data.yaml --weights best.pt --imgsz 720
```

All scripts require `--data` (YOLO data.yaml) and `--weights` (.pt file). Default `--imgsz` is **720** (not 640).

## Architecture

```
closed_loop_framework.py   → orchestrates 10 stages, calls modules below
error_diagnosis.py          → ErrorDiagnoser, ErrorType enum (7 FN + 3 FP types), run_diagnosis()
repair_operators.py         → 8 RepairOperator subclasses in OPERATOR_REGISTRY, create_operator() factory
repair_policy.py            → RepairPolicy (error→operator mapping + weight update), RepairValidator
sr/                         → self-contained Real-ESRGAN (no basicsr dependency)
  srvgg_arch.py             → SRVGGNetCompact nn.Module
  upsampler.py              → RealESRGANUpsampler (tile-based inference)
```

## Dual Backend (STF-YOLO / YOLOv8)

Every major script has its own copy of `init_ultralytics()` and `find_stf_yolo()`. Backend auto-detection order:

1. `--stf_yolo` CLI arg
2. `STF_YOLO_PATH` env var
3. `~/PycharmProjects/STF-YOLO`
4. `~/STF-YOLO`
5. Fallback to system `ultralytics` (standard YOLOv8)

The backend init manipulates `sys.path` and `sys.modules` to swap ultralytics implementations. If you modify one copy, the others (in `adaptive_augmentor.py`, `val_analyzer.py`, `closed_loop_framework.py`) are out of sync.

## Quirks

- **Duplicate utility functions**: `xywh2xyxy_px`, `compute_iou`, `parse_label_file`, `load_yaml`, `ensure_dir` are copy-pasted across multiple files (not a shared module).
- **matplotlib Agg backend**: `adaptive_augmentor.py` and `val_analyzer.py` call `matplotlib.use('Agg')` at import time. Required for headless/SSH environments.
- **SR fallback**: If `sr/weights/realesr-general-x4v3.pth` is missing, SR mode silently degrades to "conservative" (simple sharpened resize).
- **Chinese output**: All print statements and docstrings are in Chinese (中文). README is Chinese.
- **Label format**: YOLO normalized `class cx cy bw bh` per line. All images must have matching `.txt` label files.
- **Image suffixes**: `.jpg`, `.jpeg`, `.png`, `.bmp` (defined as `IMG_SUFFIX` tuple in each file).
