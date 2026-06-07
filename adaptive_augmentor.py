import os
import sys
import cv2
import yaml
import random
import shutil
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

# =========================
# 0. ultralytics 双后端适配
# =========================

STF_YOLO_SEARCH_PATHS = [
    os.path.expanduser("~/PycharmProjects/STF-YOLO"),
    os.path.expanduser("~/STF-YOLO"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "STF-YOLO"),
]

YOLO_WEIGHTS_URL = {
    "n": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
    "s": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt",
    "m": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt",
    "l": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt",
    "x": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt",
}


def find_stf_yolo(explicit_path=None):
    if explicit_path and os.path.isdir(explicit_path):
        return explicit_path
    env_path = os.environ.get("STF_YOLO_PATH")
    if env_path and os.path.isdir(env_path):
        return env_path
    for p in STF_YOLO_SEARCH_PATHS:
        if os.path.isdir(p):
            return p
    return None


def init_ultralytics(stf_yolo_path=None, force_backend=None):
    stf_path = find_stf_yolo(stf_yolo_path)

    if force_backend == "yolov8":
        _clean_stf_path(stf_path)
        from ultralytics import YOLO
        print("后端: 系统 ultralytics (标准 YOLOv8)")
        return YOLO, "yolov8"

    if force_backend == "stf-yolo":
        if not stf_path:
            raise FileNotFoundError("指定 stf-yolo 后端但未找到 STF-YOLO 目录")
        if stf_path not in sys.path:
            sys.path.insert(0, stf_path)
        from ultralytics import YOLO
        print(f"后端: STF-YOLO ultralytics ({stf_path})")
        return YOLO, "stf-yolo"

    # 自动检测：先尝试 STF-YOLO
    if stf_path:
        if stf_path not in sys.path:
            sys.path.insert(0, stf_path)
        try:
            import importlib
            if 'ultralytics' in sys.modules:
                del sys.modules['ultralytics']
            from ultralytics import YOLO
            print(f"后端: STF-YOLO ultralytics ({stf_path})")
            return YOLO, "stf-yolo"
        except Exception as e:
            print(f"STF-YOLO ultralytics 加载失败: {e}")

    # 降级到系统 ultralytics
    _clean_stf_path(stf_path)
    try:
        import importlib
        if 'ultralytics' in sys.modules:
            del sys.modules['ultralytics']
        from ultralytics import YOLO
        print("后端: 系统 ultralytics (标准 YOLOv8)")
        return YOLO, "yolov8"
    except ImportError:
        raise RuntimeError("ultralytics 未安装，请执行: pip install ultralytics")


def _clean_stf_path(stf_path):
    if stf_path and stf_path in sys.path:
        sys.path.remove(stf_path)
    for p in list(sys.modules.keys()):
        if p.startswith('ultralytics'):
            del sys.modules[p]


def load_model(YOLO, weights_path, backend, stf_yolo_path=None):
    try:
        model = YOLO(weights_path)
        print(f"模型加载成功: {weights_path}")
        return model
    except (KeyError, RuntimeError, Exception) as e:
        if backend == "stf-yolo":
            print(f"自定义模型加载失败 ({e})，降级到标准 YOLOv8")
            YOLO_vanilla, _ = init_ultralytics(stf_yolo_path=None, force_backend="yolov8")
            size = "n"
            for s in ["n", "s", "m", "l", "x"]:
                if f"yolov8{s}" in weights_path.lower():
                    size = s
                    break
            fallback = download_yolov8_weights(size)
            return YOLO_vanilla(fallback)
        raise


def download_yolov8_weights(size="n", save_dir=None):
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
    os.makedirs(save_dir, exist_ok=True)
    filename = f"yolov8{size}.pt"
    save_path = os.path.join(save_dir, filename)
    if os.path.exists(save_path):
        return save_path
    url = YOLO_WEIGHTS_URL.get(size)
    if not url:
        raise ValueError(f"不支持的模型大小: {size}")
    print(f"下载 {filename} ...")
    import urllib.request
    urllib.request.urlretrieve(url, save_path)
    print(f"下载完成: {save_path}")
    return save_path


# =========================
# 1. 基础工具
# =========================

IMG_SUFFIX = (".jpg", ".jpeg", ".png", ".bmp")


def conservative_sr_upscale(img, out_size, alpha=0.18):
    ow, oh = out_size
    up = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(up, (0, 0), sigmaX=1.0, sigmaY=1.0)
    sharp = cv2.addWeighted(up, 1.0 + alpha, blur, -alpha, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def stem_and_ext(filename):
    stem, ext = os.path.splitext(filename)
    return stem, ext.lower()


def list_images(img_dir):
    return [f for f in os.listdir(img_dir) if f.lower().endswith(IMG_SUFFIX)]


def label_path_from_img(label_dir, img_name):
    stem, _ = stem_and_ext(img_name)
    return os.path.join(label_dir, f"{stem}.txt")


def xywh2xyxy_px(box, w, h):
    cx, cy, bw, bh = box
    return [
        (cx - bw / 2.0) * w,
        (cy - bh / 2.0) * h,
        (cx + bw / 2.0) * w,
        (cy + bh / 2.0) * h,
    ]


def xyxy2xywh_norm(box, w, h):
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    cx, cy = x1 + bw / 2.0, y1 + bh / 2.0
    return [cx / w, cy / h, bw / w, bh / h]


def parse_yolo_label_lines(lines):
    out = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        arr = s.split()
        if len(arr) < 5:
            continue
        cls, x, y, bw, bh = map(float, arr[:5])
        out.append((int(cls), x, y, bw, bh))
    return out


def compute_iou(box1, box2):
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(0.0, float(box1[3]) - float(box1[1]))
    area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(0.0, float(box2[3]) - float(box2[1]))
    return inter / (area1 + area2 - inter + 1e-6)


def box_area_ratio_in_crop(gt_box, crop_box):
    g_x1, g_y1, g_x2, g_y2 = gt_box
    c_x1, c_y1, c_x2, c_y2 = crop_box
    inter_x1 = max(g_x1, c_x1)
    inter_y1 = max(g_y1, c_y1)
    inter_x2 = min(g_x2, c_x2)
    inter_y2 = min(g_y2, c_y2)
    inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    gt_area = max(1.0, (g_x2 - g_x1) * (g_y2 - g_y1))
    return inter / gt_area


def classify_view_distance(lines, stats):
    parsed = parse_yolo_label_lines(lines)
    if not parsed:
        return "unknown"
    areas = np.array([bw * bh for _, _, _, bw, bh in parsed], dtype=np.float32)
    small_ratio = float(np.mean(areas < stats["high_threshold"]))
    large_ratio = float(np.mean(areas > stats["low_threshold"]))
    if small_ratio >= 0.4:
        return "far"
    if large_ratio >= 0.3:
        return "near"
    return "mid"


def letterbox_resize(img, out_size=(720, 720), color=(0, 0, 0), sr_mode="none", sr_engine=None):
    ow, oh = out_size
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((oh, ow, 3), dtype=np.uint8), 1.0, 0, 0

    scale = min(ow / w, oh / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))

    if sr_mode == "realesrgan" and sr_engine is not None and scale >= 1.2:
        upscaled = sr_engine.enhance(img, outscale=None)
        resized = cv2.resize(upscaled, (nw, nh), interpolation=cv2.INTER_AREA)
    elif sr_mode == "conservative" and scale >= 1.2:
        resized = conservative_sr_upscale(img, (nw, nh), alpha=0.18)
    else:
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)

    canvas = np.full((oh, ow, 3), color, dtype=np.uint8)
    dx = (ow - nw) // 2
    dy = (oh - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = resized
    return canvas, scale, dx, dy


def smart_resize(img, out_size=(720, 720), sr_mode="none", sr_engine=None):
    return letterbox_resize(img, out_size=out_size, sr_mode=sr_mode, sr_engine=sr_engine)[0]


def transform_and_filter(lines, img_shape, crop_box, out_size=(720, 720), min_size_px_after=3):
    orig_h, orig_w = img_shape[:2]
    cx1, cy1, cx2, cy2 = crop_box
    cw, ch = max(1, cx2 - cx1), max(1, cy2 - cy1)
    ow, oh = out_size

    labels = []
    parsed = parse_yolo_label_lines(lines)

    scale = min(ow / max(1, cw), oh / max(1, ch))
    nw, nh = int(round(cw * scale)), int(round(ch * scale))
    dx = (ow - nw) // 2
    dy = (oh - nh) // 2

    for cls, x, y, bw, bh in parsed:
        px_box = xywh2xyxy_px([x, y, bw, bh], orig_w, orig_h)
        nx1 = max(0.0, px_box[0] - cx1)
        ny1 = max(0.0, px_box[1] - cy1)
        nx2 = min(float(cw), px_box[2] - cx1)
        ny2 = min(float(ch), px_box[3] - cy1)
        if nx2 <= nx1 or ny2 <= ny1:
            continue

        bw_after = (nx2 - nx1) * scale
        bh_after = (ny2 - ny1) * scale
        if bw_after < min_size_px_after or bh_after < min_size_px_after:
            continue

        lnx1 = nx1 * scale + dx
        lny1 = ny1 * scale + dy
        lnx2 = nx2 * scale + dx
        lny2 = ny2 * scale + dy

        norm = xyxy2xywh_norm([lnx1, lny1, lnx2, lny2], ow, oh)
        norm = [np.clip(v, 0.0, 1.0) for v in norm]
        labels.append(f"{int(cls)} {norm[0]:.6f} {norm[1]:.6f} {norm[2]:.6f} {norm[3]:.6f}")

    return labels


# =========================
# 2. 数据集统计
# =========================

def analyze_dataset_scale(train_img_dir, train_lab_dir):
    areas = []
    img_files = list_images(train_img_dir)

    for img_name in tqdm(img_files, desc="扫描标注统计尺度"):
        img_path = os.path.join(train_img_dir, img_name)
        lab_path = label_path_from_img(train_lab_dir, img_name)
        if not os.path.exists(lab_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        with open(lab_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        for _, _, _, bw, bh in parse_yolo_label_lines(lines):
            rel_area = float(bw * bh)
            px_area = rel_area * w * h
            if px_area >= 1.0:
                areas.append(rel_area)

    if len(areas) == 0:
        return {
            "area_median": 0.002, "area_p25": 0.001, "area_p75": 0.004,
            "high_threshold": 0.001, "low_threshold": 0.004,
            "small_ratio": 0.25, "medium_ratio": 0.50, "large_ratio": 0.25,
        }

    arr = np.array(areas, dtype=np.float32)
    p25, med, p75 = np.percentile(arr, [25, 50, 75])
    high_thresh = float(max(1e-6, p25))
    low_thresh = float(max(p75, p25 + 1e-6))
    small_cnt = np.sum(arr < high_thresh)
    large_cnt = np.sum(arr > low_thresh)
    total = len(areas)

    return {
        "area_median": float(med), "area_p25": float(p25), "area_p75": float(p75),
        "high_threshold": high_thresh, "low_threshold": low_thresh,
        "small_ratio": float(small_cnt / total) if total > 0 else 0.25,
        "large_ratio": float(large_cnt / total) if total > 0 else 0.25,
    }


# =========================
# 3. 指标闭环加权
# =========================

def safe_get_metric(d, keys, default=0.0):
    for k in keys:
        if k in d and d[k] is not None:
            return float(d[k])
    return float(default)


def compute_adaptive_weights(model, data_yaml, stats, imgsz=720):
    print("正在 val 上评估模型并计算自适应权重...")
    val_res = model.val(data=data_yaml, verbose=False, imgsz=imgsz, rect=False)
    rd = getattr(val_res, "results_dict", {}) or {}

    recall = safe_get_metric(rd, ["metrics/recall(B)"], default=0.5)
    precision = safe_get_metric(rd, ["metrics/precision(B)"], default=0.5)

    n = max(0.0, 1.0 - recall)
    m = max(0.0, 1.0 - precision)

    target_small_ratio = 0.40
    target_large_ratio = 0.30
    current_small_ratio = stats.get("small_ratio", 0.25)
    current_large_ratio = stats.get("large_ratio", 0.25)

    q = max(0.0, target_large_ratio - current_large_ratio)
    p = max(0.0, target_small_ratio - current_small_ratio)

    raw = np.array([n, m, q, p], dtype=np.float32)
    raw = raw + 0.1
    weights = raw / (raw.sum() + 1e-6)

    print(f"val指标: Recall={recall:.4f}, Precision={precision:.4f}")
    print(f"数据集分布: small_ratio={current_small_ratio:.3f}, large_ratio={current_large_ratio:.3f}")
    print(f"自动策略比例 (n:m:q:p) = {raw[0]:.3f}:{raw[1]:.3f}:{raw[2]:.3f}:{raw[3]:.3f}")
    print(f"归一化概率 [FN, FP_BG, ALT, SMALL_ZOOM] = {weights.tolist()}")

    return {
        "raw_ratio": raw, "weights": weights,
        "metrics": {
            "recall": recall, "precision": precision,
            "small_ratio": current_small_ratio, "large_ratio": current_large_ratio,
        },
    }


# =========================
# 4. 误差样本挖掘
# =========================

def get_true_fn_and_fp_cases(model, img_dir, label_dir, conf=0.25, iou_thr=0.5, sample_limit=800):
    fn_cases = []
    fp_cases = []

    print("正在推理定位 FN / FP...")
    img_list_all = list_images(img_dir)
    actual_limit = min(sample_limit, len(img_list_all))
    if len(img_list_all) > actual_limit:
        img_list = random.sample(img_list_all, actual_limit)
        print(f"误差挖掘采样: {actual_limit}/{len(img_list_all)}")
    else:
        img_list = img_list_all

    for img_name in tqdm(img_list, desc="扫描误差样本"):
        img_path = os.path.join(img_dir, img_name)
        lab_path = label_path_from_img(label_dir, img_name)
        if not os.path.exists(lab_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        with open(lab_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        gt = parse_yolo_label_lines(lines)
        gt_boxes = [xywh2xyxy_px([x, y, bw, bh], w, h) for _, x, y, bw, bh in gt]

        pred = model.predict(img, conf=conf, verbose=False)[0]
        pred_boxes = pred.boxes.xyxy.cpu().numpy() if pred.boxes is not None else np.zeros((0, 4), dtype=np.float32)

        for g in gt_boxes:
            matched = any(compute_iou(g, p) > iou_thr for p in pred_boxes)
            if not matched:
                fn_cases.append({"img_path": img_path, "gt_box": g, "lines": lines})

        for p in pred_boxes:
            matched = any(compute_iou(p, g) > iou_thr for g in gt_boxes)
            if not matched:
                fp_cases.append({"img_path": img_path, "fp_box": p, "lines": lines})

    return fn_cases, fp_cases


# =========================
# 5. 四类策略实现
# =========================

def augment_fn_crop(img, lines, target_px_box, stats, out_size=(720, 720),
                    sr_mode="none", sr_engine=None, density_overlap_thresh=0.3):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = target_px_box

    bw_t = x2 - x1
    bh_t = y2 - y1
    area = (bw_t * bh_t) / (w * h)

    if area < stats["high_threshold"]:
        scale = random.uniform(3.5, 6.0)
    elif area > stats["low_threshold"]:
        scale = random.uniform(1.8, 3.0)
    else:
        scale = random.uniform(2.5, 4.0)

    if lines:
        try:
            parsed = parse_yolo_label_lines(lines)
            gt_boxes = []
            for cls_id, cx_r, cy_r, bw_r, bh_r in parsed:
                g_x1 = (cx_r - bw_r / 2) * w
                g_y1 = (cy_r - bh_r / 2) * h
                g_x2 = (cx_r + bw_r / 2) * w
                g_y2 = (cy_r + bh_r / 2) * h
                gt_boxes.append([g_x1, g_y1, g_x2, g_y2])

            cx_gt = (x1 + x2) / 2.0
            cy_gt = (y1 + y2) / 2.0
            half_w = bw_t * scale / 2
            half_h = bh_t * scale / 2
            rough_c_x1 = max(0, cx_gt - half_w)
            rough_c_y1 = max(0, cy_gt - half_h)
            rough_c_x2 = min(w, cx_gt + half_w)
            rough_c_y2 = min(h, cy_gt + half_h)

            num_objs = 0
            for g in gt_boxes:
                if box_area_ratio_in_crop(g, [rough_c_x1, rough_c_y1, rough_c_x2, rough_c_y2]) > density_overlap_thresh:
                    num_objs += 1

            if num_objs >= 3:
                scale *= 1.2
            elif num_objs == 1:
                scale *= 0.9
        except Exception:
            pass

    bw, bh = bw_t * scale, bh_t * scale
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

    c_x1 = int(max(0, cx - bw / 2))
    c_y1 = int(max(0, cy - bh / 2))
    c_x2 = int(min(w, cx + bw / 2))
    c_y2 = int(min(h, cy + bh / 2))

    if c_x2 <= c_x1 or c_y2 <= c_y1:
        return None, None

    crop = img[c_y1:c_y2, c_x1:c_x2]
    if crop.size == 0:
        return None, None

    labels = transform_and_filter(lines, img.shape, [c_x1, c_y1, c_x2, c_y2], out_size=out_size)
    if not labels:
        return None, None

    out = smart_resize(crop, out_size=out_size, sr_mode=sr_mode, sr_engine=sr_engine)
    return out, labels


def augment_fp_background(img, lines, fp_box, out_size=(720, 720),
                          strict_negative=True, sr_mode="none", sr_engine=None):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = fp_box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = (x2 - x1), (y2 - y1)
    scale = random.uniform(2.5, 5.0)

    rw, rh = bw * scale, bh * scale
    c_x1 = int(max(0, cx - rw / 2))
    c_y1 = int(max(0, cy - rh / 2))
    c_x2 = int(min(w, cx + rw / 2))
    c_y2 = int(min(h, cy + rh / 2))

    if c_x2 <= c_x1 or c_y2 <= c_y1:
        return None, None

    crop = img[c_y1:c_y2, c_x1:c_x2]
    if crop.size == 0:
        return None, None

    labels = transform_and_filter(lines, img.shape, [c_x1, c_y1, c_x2, c_y2], out_size=out_size)
    if strict_negative and labels:
        return None, None

    out = smart_resize(crop, out_size=out_size, sr_mode=sr_mode, sr_engine=sr_engine)
    return out, labels


def simulate_altitude(img, lines, mode="high", sr_mode="none", sr_engine=None):
    h, w = img.shape[:2]
    if mode == "high":
        return img, lines
    else:
        x1, y1, x2, y2 = int(0.25 * w), int(0.25 * h), int(0.75 * w), int(0.75 * h)
        crop = img[y1:y2, x1:x2]
        labels = transform_and_filter(lines, img.shape, [x1, y1, x2, y2], out_size=(w, h))
        out = smart_resize(crop, out_size=(w, h), sr_mode=sr_mode, sr_engine=sr_engine)
        return out, labels


def small_object_zoom(img, lines, stats, sr_mode="none", sr_engine=None, max_crop_size=None, imgsz=720):
    parsed = parse_yolo_label_lines(lines)
    if not parsed:
        return None, None

    areas = np.array([bw * bh for _, _, _, bw, bh in parsed], dtype=np.float32)
    small_ratio = float(np.mean(areas < stats["high_threshold"]))

    h, w = img.shape[:2]
    if small_ratio > 0.5:
        target = min(parsed, key=lambda x: x[3] * x[4])
        cls_id, cx_r, cy_r, bw_r, bh_r = target
        cx, cy = cx_r * w, cy_r * h

        scale = random.uniform(4.0, 8.0)
        bw_px = bw_r * w * scale
        bh_px = bh_r * h * scale

        if max_crop_size is None:
            max_crop_size = int(imgsz * 0.6)
        bw_px = min(bw_px, max_crop_size)
        bh_px = min(bh_px, max_crop_size)

        x1 = int(max(0, cx - bw_px / 2))
        y1 = int(max(0, cy - bh_px / 2))
        x2 = int(min(w, cx + bw_px / 2))
        y2 = int(min(h, cy + bh_px / 2))

        crop = img[y1:y2, x1:x2]
        if crop.size == 0 or (x2 <= x1) or (y2 <= y1):
            return None, None

        labs = transform_and_filter(lines, img.shape, [x1, y1, x2, y2], out_size=(w, h))
        out = smart_resize(crop, out_size=(w, h), sr_mode=sr_mode, sr_engine=sr_engine)
        return out, labs
    else:
        return simulate_altitude(img, lines, mode="high", sr_mode=sr_mode, sr_engine=sr_engine)


# =========================
# 6. 统计报告
# =========================

def analyze_dataset_detailed(img_dir, label_dir, stats):
    records = []
    img_list = list_images(img_dir)
    for img_name in tqdm(img_list, desc="统计数据集结构"):
        lab_path = label_path_from_img(label_dir, img_name)
        if not os.path.exists(lab_path):
            continue
        with open(lab_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        parsed = parse_yolo_label_lines(lines)
        if not parsed:
            continue
        areas = np.array([bw * bh for _, _, _, bw, bh in parsed])
        small = np.sum(areas < stats["high_threshold"])
        large = np.sum(areas > stats["low_threshold"])
        total = len(areas)
        view = classify_view_distance(lines, stats)
        records.append({
            "image": img_name, "num_objects": total,
            "small_count": small, "large_count": large,
            "small_ratio": small / total if total > 0 else 0,
            "large_ratio": large / total if total > 0 else 0,
            "view": view
        })
    return pd.DataFrame(records)


def analyze_errors(fn_cases, fp_cases):
    fn_sizes = []
    fp_sizes = []
    for c in fn_cases:
        x1, y1, x2, y2 = c["gt_box"]
        fn_sizes.append((x2 - x1) * (y2 - y1))
    for c in fp_cases:
        x1, y1, x2, y2 = c["fp_box"]
        fp_sizes.append((x2 - x1) * (y2 - y1))
    return {
        "fn_count": len(fn_cases), "fp_count": len(fp_cases),
        "fn_mean_area": np.mean(fn_sizes) if fn_sizes else 0,
        "fp_mean_area": np.mean(fp_sizes) if fp_sizes else 0
    }


def save_reports(df_train, error_stats, fn_cases, fp_cases, out_base_dir):
    report_dir = os.path.join(out_base_dir, "reports")
    ensure_dir(report_dir)

    plt.figure()
    df_train["small_ratio"].hist(bins=30, color='skyblue', edgecolor='black')
    plt.title("Small Object Ratio Distribution")
    plt.savefig(os.path.join(report_dir, "small_ratio_hist.png"))
    plt.close()

    plt.figure()
    df_train["num_objects"].hist(bins=30, color='lightgreen', edgecolor='black')
    plt.title("Object Count Distribution")
    plt.savefig(os.path.join(report_dir, "density_hist.png"))
    plt.close()

    plt.figure()
    df_train["view"].value_counts().plot(kind='bar', color='coral')
    plt.title("View Distribution")
    plt.savefig(os.path.join(report_dir, "view_dist.png"))
    plt.close()

    fn_dir = os.path.join(report_dir, "fn_samples")
    fp_dir = os.path.join(report_dir, "fp_samples")
    ensure_dir(fn_dir)
    ensure_dir(fp_dir)

    for i, c in enumerate(fn_cases[:50]):
        img = cv2.imread(c["img_path"])
        if img is not None:
            x1, y1, x2, y2 = map(int, c["gt_box"])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.imwrite(os.path.join(fn_dir, f"fn_{i}.jpg"), img)

    for i, c in enumerate(fp_cases[:50]):
        img = cv2.imread(c["img_path"])
        if img is not None:
            x1, y1, x2, y2 = map(int, c["fp_box"])
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.imwrite(os.path.join(fp_dir, f"fp_{i}.jpg"), img)

    return report_dir


# =========================
# 7. SR 引擎初始化
# =========================

def init_sr_engine(sr_mode, sr_weights=None, device='cuda:0'):
    if sr_mode != "realesrgan":
        return None

    if sr_weights is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sr_weights = os.path.join(script_dir, "sr", "weights", "realesr-general-x4v3.pth")

    if not os.path.exists(sr_weights):
        print(f"SR 权重不存在: {sr_weights}")
        print("降级到 conservative SR 模式")
        return "fallback_conservative"

    try:
        from sr.upsampler import RealESRGANUpsampler
        engine = RealESRGANUpsampler(
            model_path=sr_weights, scale=4, tile=128, device=device
        )
        print(f"Real-ESRGAN 加载成功: {sr_weights}")
        return engine
    except Exception as e:
        print(f"Real-ESRGAN 加载失败: {e}")
        print("降级到 conservative SR 模式")
        return "fallback_conservative"


# =========================
# 8. 数据集路径解析
# =========================

def resolve_data_paths(data_yaml_path):
    data = load_yaml(data_yaml_path)
    base_path = data.get("path", "")
    train_img = data["train"]

    if base_path and not os.path.isabs(train_img):
        train_img = os.path.join(base_path, train_img)

    if not os.path.isdir(train_img):
        print(f"警告: 训练图片目录不存在: {train_img}")

    train_lab = str(Path(train_img).parent.parent / "labels" / Path(train_img).name)

    if not os.path.isdir(train_lab):
        alt_lab = train_img.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
        if os.path.isdir(alt_lab):
            train_lab = alt_lab

    return data, train_img, train_lab


# =========================
# 9. 主流程
# =========================

def run_pipeline(data_yaml, weights_path, out_base_dir, ratio=1.35,
                 fp_strict_negative=True, sample_limit=800, sr_mode="none",
                 sr_weights=None, imgsz=720, device='cuda:0',
                 stf_yolo_path=None, backend=None):

    YOLO, actual_backend = init_ultralytics(stf_yolo_path, backend)
    print(f"实际后端: {actual_backend}")

    data, train_img_src, train_lab_src = resolve_data_paths(data_yaml)
    print(f"训练图片: {train_img_src}")
    print(f"训练标签: {train_lab_src}")

    out_size = (imgsz, imgsz)

    ensure_dir(os.path.join(out_base_dir, "images", "train"))
    ensure_dir(os.path.join(out_base_dir, "labels", "train"))
    ensure_dir(os.path.join(out_base_dir, "images", "new"))
    ensure_dir(os.path.join(out_base_dir, "labels", "new"))

    print("同步原始训练集...")
    all_imgs = list_images(train_img_src)
    for f in tqdm(all_imgs, desc="Copy train"):
        src_img = os.path.join(train_img_src, f)
        dst_img = os.path.join(out_base_dir, "images", "train", f)
        shutil.copy(src_img, dst_img)
        src_lab = label_path_from_img(train_lab_src, f)
        if os.path.exists(src_lab):
            dst_lab = os.path.join(out_base_dir, "labels", "train", os.path.basename(src_lab))
            shutil.copy(src_lab, dst_lab)

    sr_engine = init_sr_engine(sr_mode, sr_weights, device)
    if sr_engine == "fallback_conservative":
        sr_mode = "conservative"
        sr_engine = None

    model = load_model(YOLO, weights_path, actual_backend, stf_yolo_path)

    stats = analyze_dataset_scale(train_img_src, train_lab_src)
    print(f"ScaleStats: {stats}")

    aw = compute_adaptive_weights(model, data_yaml, stats, imgsz=imgsz)
    strategy_weights = aw["weights"]

    fn_cases, fp_cases = get_true_fn_and_fp_cases(
        model=model, img_dir=train_img_src, label_dir=train_lab_src,
        conf=0.25, iou_thr=0.5, sample_limit=sample_limit,
    )
    print(f"FN={len(fn_cases)} | FP={len(fp_cases)}")

    if len(fn_cases) == 0:
        print("警告: FN 样本池为空，FN_CROP 策略将使用随机图 fallback")
    if len(fp_cases) == 0:
        print("警告: FP 样本池为空，FP_BG 策略将使用随机图 fallback")

    print("正在生成统计报告...")
    df_train = analyze_dataset_detailed(train_img_src, train_lab_src, stats)
    error_stats = analyze_errors(fn_cases, fp_cases)
    save_reports(df_train, error_stats, fn_cases, fp_cases, out_base_dir)

    summary = {
        "总样本图数": len(df_train),
        "单图平均目标密度": df_train["num_objects"].mean() if not df_train.empty else 0,
        "整体偏远景比例": (df_train["view"] == "far").mean() if not df_train.empty else 0,
        "整体偏近景比例": (df_train["view"] == "near").mean() if not df_train.empty else 0,
        "检出漏检 (FN) 数量": error_stats["fn_count"],
        "检出误报 (FP) 数量": error_stats["fp_count"],
    }

    print("\n======================================")
    print("数据集 & 模型误差结构总结：")
    for k, v in summary.items():
        print(f"   - {k}: {v:.4f}")
    print("======================================\n")

    report_dir = os.path.join(out_base_dir, "reports")
    ensure_dir(report_dir)
    decision_params = {
        "summary": {k: float(v) for k, v in summary.items()},
        "dataset_stats": {k: float(v) if isinstance(v, (int, float, np.float32, np.float64)) else v for k, v in stats.items()},
        "strategy_weights": {
            "FN_CROP": float(strategy_weights[0]),
            "FP_BG": float(strategy_weights[1]),
            "ALT_SIM": float(strategy_weights[2]),
            "SMALL_ZOOM": float(strategy_weights[3])
        }
    }
    with open(os.path.join(report_dir, "decision_params.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(decision_params, f, allow_unicode=True, default_flow_style=False)

    target_new = int(len(all_imgs) * max(0.0, ratio - 1.0))
    print(f"计划新增样本: {target_new}")
    if target_new <= 0:
        print("ratio <= 1.0，不生成新增样本。")
        return

    strategies = ["FN_CROP", "FP_BG", "ALT_SIM", "SMALL_ZOOM"]
    idx = 0
    empty_label_count = 0
    pbar = tqdm(total=target_new, desc="生成增强样本")

    max_trials = target_new * 30
    trials = 0

    while idx < target_new and trials < max_trials:
        trials += 1
        strat = random.choices(strategies, weights=strategy_weights, k=1)[0]
        aug_img, aug_labels = None, None

        if strat == "FN_CROP" and fn_cases:
            c = random.choice(fn_cases)
            img = cv2.imread(c["img_path"])
            if img is not None:
                aug_img, aug_labels = augment_fn_crop(
                    img, c["lines"], c["gt_box"], stats,
                    out_size=out_size, sr_mode=sr_mode, sr_engine=sr_engine)

        elif strat == "FP_BG" and fp_cases:
            c = random.choice(fp_cases)
            img = cv2.imread(c["img_path"])
            if img is not None:
                aug_img, aug_labels = augment_fp_background(
                    img=img, lines=c["lines"], fp_box=c["fp_box"],
                    out_size=out_size, strict_negative=fp_strict_negative,
                    sr_mode=sr_mode, sr_engine=sr_engine)

        else:
            img_name = random.choice(all_imgs)
            img_path = os.path.join(train_img_src, img_name)
            lab_path = label_path_from_img(train_lab_src, img_name)
            if not os.path.exists(lab_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue

            with open(lab_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]

            if strat == "ALT_SIM":
                dist = classify_view_distance(lines, stats)
                if dist == "far":
                    mode = "low" if random.random() < 0.85 else "skip"
                elif dist == "near":
                    mode = "high" if random.random() < 0.7 else "low"
                else:
                    mode = random.choice(["high", "low"])

                if mode == "skip":
                    continue
                aug_img, aug_labels = simulate_altitude(
                    img, lines, mode=mode, sr_mode=sr_mode, sr_engine=sr_engine)
            else:
                aug_img, aug_labels = small_object_zoom(
                    img, lines, stats, sr_mode=sr_mode, sr_engine=sr_engine, imgsz=imgsz)

        if aug_img is None:
            continue

        if not aug_labels:
            empty_label_count += 1
            max_empty = int(target_new * 0.3)
            if empty_label_count > max_empty:
                continue

        name = f"aug_sys_{idx:06d}"
        out_img_path = os.path.join(out_base_dir, "images", "new", f"{name}.jpg")
        out_lab_path = os.path.join(out_base_dir, "labels", "new", f"{name}.txt")

        cv2.imwrite(out_img_path, aug_img)
        with open(out_lab_path, "w", encoding="utf-8") as f:
            if aug_labels:
                f.write("\n".join(aug_labels))
            else:
                f.write("")

        idx += 1
        pbar.update(1)

    pbar.close()

    if idx < target_new:
        print(f"提前结束：成功生成 {idx}/{target_new}（可适当放宽 strict_negative 或提高 sample_limit）")
    else:
        print("增强完成。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="模型驱动自适应数据增强 (支持 STF-YOLO / YOLOv8 双后端)")
    parser.add_argument("--data", type=str, required=True, help="data.yaml 路径")
    parser.add_argument("--weights", type=str, required=True, help="YOLO .pt 权重路径")
    parser.add_argument("--out", type=str, default="output_aug", help="输出目录")
    parser.add_argument("--ratio", type=float, default=1.35, help="总量比例，1.35表示新增35%%")
    parser.add_argument("--sample_limit", type=int, default=800, help="FN/FP挖掘采样上限")
    parser.add_argument("--fp_strict_negative", action="store_true", help="FP背景仅保留纯负样本")
    parser.add_argument("--sr_mode", type=str, default="none",
                        choices=["none", "conservative", "realesrgan"],
                        help="超分模式：none / conservative / realesrgan")
    parser.add_argument("--sr_weights", type=str, default=None, help="Real-ESRGAN 权重路径")
    parser.add_argument("--imgsz", type=int, default=720, help="输出图像尺寸")
    parser.add_argument("--device", type=str, default="cuda:0", help="GPU 设备")
    parser.add_argument("--stf_yolo", type=str, default=None, help="STF-YOLO 项目路径")
    parser.add_argument("--backend", type=str, default=None,
                        choices=["stf-yolo", "yolov8"],
                        help="强制指定后端（默认自动检测）")
    args = parser.parse_args()

    run_pipeline(
        data_yaml=args.data,
        weights_path=args.weights,
        out_base_dir=args.out,
        ratio=args.ratio,
        fp_strict_negative=args.fp_strict_negative,
        sample_limit=args.sample_limit,
        sr_mode=args.sr_mode,
        sr_weights=args.sr_weights,
        imgsz=args.imgsz,
        device=args.device,
        stf_yolo_path=args.stf_yolo,
        backend=args.backend,
    )

    """
    用法示例：

    # 自动检测后端（优先 STF-YOLO，不适配则 YOLOv8）
    python adaptive_augmentor.py --data data.yaml --weights yolov8n.pt --imgsz 720

    # 强制使用标准 YOLOv8
    python adaptive_augmentor.py --data data.yaml --weights yolov8n.pt --backend yolov8 --imgsz 720

    # 指定 STF-YOLO 路径
    python adaptive_augmentor.py --data data.yaml --weights best.pt --stf_yolo ~/PycharmProjects/STF-YOLO --imgsz 720

    # 启用 Real-ESRGAN 超分
    python adaptive_augmentor.py --data data.yaml --weights yolov8n.pt --sr_mode realesrgan --imgsz 720

    # 完整参数
    python adaptive_augmentor.py \\
      --data data.yaml \\
      --weights yolov8n.pt \\
      --out output_aug \\
      --ratio 1.35 \\
      --sample_limit 800 \\
      --fp_strict_negative \\
      --sr_mode realesrgan \\
      --imgsz 720 \\
      --device cuda:0
    """
