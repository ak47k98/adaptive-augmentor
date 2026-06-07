import os
import sys
import cv2
import yaml
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


# =========================
# 0. ultralytics 双后端适配（复用 adaptive_augmentor 逻辑）
# =========================

STF_YOLO_SEARCH_PATHS = [
    os.path.expanduser("~/PycharmProjects/STF-YOLO"),
    os.path.expanduser("~/STF-YOLO"),
]


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


def _clean_stf_path(stf_path):
    if stf_path and stf_path in sys.path:
        sys.path.remove(stf_path)
    for p in list(sys.modules.keys()):
        if p.startswith('ultralytics'):
            del sys.modules[p]


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


# =========================
# 1. 基础工具
# =========================

IMG_SUFFIX = (".jpg", ".jpeg", ".png", ".bmp")


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def xywh2xyxy_px(box, w, h):
    cx, cy, bw, bh = box
    return [
        (cx - bw / 2.0) * w,
        (cy - bh / 2.0) * h,
        (cx + bw / 2.0) * w,
        (cy + bh / 2.0) * h,
    ]


def compute_iou(box1, box2):
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(0.0, float(box1[3]) - float(box1[1]))
    area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(0.0, float(box2[3]) - float(box2[1]))
    return inter / (area1 + area2 - inter + 1e-6)


def object_size_category(area_px, img_area_px):
    """根据目标像素面积分类：小(<32x32) / 中(32x32~96x96) / 大(>96x96)"""
    side = np.sqrt(area_px)
    if side < 32:
        return "small"
    elif side < 96:
        return "medium"
    else:
        return "large"


def parse_label_file(label_path):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                boxes.append((cls, cx, cy, bw, bh))
    return boxes


# =========================
# 2. 数据集特征分析
# =========================

def analyze_dataset_features(val_img_dir, val_lab_dir, class_names, out_dir):
    """遍历 val 数据集，提取数据集特征"""
    records = []
    class_counts = defaultdict(int)
    size_counts = {"small": 0, "medium": 0, "large": 0}
    all_areas = []
    all_aspects = []

    img_files = [f for f in os.listdir(val_img_dir) if f.lower().endswith(IMG_SUFFIX)]

    for img_name in tqdm(img_files, desc="分析数据集特征"):
        img_path = os.path.join(val_img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_h, img_w = img.shape[:2]
        img_area = img_h * img_w

        stem = os.path.splitext(img_name)[0]
        lab_path = os.path.join(val_lab_dir, f"{stem}.txt")
        boxes = parse_label_file(lab_path)

        num_objs = len(boxes)
        small_n, medium_n, large_n = 0, 0, 0
        obj_areas = []

        for cls, cx, cy, bw, bh in boxes:
            area_px = bw * img_w * bh * img_h
            all_areas.append(area_px)
            all_aspects.append(bw / max(bh, 1e-6))
            obj_areas.append(area_px)
            class_counts[cls] += 1

            cat = object_size_category(area_px, img_area)
            size_counts[cat] += 1
            if cat == "small":
                small_n += 1
            elif cat == "medium":
                medium_n += 1
            else:
                large_n += 1

        records.append({
            "image": img_name,
            "width": img_w,
            "height": img_h,
            "num_objects": num_objs,
            "small_count": small_n,
            "medium_count": medium_n,
            "large_count": large_n,
            "mean_area_px": float(np.mean(obj_areas)) if obj_areas else 0,
            "min_area_px": float(np.min(obj_areas)) if obj_areas else 0,
        })

    df = pd.DataFrame(records)

    features = {
        "总图片数": len(df),
        "总目标数": sum(class_counts.values()),
        "平均每图目标数": float(df["num_objects"].mean()) if not df.empty else 0,
        "最大单图目标数": int(df["num_objects"].max()) if not df.empty else 0,
        "空图数量（无目标）": int((df["num_objects"] == 0).sum()) if not df.empty else 0,
        "目标尺寸分布": {
            "小目标(<32px侧边)": size_counts["small"],
            "中目标(32-96px)": size_counts["medium"],
            "大目标(>96px)": size_counts["large"],
        },
        "类别分布": {class_names.get(k, f"class_{k}"): v for k, v in sorted(class_counts.items())},
        "目标面积统计_px": {
            "mean": float(np.mean(all_areas)) if all_areas else 0,
            "median": float(np.median(all_areas)) if all_areas else 0,
            "min": float(np.min(all_areas)) if all_areas else 0,
            "max": float(np.max(all_areas)) if all_areas else 0,
            "p10": float(np.percentile(all_areas, 10)) if all_areas else 0,
            "p90": float(np.percentile(all_areas, 90)) if all_areas else 0,
        },
        "宽高比统计": {
            "mean": float(np.mean(all_aspects)) if all_aspects else 0,
            "median": float(np.median(all_aspects)) if all_aspects else 0,
        },
    }

    # 保存
    df.to_csv(os.path.join(out_dir, "dataset_features_per_image.csv"), index=False)

    with open(os.path.join(out_dir, "dataset_features_summary.json"), "w", encoding="utf-8") as f:
        json.dump(features, f, ensure_ascii=False, indent=2)

    # 可视化
    _plot_dataset_features(df, class_names, size_counts, all_areas, out_dir)

    print(f"数据集特征已保存到: {out_dir}")
    return features, df


def _plot_dataset_features(df, class_names, size_counts, all_areas, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 目标数量分布
    axes[0, 0].hist(df["num_objects"], bins=30, color='steelblue', edgecolor='black')
    axes[0, 0].set_title("Objects per Image Distribution")
    axes[0, 0].set_xlabel("Number of Objects")
    axes[0, 0].set_ylabel("Image Count")

    # 2. 尺寸分布饼图
    labels = ["Small (<32px)", "Medium (32-96px)", "Large (>96px)"]
    sizes = [size_counts["small"], size_counts["medium"], size_counts["large"]]
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    axes[0, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[0, 1].set_title("Object Size Distribution")

    # 3. 类别分布
    if class_names:
        cls_labels = [class_names.get(k, str(k)) for k in sorted(class_names.keys())]
        cls_vals = [size_counts.get(k, 0) for k in sorted(class_names.keys())]
        # 用 features 的类别计数
        axes[1, 0].barh(cls_labels, cls_vals, color='coral')
        axes[1, 0].set_title("Class Distribution")

    # 4. 目标面积分布
    if all_areas:
        log_areas = np.log10(np.array(all_areas) + 1)
        axes[1, 1].hist(log_areas, bins=40, color='lightgreen', edgecolor='black')
        axes[1, 1].set_title("Object Area Distribution (log10 px)")
        axes[1, 1].set_xlabel("log10(Area)")
        axes[1, 1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dataset_features.png"), dpi=150)
    plt.close()


# =========================
# 3. 模型缺陷分析
# =========================

def analyze_model_defects(model, val_img_dir, val_lab_dir, class_names,
                          conf=0.25, iou_thr=0.5, imgsz=720, out_dir="val_results"):
    """逐图推理，分析 FN/FP，按尺寸/类别/置信度统计"""

    ensure_dir(os.path.join(out_dir, "fn_samples"))
    ensure_dir(os.path.join(out_dir, "fp_samples"))

    img_files = sorted([f for f in os.listdir(val_img_dir) if f.lower().endswith(IMG_SUFFIX)])

    # 全局统计
    total_gt = 0
    total_pred = 0
    total_fn = 0
    total_fp = 0
    total_tp = 0

    fn_by_size = {"small": 0, "medium": 0, "large": 0}
    fn_by_class = defaultdict(int)
    fp_by_class = defaultdict(int)
    fp_by_conf = {"0.25-0.5": 0, "0.5-0.7": 0, "0.7-0.9": 0, "0.9-1.0": 0}

    per_image_results = []
    fn_details = []
    fp_details = []

    for img_name in tqdm(img_files, desc="逐图推理分析"):
        img_path = os.path.join(val_img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_h, img_w = img.shape[:2]
        img_area = img_h * img_w

        stem = os.path.splitext(img_name)[0]
        lab_path = os.path.join(val_lab_dir, f"{stem}.txt")
        gt_boxes_raw = parse_label_file(lab_path)

        gt_boxes = []
        for cls, cx, cy, bw, bh in gt_boxes_raw:
            px = xywh2xyxy_px([cx, cy, bw, bh], img_w, img_h)
            area_px = bw * img_w * bh * img_h
            gt_boxes.append({"cls": cls, "px": px, "area_px": area_px,
                             "size_cat": object_size_category(area_px, img_area)})

        # 推理
        results = model.predict(img, conf=conf, verbose=False, imgsz=imgsz)
        pred = results[0]
        pred_boxes = []
        if pred.boxes is not None and len(pred.boxes) > 0:
            for i in range(len(pred.boxes)):
                px = pred.boxes.xyxy[i].cpu().numpy().tolist()
                cls_id = int(pred.boxes.cls[i].cpu().item())
                conf_val = float(pred.boxes.conf[i].cpu().item())
                pred_boxes.append({"cls": cls_id, "px": px, "conf": conf_val})

        # 匈牙利匹配（贪心 IoU 匹配）
        gt_matched = [False] * len(gt_boxes)
        pred_matched = [False] * len(pred_boxes)

        # 按置信度降序排列预测
        pred_order = sorted(range(len(pred_boxes)), key=lambda i: pred_boxes[i]["conf"], reverse=True)

        for pi in pred_order:
            best_iou = 0
            best_gi = -1
            for gi, gt in enumerate(gt_boxes):
                if gt_matched[gi]:
                    continue
                iou = compute_iou(pred_boxes[pi]["px"], gt["px"])
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi
            if best_iou >= iou_thr and best_gi >= 0:
                if pred_boxes[pi]["cls"] == gt_boxes[best_gi]["cls"]:
                    gt_matched[best_gi] = True
                    pred_matched[pi] = True
                    total_tp += 1

        # FN: 未匹配的 GT
        for gi, gt in enumerate(gt_boxes):
            if not gt_matched[gi]:
                total_fn += 1
                fn_by_size[gt["size_cat"]] += 1
                fn_by_class[gt["cls"]] += 1
                fn_details.append({
                    "image": img_name,
                    "class": class_names.get(gt["cls"], str(gt["cls"])),
                    "class_id": gt["cls"],
                    "size_cat": gt["size_cat"],
                    "area_px": gt["area_px"],
                    "box": gt["px"],
                })

        # FP: 未匹配的预测
        for pi, pred_b in enumerate(pred_boxes):
            if not pred_matched[pi]:
                total_fp += 1
                fp_by_class[pred_b["cls"]] += 1
                conf_val = pred_b["conf"]
                if conf_val < 0.5:
                    fp_by_conf["0.25-0.5"] += 1
                elif conf_val < 0.7:
                    fp_by_conf["0.5-0.7"] += 1
                elif conf_val < 0.9:
                    fp_by_conf["0.7-0.9"] += 1
                else:
                    fp_by_conf["0.9-1.0"] += 1
                fp_details.append({
                    "image": img_name,
                    "class": class_names.get(pred_b["cls"], str(pred_b["cls"])),
                    "class_id": pred_b["cls"],
                    "confidence": conf_val,
                    "box": pred_b["px"],
                })

        total_gt += len(gt_boxes)
        total_pred += len(pred_boxes)

        # 每张图统计
        fn_count = sum(1 for m in gt_matched if not m)
        fp_count = sum(1 for m in pred_matched if not m)
        per_image_results.append({
            "image": img_name,
            "gt_count": len(gt_boxes),
            "pred_count": len(pred_boxes),
            "tp": total_tp,
            "fn": fn_count,
            "fp": fp_count,
            "small_gt": sum(1 for g in gt_boxes if g["size_cat"] == "small"),
            "small_fn": sum(1 for g, m in zip(gt_boxes, gt_matched) if not m and g["size_cat"] == "small"),
        })

    # =========================
    # 汇总报告
    # =========================
    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)

    defect_summary = {
        "标准指标": {
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-Score": round(f1, 4),
            "TP": total_tp,
            "FP": total_fp,
            "FN": total_fn,
            "GT总数": total_gt,
            "预测总数": total_pred,
        },
        "漏检分析（FN）": {
            "总漏检数": total_fn,
            "小目标漏检": fn_by_size["small"],
            "中目标漏检": fn_by_size["medium"],
            "大目标漏检": fn_by_size["large"],
            "按类别漏检": {class_names.get(k, f"class_{k}"): v for k, v in sorted(fn_by_class.items())},
        },
        "误检分析（FP）": {
            "总误检数": total_fp,
            "按置信度分布": fp_by_conf,
            "按类别误检": {class_names.get(k, f"class_{k}"): v for k, v in sorted(fp_by_class.items())},
        },
    }

    # 保存
    with open(os.path.join(out_dir, "defect_summary.json"), "w", encoding="utf-8") as f:
        json.dump(defect_summary, f, ensure_ascii=False, indent=2)

    pd.DataFrame(per_image_results).to_csv(os.path.join(out_dir, "per_image_results.csv"), index=False)
    pd.DataFrame(fn_details).to_csv(os.path.join(out_dir, "fn_details.csv"), index=False)
    pd.DataFrame(fp_details).to_csv(os.path.join(out_dir, "fp_details.csv"), index=False)

    # 可视化
    _plot_defect_analysis(defect_summary, fn_details, fp_details, out_dir)

    # 保存 FN/FP 样例图片（最多 30 张）
    _save_error_samples(fn_details, val_img_dir, out_dir, "fn", max_samples=30)
    _save_error_samples(fp_details, val_img_dir, out_dir, "fp", max_samples=30)

    # 打印摘要
    print("\n" + "=" * 60)
    print("模型缺陷分析报告")
    print("=" * 60)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"TP={total_tp}  FP={total_fp}  FN={total_fn}  GT={total_gt}")
    print(f"\n小目标漏检: {fn_by_size['small']} / {fn_by_size['small'] + fn_by_size['medium'] + fn_by_size['large']}")
    print(f"中目标漏检: {fn_by_size['medium']}")
    print(f"大目标漏检: {fn_by_size['large']}")
    if fn_by_class:
        print("\n按类别漏检:")
        for cls_id, count in sorted(fn_by_class.items(), key=lambda x: -x[1]):
            print(f"  {class_names.get(cls_id, f'class_{cls_id}')}: {count}")
    print("=" * 60)
    print(f"详细报告已保存到: {out_dir}/")

    return defect_summary


def _plot_defect_analysis(defect_summary, fn_details, fp_details, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. FN 按尺寸
    fn_size = defect_summary["漏检分析（FN）"]
    sizes = ["small", "medium", "large"]
    vals = [fn_size["小目标漏检"], fn_size["中目标漏检"], fn_size["大目标漏检"]]
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    axes[0, 0].bar(sizes, vals, color=colors)
    axes[0, 0].set_title(f"False Negatives by Size (Total: {fn_size['总漏检数']})")
    axes[0, 0].set_ylabel("Count")

    # 2. FN 按类别
    fn_cls = fn_size["按类别漏检"]
    if fn_cls:
        cls_names = list(fn_cls.keys())
        cls_vals = list(fn_cls.values())
        axes[0, 1].barh(cls_names, cls_vals, color='coral')
        axes[0, 1].set_title("False Negatives by Class")

    # 3. FP 按置信度
    fp_conf = defect_summary["误检分析（FP）"]["按置信度分布"]
    conf_labels = list(fp_conf.keys())
    conf_vals = list(fp_conf.values())
    axes[1, 0].bar(conf_labels, conf_vals, color='salmon')
    axes[1, 0].set_title(f"False Positives by Confidence (Total: {defect_summary['误检分析（FP）']['总误检数']})")
    axes[1, 0].set_ylabel("Count")

    # 4. FP 按类别
    fp_cls = defect_summary["误检分析（FP）"]["按类别误检"]
    if fp_cls:
        cls_names = list(fp_cls.keys())
        cls_vals = list(fp_cls.values())
        axes[1, 1].barh(cls_names, cls_vals, color='lightsalmon')
        axes[1, 1].set_title("False Positives by Class")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "defect_analysis.png"), dpi=150)
    plt.close()


def _save_error_samples(details, img_dir, out_dir, prefix, max_samples=30):
    sample_dir = os.path.join(out_dir, f"{prefix}_samples")
    ensure_dir(sample_dir)

    seen = set()
    count = 0
    for d in details:
        if d["image"] in seen:
            continue
        seen.add(d["image"])

        img_path = os.path.join(img_dir, d["image"])
        img = cv2.imread(img_path)
        if img is None:
            continue

        # 画同一张图上所有的 FN/FP
        same_img = [x for x in details if x["image"] == d["image"]]
        for item in same_img:
            x1, y1, x2, y2 = map(int, item["box"])
            if prefix == "fn":
                color = (0, 0, 255)  # 红色 = 漏检
                label = f"FN:{item['class']}"
            else:
                color = (0, 165, 255)  # 橙色 = 误检
                label = f"FP:{item['class']} {item.get('confidence', 0):.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        save_path = os.path.join(sample_dir, f"{prefix}_{d['image']}")
        cv2.imwrite(save_path, img)
        count += 1
        if count >= max_samples:
            break


# =========================
# 4. 标准 val 输出
# =========================

def run_standard_val(model, data_yaml, imgsz=720, out_dir="val_results"):
    """运行 ultralytics 标准 val，保存结果"""
    print("\n运行标准 YOLO val...")
    results = model.val(data=data_yaml, imgsz=imgsz, verbose=True, plots=True)

    val_results = {}
    rd = getattr(results, "results_dict", {}) or {}
    for k, v in rd.items():
        if v is not None:
            val_results[k] = float(v) if isinstance(v, (int, float, np.floating)) else str(v)

    with open(os.path.join(out_dir, "standard_val_results.json"), "w", encoding="utf-8") as f:
        json.dump(val_results, f, ensure_ascii=False, indent=2)

    print(f"标准 val 结果已保存到: {out_dir}/standard_val_results.json")
    return results


# =========================
# 5. 主流程
# =========================

def resolve_data_paths(data_yaml_path):
    data = load_yaml(data_yaml_path)
    base_path = data.get("path", "")
    val_img = data.get("val", data.get("train", ""))

    if base_path and not os.path.isabs(val_img):
        val_img = os.path.join(base_path, val_img)

    val_lab = str(Path(val_img).parent.parent / "labels" / Path(val_img).name)
    if not os.path.isdir(val_lab):
        alt_lab = val_img.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
        if os.path.isdir(alt_lab):
            val_lab = alt_lab

    return data, val_img, val_lab


def main():
    parser = argparse.ArgumentParser(description="模型 val 详细分析工具")
    parser.add_argument("--data", type=str, required=True, help="data.yaml 路径")
    parser.add_argument("--weights", type=str, required=True, help="模型权重 .pt")
    parser.add_argument("--imgsz", type=int, default=720, help="推理图像尺寸")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU 匹配阈值")
    parser.add_argument("--out", type=str, default="val_results", help="输出目录")
    parser.add_argument("--device", type=str, default="cuda:0", help="GPU 设备")
    parser.add_argument("--stf_yolo", type=str, default=None, help="STF-YOLO 路径")
    parser.add_argument("--backend", type=str, default=None, choices=["stf-yolo", "yolov8"])
    parser.add_argument("--skip_standard_val", action="store_true", help="跳过标准 val（只做缺陷分析）")
    args = parser.parse_args()

    ensure_dir(args.out)

    # 初始化后端
    YOLO, backend = init_ultralytics(args.stf_yolo, args.backend)
    print(f"后端: {backend}")

    # 加载数据集配置
    data, val_img_dir, val_lab_dir = resolve_data_paths(args.data)
    class_names = data.get("names", {})
    if isinstance(class_names, list):
        class_names = {i: n for i, n in enumerate(class_names)}
    print(f"验证集图片: {val_img_dir}")
    print(f"验证集标签: {val_lab_dir}")
    print(f"类别: {class_names}")

    # 加载模型
    model = YOLO(args.weights)
    print(f"模型加载: {args.weights}")

    # 1. 标准 val
    if not args.skip_standard_val:
        run_standard_val(model, args.data, imgsz=args.imgsz, out_dir=args.out)

    # 2. 数据集特征分析
    print("\n" + "=" * 60)
    print("数据集特征分析")
    print("=" * 60)
    features, df_features = analyze_dataset_features(
        val_img_dir, val_lab_dir, class_names, args.out)

    # 3. 模型缺陷分析
    print("\n" + "=" * 60)
    print("模型缺陷分析")
    print("=" * 60)
    defect_summary = analyze_model_defects(
        model, val_img_dir, val_lab_dir, class_names,
        conf=args.conf, iou_thr=args.iou, imgsz=args.imgsz, out_dir=args.out)

    print("\n分析完成！输出文件：")
    for f in sorted(os.listdir(args.out)):
        print(f"  {f}")


if __name__ == "__main__":
    main()

    """
    用法示例：

    # 标准用法（自动检测后端）
    python val_analyzer.py --data data.yaml --weights best.pt --imgsz 720

    # 指定 YOLOv8 后端
    python val_analyzer.py --data data.yaml --weights yolov8n.pt --backend yolov8 --imgsz 720

    # 跳过标准 val（只做缺陷分析）
    python val_analyzer.py --data data.yaml --weights best.pt --skip_standard_val

    # 低置信度阈值（发现更多潜在漏检）
    python val_analyzer.py --data data.yaml --weights best.pt --conf 0.1 --iou 0.5
    """
