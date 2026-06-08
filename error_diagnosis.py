"""
错误诊断模块 - Error Diagnosis & Taxonomy
对 FN/FP 进行细粒度分类，输出错误类型分布
"""

import os
import cv2
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

IMG_SUFFIX = (".jpg", ".jpeg", ".png", ".bmp")


# =========================
# 错误类型定义
# =========================

class ErrorType:
    # FN 错误类型
    SCALE_FN = "scale_fn"               # 目标太小
    BOUNDARY_FN = "boundary_fn"         # 目标在边界
    OCCLUSION_FN = "occlusion_fn"       # 遮挡
    CROWDING_FN = "crowding_fn"         # 目标过密
    BLUR_FN = "blur_fn"                 # 模糊
    LOW_CONTRAST_FN = "low_contrast_fn" # 低对比度
    OTHER_FN = "other_fn"               # 其他漏检

    # FP 错误类型
    BACKGROUND_FP = "background_fp"     # 背景误检
    CLUSTER_FP = "cluster_fp"           # 重复检测
    HIGH_CONF_FP = "high_conf_fp"       # 高置信度误检

    ALL_FN = [SCALE_FN, BOUNDARY_FN, OCCLUSION_FN, CROWDING_FN, BLUR_FN, LOW_CONTRAST_FN, OTHER_FN]
    ALL_FP = [BACKGROUND_FP, CLUSTER_FP, HIGH_CONF_FP]


# =========================
# 基础工具
# =========================

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


# =========================
# 图像特征分析器
# =========================

class ImageFeatureAnalyzer:
    """分析图像局部特征，用于错误分类"""

    @staticmethod
    def compute_blur_score(img, box_px):
        """计算目标区域的模糊程度（Laplacian 方差）"""
        x1, y1, x2, y2 = map(int, box_px)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    @staticmethod
    def compute_contrast_score(img, box_px):
        """计算目标区域的对比度（标准差）"""
        x1, y1, x2, y2 = map(int, box_px)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        return float(np.std(gray))

    @staticmethod
    def compute_boundary_ratio(box_px, img_w, img_h):
        """计算目标超出图像边界的比例"""
        x1, y1, x2, y2 = box_px
        # 裁剪到图像范围
        cx1 = max(0, x1)
        cy1 = max(0, y1)
        cx2 = min(img_w, x2)
        cy2 = min(img_h, y2)
        if cx2 <= cx1 or cy2 <= cy1:
            return 1.0
        clipped_area = (cx2 - cx1) * (cy2 - cy1)
        full_area = max(1.0, (x2 - x1) * (y2 - y1))
        return 1.0 - clipped_area / full_area

    @staticmethod
    def compute_density(gt_boxes, target_box, radius_ratio=3.0):
        """计算目标周围的密度（半径内的其他目标数）"""
        tx1, ty1, tx2, ty2 = target_box
        tcx = (tx1 + tx2) / 2
        tcy = (ty1 + ty2) / 2
        t_size = max(tx2 - tx1, ty2 - ty1)
        radius = t_size * radius_ratio

        count = 0
        for other_box in gt_boxes:
            ox1, oy1, ox2, oy2 = other_box
            ocx = (ox1 + ox2) / 2
            ocy = (oy1 + oy2) / 2
            dist = np.sqrt((tcx - ocx) ** 2 + (tcy - ocy) ** 2)
            if dist < radius:
                count += 1
        return count - 1  # 减去自身


# =========================
# 错误诊断器
# =========================

class ErrorDiagnoser:
    def __init__(self, imgsz=720, conf=0.25, iou_thr=0.5):
        self.imgsz = imgsz
        self.conf = conf
        self.iou_thr = iou_thr
        self.analyzer = ImageFeatureAnalyzer()

    def diagnose_image(self, img, gt_boxes_raw, pred_boxes, img_w, img_h):
        """对单张图片进行错误诊断"""
        img_area = img_w * img_h

        # 转换 GT 为像素坐标
        gt_boxes = []
        for cls, cx, cy, bw, bh in gt_boxes_raw:
            px = xywh2xyxy_px([cx, cy, bw, bh], img_w, img_h)
            area_px = bw * img_w * bh * img_h
            side = np.sqrt(area_px)
            gt_boxes.append({
                "cls": cls, "px": px, "area_px": area_px, "side": side,
                "cx": cx, "cy": cy, "bw": bw, "bh": bh,
            })

        # 匹配
        gt_matched = [False] * len(gt_boxes)
        pred_matched = [False] * len(pred_boxes)

        pred_order = sorted(range(len(pred_boxes)),
                            key=lambda i: pred_boxes[i]["conf"], reverse=True)

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
            if best_iou >= self.iou_thr and best_gi >= 0:
                if pred_boxes[pi]["cls"] == gt_boxes[best_gi]["cls"]:
                    gt_matched[best_gi] = True
                    pred_matched[pi] = True

        # 诊断 FN
        fn_results = []
        for gi, gt in enumerate(gt_boxes):
            if not gt_matched[gi]:
                error_type, secondary_type = self._classify_fn(img, gt, gt_boxes, img_w, img_h, img_area)
                result = {
                    "gt_index": gi,
                    "class": gt["cls"],
                    "box_px": gt["px"],
                    "area_px": gt["area_px"],
                    "side": gt["side"],
                    "error_type": error_type,
                }
                if secondary_type:
                    result["secondary_type"] = secondary_type
                fn_results.append(result)

        # 诊断 FP
        fp_results = []
        for pi, pred_b in enumerate(pred_boxes):
            if not pred_matched[pi]:
                error_type = self._classify_fp(pred_b, gt_boxes)
                fp_results.append({
                    "pred_index": pi,
                    "class": pred_b["cls"],
                    "box_px": pred_b["px"],
                    "confidence": pred_b["conf"],
                    "error_type": error_type,
                })

        return fn_results, fp_results

    def _classify_fn(self, img, gt, all_gt, img_w, img_h, img_area):
        """对单个 FN 进行错误分类，返回 (primary_type, secondary_type)"""
        px = gt["px"]
        area_px = gt["area_px"]
        side = gt["side"]

        # 按优先级评估所有条件
        conditions = []

        # 1. Scale FN: 目标太小
        if side < 32:
            conditions.append(ErrorType.SCALE_FN)

        # 2. Boundary FN: 目标在边界
        boundary_ratio = self.analyzer.compute_boundary_ratio(px, img_w, img_h)
        if boundary_ratio > 0.3:
            conditions.append(ErrorType.BOUNDARY_FN)

        # 3. Occlusion FN: 检查遮挡（通过与相邻目标的 IoU 判断）
        occluded = False
        for other_gt in all_gt:
            if other_gt is gt:
                continue
            iou = compute_iou(px, other_gt["px"])
            if iou > 0.3:
                occluded = True
                break
        if occluded:
            conditions.append(ErrorType.OCCLUSION_FN)

        # 4. Crowding FN: 目标周围过密
        density = self.analyzer.compute_density(
            [g["px"] for g in all_gt], px, radius_ratio=3.0)
        if density >= 3:
            conditions.append(ErrorType.CROWDING_FN)

        # 5. Blur FN: 模糊
        blur_score = self.analyzer.compute_blur_score(img, px)
        if blur_score < 50:
            conditions.append(ErrorType.BLUR_FN)

        # 6. Low Contrast FN: 低对比度
        contrast_score = self.analyzer.compute_contrast_score(img, px)
        if contrast_score < 20:
            conditions.append(ErrorType.LOW_CONTRAST_FN)

        # 7. 兜底
        if not conditions:
            return ErrorType.OTHER_FN, None

        primary = conditions[0]
        secondary = conditions[1] if len(conditions) > 1 else None
        return primary, secondary

    def _classify_fp(self, pred_b, gt_boxes):
        """对单个 FP 进行错误分类"""
        conf = pred_b["conf"]

        # 高置信度误检
        if conf >= 0.7:
            return ErrorType.HIGH_CONF_FP

        # 检查是否与多个预测重叠（重复检测）
        px = pred_b["px"]
        # 如果与任何 GT 都没有明显重叠，视为背景误检
        max_iou = 0
        for gt in gt_boxes:
            iou = compute_iou(px, gt["px"])
            max_iou = max(max_iou, iou)

        if max_iou < 0.1:
            return ErrorType.BACKGROUND_FP
        else:
            return ErrorType.CLUSTER_FP


# =========================
# 批量诊断
# =========================

def run_diagnosis(model, val_img_dir, val_lab_dir, class_names,
                  imgsz=720, conf=0.25, iou_thr=0.5, out_dir="diagnosis"):
    """对整个验证集进行错误诊断"""
    os.makedirs(out_dir, exist_ok=True)

    diagnoser = ErrorDiagnoser(imgsz=imgsz, conf=conf, iou_thr=iou_thr)
    analyzer = ImageFeatureAnalyzer()

    img_files = sorted([f for f in os.listdir(val_img_dir) if f.lower().endswith(IMG_SUFFIX)])

    all_fn = []
    all_fp = []
    per_image = []

    # 全局统计
    fn_type_counts = defaultdict(int)
    fp_type_counts = defaultdict(int)

    for img_name in tqdm(img_files, desc="错误诊断"):
        img_path = os.path.join(val_img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        stem = os.path.splitext(img_name)[0]
        lab_path = os.path.join(val_lab_dir, f"{stem}.txt")
        gt_raw = parse_label_file(lab_path)

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

        # 诊断
        fn_results, fp_results = diagnoser.diagnose_image(
            img, gt_raw, pred_boxes, img_w, img_h)

        # 记录
        for fn in fn_results:
            fn["image"] = img_name
            fn["class_name"] = class_names.get(fn["class"], str(fn["class"]))
            all_fn.append(fn)
            fn_type_counts[fn["error_type"]] += 1

        for fp in fp_results:
            fp["image"] = img_name
            fp["class_name"] = class_names.get(fp["class"], str(fp["class"]))
            all_fp.append(fp)
            fp_type_counts[fp["error_type"]] += 1

        per_image.append({
            "image": img_name,
            "gt_count": len(gt_raw),
            "pred_count": len(pred_boxes),
            "fn_count": len(fn_results),
            "fp_count": len(fp_results),
        })

    # 生成报告
    secondary_type_counts = defaultdict(int)
    for fn in all_fn:
        st = fn.get("secondary_type")
        if st:
            secondary_type_counts[st] += 1

    report = {
        "总图片数": len(img_files),
        "FN 总数": len(all_fn),
        "FP 总数": len(all_fp),
        "FN 类型分布": {k: v for k, v in sorted(fn_type_counts.items(), key=lambda x: -x[1])},
        "FP 类型分布": {k: v for k, v in sorted(fp_type_counts.items(), key=lambda x: -x[1])},
        "FN 类型占比": {},
        "FN 次要类型分布": {k: v for k, v in sorted(secondary_type_counts.items(), key=lambda x: -x[1])},
    }

    if len(all_fn) > 0:
        for k, v in fn_type_counts.items():
            report["FN 类型占比"][k] = f"{v / len(all_fn) * 100:.1f}%"

    # 保存
    with open(os.path.join(out_dir, "diagnosis_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # FN 详情
    fn_df = pd.DataFrame([{
        "image": fn["image"],
        "class": fn["class_name"],
        "error_type": fn["error_type"],
        "secondary_type": fn.get("secondary_type", ""),
        "area_px": fn["area_px"],
        "side": fn["side"],
        "box_x1": fn["box_px"][0],
        "box_y1": fn["box_px"][1],
        "box_x2": fn["box_px"][2],
        "box_y2": fn["box_px"][3],
    } for fn in all_fn])
    fn_df.to_csv(os.path.join(out_dir, "fn_diagnosis.csv"), index=False)

    # FP 详情
    fp_df = pd.DataFrame([{
        "image": fp["image"],
        "class": fp["class_name"],
        "error_type": fp["error_type"],
        "confidence": fp["confidence"],
        "box_x1": fp["box_px"][0],
        "box_y1": fp["box_px"][1],
        "box_x2": fp["box_px"][2],
        "box_y2": fp["box_px"][3],
    } for fp in all_fp])
    fp_df.to_csv(os.path.join(out_dir, "fp_diagnosis.csv"), index=False)

    # 逐图统计
    pd.DataFrame(per_image).to_csv(os.path.join(out_dir, "per_image_diagnosis.csv"), index=False)

    # 可视化
    _plot_diagnosis(report, fn_type_counts, fp_type_counts, out_dir)

    # 打印
    print("\n" + "=" * 60)
    print("错误诊断报告")
    print("=" * 60)
    print(f"FN 总数: {len(all_fn)}")
    print(f"FP 总数: {len(all_fp)}")
    print("\nFN 类型分布:")
    for k, v in fn_type_counts.items():
        pct = v / max(len(all_fn), 1) * 100
        print(f"  {k:25s}: {v:4d} ({pct:.1f}%)")
    print("\nFP 类型分布:")
    for k, v in fp_type_counts.items():
        pct = v / max(len(all_fp), 1) * 100
        print(f"  {k:25s}: {v:4d} ({pct:.1f}%)")
    if secondary_type_counts:
        print("\nFN 次要类型分布 (同目标满足多个条件):")
        for k, v in secondary_type_counts.items():
            print(f"  {k:25s}: {v:4d}")
    print("=" * 60)

    return report, all_fn, all_fp


def _plot_diagnosis(report, fn_counts, fp_counts, out_dir):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # FN 饼图
    if fn_counts:
        labels = list(fn_counts.keys())
        sizes = list(fn_counts.values())
        axes[0].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        axes[0].set_title(f"FN Error Types (Total: {sum(sizes)})")

    # FP 饼图
    if fp_counts:
        labels = list(fp_counts.keys())
        sizes = list(fp_counts.values())
        axes[1].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        axes[1].set_title(f"FP Error Types (Total: {sum(sizes)})")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "diagnosis_chart.png"), dpi=150)
    plt.close()
