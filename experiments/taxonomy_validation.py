"""
Taxonomy 验证工具 — Step 0
消融实验前置步骤：验证自动 FN 分类规则的准确性

流程:
  1. 运行模型诊断，获取 FN 列表
  2. 随机采样 N 个 FN 样本
  3. 对每个样本: 裁剪 + 保存可视化 + 生成 CSV
  4. 人工标注后，读取 CSV 计算准确率

用法:
  # 生成待标注样本
  python experiments/taxonomy_validation.py --data data.yaml --weights best.pt \
    --out taxonomy_val --sample_size 200 --mode generate

  # 标注完成后，计算准确率
  python experiments/taxonomy_validation.py --out taxonomy_val --mode evaluate
"""

import os
import sys
import cv2
import csv
import json
import shutil
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from error_diagnosis import ErrorType, ErrorDiagnoser, xywh2xyxy_px, parse_label_file

ALL_FN_TYPES = [ErrorType.SCALE_FN, ErrorType.BOUNDARY_FN, ErrorType.OCCLUSION_FN,
                ErrorType.CROWDING_FN, ErrorType.BLUR_FN, ErrorType.LOW_CONTRAST_FN,
                ErrorType.OTHER_FN]

FN_TYPE_CN = {
    ErrorType.SCALE_FN: "目标太小",
    ErrorType.BOUNDARY_FN: "边界截断",
    ErrorType.OCCLUSION_FN: "遮挡",
    ErrorType.CROWDING_FN: "密集场景",
    ErrorType.BLUR_FN: "模糊",
    ErrorType.LOW_CONTRAST_FN: "低对比度",
    ErrorType.OTHER_FN: "其他",
}

POSSIBLE_CAUSES = [
    "Scale (目标像素不足)",
    "Boundary (边缘截断/超出)",
    "Occlusion (遮挡)",
    "Crowding (密集/重叠)",
    "Blur (运动模糊)",
    "Low Contrast (低对比度)",
    "Other (其他原因)",
    "Unclear (无法判断)",
]


def load_yaml(path):
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_data_paths(data_path):
    data = load_yaml(data_path)
    base = os.path.dirname(os.path.abspath(data_path))
    train_img = os.path.join(base, data.get("train", "images/train"))
    val_img = os.path.join(base, data.get("val", "images/val"))
    train_lab = os.path.join(base, data.get("train", "images/train")
                              .replace("images", "labels"))
    val_lab = os.path.join(base, data.get("val", "images/val")
                            .replace("images", "labels"))
    return data, train_img, train_lab, val_img, val_lab


def generate_validation_samples(args):
    """生成待人工标注的样本集"""
    from ultralytics import YOLO

    ensure_dir = lambda d: os.makedirs(d, exist_ok=True)

    data, train_img, train_lab, val_img, val_lab = resolve_data_paths(args.data)
    class_names = data.get("names", {})
    if isinstance(class_names, list):
        class_names = {i: n for i, n in enumerate(class_names)}

    model = YOLO(args.weights)
    print(f"模型: {args.weights}")

    out_dir = args.out
    ensure_dir(out_dir)
    img_out = os.path.join(out_dir, "crops")
    ensure_dir(img_out)

    # 运行诊断
    print("正在运行错误诊断...")
    from error_diagnosis import run_diagnosis
    report, all_fn, all_fp = run_diagnosis(
        model, val_img, val_lab, class_names,
        imgsz=args.imgsz, conf=args.conf, iou_thr=args.iou,
        out_dir=os.path.join(out_dir, "diagnosis"))

    print(f"\n诊断完成: FN={len(all_fn)}, FP={len(all_fp)}")
    print("\nFN 类型分布:")
    fn_by_type = defaultdict(int)
    for fn in all_fn:
        fn_by_type[fn["error_type"]] += 1
    for t, c in sorted(fn_by_type.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")

    # 分层采样
    sample_size = min(args.sample_size, len(all_fn))
    sampled = []
    # 尽量每类均匀
    per_type = max(1, sample_size // len(ALL_FN_TYPES))
    for etype in ALL_FN_TYPES:
        type_fns = [fn for fn in all_fn if fn["error_type"] == etype]
        random.shuffle(type_fns)
        sampled.extend(type_fns[:per_type])

    # 若不足，随机补充
    if len(sampled) < sample_size:
        remaining = [fn for fn in all_fn if fn not in sampled]
        random.shuffle(remaining)
        sampled.extend(remaining[:sample_size - len(sampled)])

    sampled = sampled[:sample_size]

    # 生成标注 CSV
    csv_path = os.path.join(out_dir, "annotation_sheet.csv")
    rows = []

    for idx, fn in enumerate(sampled):
        img_path = os.path.join(val_img, fn["image"])
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]

        # 裁剪 FN 区域 (带上下文)
        bx1, by1, bx2, by2 = fn["box_px"]
        pad = 20
        cx1 = max(0, int(bx1) - pad)
        cy1 = max(0, int(by1) - pad)
        cx2 = min(w, int(bx2) + pad)
        cy2 = min(h, int(by2) + pad)
        if cx2 <= cx1 or cy2 <= cy1:
            continue

        crop = img[cy1:cy2, cx1:cx2]

        # 画 bbox
        vis = crop.copy()
        cv2.rectangle(vis,
                      (int(bx1) - cx1, int(by1) - cy1),
                      (int(bx2) - cx1, int(by2) - cy1),
                      (0, 255, 0), 2)

        # 也标记周围的 GT
        stem = os.path.splitext(fn["image"])[0]
        lab_path = os.path.join(val_lab, f"{stem}.txt")
        gt_boxes = parse_label_file(lab_path)
        for cls, gcx, gcy, gbw, gbh in gt_boxes:
            gb = xywh2xyxy_px([gcx, gcy, gbw, gbh], w, h)
            if gb[0] >= cx1 and gb[1] >= cy1 and gb[2] <= cx2 and gb[3] <= cy2:
                cv2.rectangle(vis,
                              (int(gb[0]) - cx1, int(gb[1]) - cy1),
                              (int(gb[2]) - cx1, int(gb[3]) - cy1),
                              (0, 0, 255), 1)

        crop_name = f"fn_{idx:04d}.jpg"
        cv2.imwrite(os.path.join(img_out, crop_name), vis)

        area_px = (bx2 - bx1) * (by2 - by1)
        side_px = np.sqrt(area_px)

        rows.append({
            "fn_id": idx,
            "image": fn["image"],
            "crop_file": crop_name,
            "class_name": fn.get("class_name", "?"),
            "auto_label": fn["error_type"],
            "auto_label_cn": FN_TYPE_CN.get(fn["error_type"], "?"),
            "bbox_x1": round(bx1, 1),
            "bbox_y1": round(by1, 1),
            "bbox_x2": round(bx2, 1),
            "bbox_y2": round(by2, 1),
            "side_px": round(side_px, 1),
            "area_px": round(area_px, 1),
            "confidence": fn.get("confidence", 0),
            # 人工标注列
            "human_label": "",           # 逗号分隔，可多选
            "human_notes": "",
        })

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n生成 {len(rows)} 个待标注样本到: {csv_path}")
    print(f"裁剪图保存到: {img_out}/")
    print("\n请打开 CSV 文件，在 human_label 列标注真实原因（可多选，逗号分隔）:")
    for cause in POSSIBLE_CAUSES:
        print(f"  - {cause}")

    # 生成标注指南
    guide = os.path.join(out_dir, "annotation_guide.txt")
    with open(guide, "w") as f:
        f.write("FN Taxonomy 人工标注指南\n")
        f.write("=" * 50 + "\n\n")
        f.write("请对每个 FN 样本标注其真实漏检原因（可多选）。\n\n")
        f.write("标注选项:\n")
        for cause in POSSIBLE_CAUSES:
            f.write(f"  {cause}\n")
        f.write("\n标注方式: 在 CSV 的 human_label 列填写对应选项的英文关键词，\n")
        f.write("多选用逗号分隔，例如: Scale, Occlusion\n\n")
        f.write("判定标准:\n")
        f.write("  Scale:      目标框边长 < 32 像素（图片中非常小的目标）\n")
        f.write("  Boundary:   目标框有超过 20% 的部分在图像边缘之外或紧贴边缘\n")
        f.write("  Occlusion:  目标被其他物体遮挡超过 30%\n")
        f.write("  Crowding:   目标周围存在 4 个以上其他目标紧密排列\n")
        f.write("  Blur:       目标区域运动模糊/焦点模糊\n")
        f.write("  Low Contrast: 目标与背景颜色/亮度差异很小\n")
        f.write("  Other:      不属于以上任何类型\n")
        f.write("  Unclear:    裁剪区域太小，无法判断\n")
    print(f"标注指南: {guide}")

    return sampled, rows


def evaluate_labels(args):
    """评估人工标注，计算自动分类准确率"""
    csv_path = os.path.join(args.out, "annotation_sheet.csv")
    if not os.path.exists(csv_path):
        print(f"错误: 未找到标注文件 {csv_path}")
        return

    rows = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # 解析人工标注
    cause_to_type = {
        "scale": ErrorType.SCALE_FN,
        "boundary": ErrorType.BOUNDARY_FN,
        "occlusion": ErrorType.OCCLUSION_FN,
        "crowding": ErrorType.CROWDING_FN,
        "blur": ErrorType.BLUR_FN,
        "low contrast": ErrorType.LOW_CONTRAST_FN,
        "lowcontrast": ErrorType.LOW_CONTRAST_FN,
        "other": ErrorType.OTHER_FN,
        "unclear": "unclear",
    }

    per_type_stats = defaultdict(lambda: {
        "tp": 0, "fp": 0, "fn": 0,
        "auto_total": 0, "human_total": 0,
    })

    total_matches = 0
    total_labeled = 0
    unclear_count = 0

    for row in rows:
        human_str = row.get("human_label", "").strip().lower()
        if not human_str:
            continue

        total_labeled += 1
        auto_label = row["auto_label"]

        # 解析人工标注
        human_types = set()
        for part in human_str.replace("，", ",").split(","):
            part = part.strip()
            mapped = cause_to_type.get(part)
            if mapped == "unclear":
                unclear_count += 1
                break
            if mapped:
                human_types.add(mapped)

        if "unclear" in [cause_to_type.get(p.strip()) for p in human_str.replace("，", ",").split(",")]:
            unclear_count += 1
            continue

        # 统计各类别人工标注
        for ht in human_types:
            per_type_stats[ht]["human_total"] += 1

        per_type_stats[auto_label]["auto_total"] += 1

        # 检查自动分类是否正确
        if auto_label in human_types:
            per_type_stats[auto_label]["tp"] += 1
            total_matches += 1
        else:
            # FP: 自动分类了此类但人工认为不是
            per_type_stats[auto_label]["fp"] += 1
            # FN: 人工标注了此类但自动分类为其他
            for ht in human_types:
                if ht != auto_label:
                    per_type_stats[ht]["fn"] += 1

    print("\n" + "=" * 70)
    print("Taxonomy 验证结果")
    print("=" * 70)
    print(f"有效标注样本: {total_labeled}")
    print(f"无法判断 (Unclear): {unclear_count}")
    print(f"总体一致率: {total_matches}/{total_labeled - unclear_count} = "
          f"{total_matches / max(total_labeled - unclear_count, 1):.1%}")
    print()

    # 每类详细统计
    print(f"{'FN类型':<20} {'自动预测':>8} {'人工标注':>8} {'精确率':>8} {'召回率':>8} {'F1':>8} {'合格':>6}")
    print("-" * 72)

    results = {}
    for etype in ALL_FN_TYPES:
        stats = per_type_stats[etype]
        auto_t = stats["auto_total"]
        human_t = stats["human_total"]
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]

        precision = tp / max(tp + fp, 1)
        recall = tp / max(human_t, 1)
        f1 = 2 * precision * recall / max(precision + recall, 0.001)
        qualified = "✓" if f1 >= 0.70 else "✗"

        print(f"{FN_TYPE_CN.get(etype, etype):<20} "
              f"{auto_t:>8} {human_t:>8} "
              f"{precision:>7.1%} {recall:>7.1%} {f1:>7.1%} {qualified:>6}")

        results[etype] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "qualified": f1 >= 0.70,
            "auto_total": auto_t,
            "human_total": human_t,
        }

    # 保存结果
    result_path = os.path.join(args.out, "taxonomy_validation.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_labeled": total_labeled,
            "unclear": unclear_count,
            "overall_accuracy": round(total_matches / max(total_labeled - unclear_count, 1), 4),
            "per_type": results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n验证结果已保存到: {result_path}")

    # 检查是否达到合格标准
    failed = [e for e, r in results.items() if not r["qualified"]]
    if failed:
        print(f"\n⚠ 以下类型 F1 < 0.70，需调整判定阈值:")
        for e in failed:
            print(f"  - {FN_TYPE_CN.get(e, e)} (F1={results[e]['f1']:.1%})")
        print("\n建议: 调整 error_diagnosis.py 中 _classify_fn() 的阈值参数后重新验证")
    else:
        print("\n✓ 所有 FN 类型分类 F1 >= 0.70，可以开始消融实验")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="FN Taxonomy 人工验证工具 — 消融实验 Step 0")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["generate", "evaluate"],
                        help="generate=生成标注样本 / evaluate=评估标注结果")
    parser.add_argument("--out", type=str, default="taxonomy_val",
                        help="输出目录")
    parser.add_argument("--data", type=str, help="data.yaml 路径")
    parser.add_argument("--weights", type=str, help="模型权重")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--sample_size", type=int, default=200,
                        help="采样数")

    args = parser.parse_args()

    if args.mode == "generate":
        if not args.data or not args.weights:
            parser.error("generate 模式需要 --data 和 --weights")
        random.seed(42)
        np.random.seed(42)
        generate_validation_samples(args)
    elif args.mode == "evaluate":
        evaluate_labels(args)


if __name__ == "__main__":
    main()
