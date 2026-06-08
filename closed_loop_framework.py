"""
闭环错误修复框架 - Closed-Loop Error Repair Framework
Model Feedback Driven Error Repair

完整流程：
1. 建立基线模型
2. 错误诊断 (Error Diagnosis)
3. 错误分类 (Error Taxonomy)
4. 建立修复算子库 (Repair Operator Library)
5. 错误→修复映射 (Error→Repair Mapping)
6. 样本生成 (Sample Generation)
7. 重新训练 (Retraining)
8. 修复验证 (Repair Evaluation)
9. 计算修复收益 (Repair Rate)
10. 策略更新 (Policy Update) → 下一轮迭代
"""

import os
import sys
import cv2
import yaml
import json
import shutil
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# 本地模块
from error_diagnosis import ErrorDiagnoser, ErrorType, run_diagnosis, parse_label_file
from repair_operators import create_operator, OPERATOR_REGISTRY
from repair_policy import RepairPolicy, RepairValidator


# =========================
# ultralytics 双后端适配
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
            if 'ultralytics' in sys.modules:
                del sys.modules['ultralytics']
            from ultralytics import YOLO
            print(f"后端: STF-YOLO ultralytics ({stf_path})")
            return YOLO, "stf-yolo"
        except Exception as e:
            print(f"STF-YOLO ultralytics 加载失败: {e}")

    _clean_stf_path(stf_path)
    try:
        if 'ultralytics' in sys.modules:
            del sys.modules['ultralytics']
        from ultralytics import YOLO
        print("后端: 系统 ultralytics (标准 YOLOv8)")
        return YOLO, "yolov8"
    except ImportError:
        raise RuntimeError("ultralytics 未安装")


# =========================
# SR 引擎
# =========================

def init_sr_engine(sr_mode, sr_weights=None, device='cuda:0'):
    if sr_mode != "realesrgan":
        return None

    if sr_weights is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sr_weights = os.path.join(script_dir, "sr", "weights", "realesr-general-x4v3.pth")

    if not os.path.exists(sr_weights):
        print(f"SR 权重不存在: {sr_weights}，降级到 conservative")
        return "fallback"

    try:
        from sr.upsampler import RealESRGANUpsampler
        engine = RealESRGANUpsampler(model_path=sr_weights, scale=4, tile=128, device=device)
        print(f"Real-ESRGAN 加载成功")
        return engine
    except Exception as e:
        print(f"Real-ESRGAN 加载失败: {e}，降级到 conservative")
        return "fallback"


# =========================
# 数据集路径解析
# =========================

def resolve_data_paths(data_yaml_path):
    data = load_yaml(data_yaml_path)
    base_path = data.get("path", "")

    train_img = data.get("train", "")
    val_img = data.get("val", train_img)

    if base_path:
        if not os.path.isabs(train_img):
            train_img = os.path.join(base_path, train_img)
        if not os.path.isabs(val_img):
            val_img = os.path.join(base_path, val_img)

    def img_to_lab(img_path):
        lab = str(Path(img_path).parent.parent / "labels" / Path(img_path).name)
        if not os.path.isdir(lab):
            alt = img_path.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
            if os.path.isdir(alt):
                return alt
        return lab

    return data, train_img, img_to_lab(train_img), val_img, img_to_lab(val_img)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


IMG_SUFFIX = (".jpg", ".jpeg", ".png", ".bmp")


# =========================
# 第6阶段：样本生成
# =========================

def generate_repair_samples(fn_list, fp_list, policy, train_img_dir, train_lab_dir,
                            sr_engine, imgsz, out_dir):
    """根据错误诊断结果，使用对应算子生成修复样本"""

    ensure_dir(os.path.join(out_dir, "images"))
    ensure_dir(os.path.join(out_dir, "labels"))

    generated = 0
    failed = 0
    stats = defaultdict(int)

    # 处理 FN
    for fn_item in tqdm(fn_list, desc="生成 FN 修复样本"):
        error_type = fn_item["error_type"]
        operator_name = policy.select_operator(error_type)

        img_path = os.path.join(train_img_dir, fn_item["image"])
        img = cv2.imread(img_path)
        if img is None:
            continue

        stem = os.path.splitext(fn_item["image"])[0]
        lab_path = os.path.join(train_lab_dir, f"{stem}.txt")
        with open(lab_path, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        operator = create_operator(operator_name, sr_engine=sr_engine, imgsz=imgsz)

        try:
            result_img, result_labels = operator.apply(
                img, lines, fn_item["box_px"])
        except Exception:
            result_img, result_labels = None, None

        if result_img is not None and result_labels:
            name = f"repair_fn_{generated:06d}"
            cv2.imwrite(os.path.join(out_dir, "images", f"{name}.jpg"), result_img)
            with open(os.path.join(out_dir, "labels", f"{name}.txt"), "w") as f:
                f.write("\n".join(result_labels))
            generated += 1
            stats[f"{error_type}+{operator_name}"] += 1
        else:
            failed += 1

    # 处理 FP
    for fp_item in tqdm(fp_list, desc="生成 FP 修复样本"):
        error_type = fp_item["error_type"]
        operator_name = policy.select_operator(error_type)

        img_path = os.path.join(train_img_dir, fp_item["image"])
        img = cv2.imread(img_path)
        if img is None:
            continue

        stem = os.path.splitext(fp_item["image"])[0]
        lab_path = os.path.join(train_lab_dir, f"{stem}.txt")
        with open(lab_path, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        operator = create_operator(operator_name, sr_engine=sr_engine, imgsz=imgsz)

        try:
            result_img, result_labels = operator.apply(
                img, lines, fp_item["box_px"])
        except Exception:
            result_img, result_labels = None, None

        if result_img is not None:
            name = f"repair_fp_{generated:06d}"
            cv2.imwrite(os.path.join(out_dir, "images", f"{name}.jpg"), result_img)
            with open(os.path.join(out_dir, "labels", f"{name}.txt"), "w") as f:
                if result_labels:
                    f.write("\n".join(result_labels))
            generated += 1
            stats[f"{error_type}+{operator_name}"] += 1
        else:
            failed += 1

    print(f"\n修复样本生成完成: 成功={generated}, 失败={failed}")
    print("各策略使用次数:")
    for k, v in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    return generated, stats


def generate_random_baseline(fn_list, fp_list, train_img_dir, train_lab_dir,
                             imgsz, out_dir, target_count):
    """生成随机增强对照组样本（不使用错误类型信息，仅随机裁剪+缩放）"""
    import random

    ensure_dir(os.path.join(out_dir, "images"))
    ensure_dir(os.path.join(out_dir, "labels"))

    all_items = []
    for fn in fn_list:
        all_items.append(("fn", fn))
    for fp in fp_list:
        all_items.append(("fp", fp))

    if not all_items:
        print("无错误样本，跳过随机对照组生成")
        return 0

    random.shuffle(all_items)
    items = all_items[:target_count]

    generated = 0
    for tag, item in tqdm(items, desc="生成随机对照组样本"):
        img_path = os.path.join(train_img_dir, item["image"])
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        crop_ratio = random.uniform(0.3, 0.7)
        ch, cw = int(h * crop_ratio), int(w * crop_ratio)
        cy = random.randint(0, max(0, h - ch))
        cx = random.randint(0, max(0, w - cw))
        crop = img[cy:cy + ch, cx:cx + cw]
        result = cv2.resize(crop, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)

        stem = os.path.splitext(item["image"])[0]
        lab_path = os.path.join(train_lab_dir, f"{stem}.txt")
        with open(lab_path, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        from repair_operators import transform_labels_to_crop
        result_labels = transform_labels_to_crop(
            lines, (h, w), (cx, cy, cx + cw, cy + ch), (imgsz, imgsz), min_size_px=2)

        if not result_labels:
            continue

        name = f"random_baseline_{generated:06d}"
        cv2.imwrite(os.path.join(out_dir, "images", f"{name}.jpg"), result)
        with open(os.path.join(out_dir, "labels", f"{name}.txt"), "w") as f:
            f.write("\n".join(result_labels))
        generated += 1

    print(f"随机对照组生成完成: {generated}/{target_count}")
    return generated


# =========================
# 第7阶段：合并数据集
# =========================

def merge_datasets(original_dir, repair_dir, output_dir):
    """合并原始数据集和修复数据集"""
    ensure_dir(os.path.join(output_dir, "images"))
    ensure_dir(os.path.join(output_dir, "labels"))

    # 复制原始数据
    for subset in ["train"]:
        src_img = os.path.join(original_dir, "images", subset)
        src_lab = os.path.join(original_dir, "labels", subset)
        if os.path.isdir(src_img):
            for f in os.listdir(src_img):
                shutil.copy2(os.path.join(src_img, f), os.path.join(output_dir, "images", f))
        if os.path.isdir(src_lab):
            for f in os.listdir(src_lab):
                shutil.copy2(os.path.join(src_lab, f), os.path.join(output_dir, "labels", f))

    # 复制修复数据
    repair_img = os.path.join(repair_dir, "images")
    repair_lab = os.path.join(repair_dir, "labels")
    if os.path.isdir(repair_img):
        for f in os.listdir(repair_img):
            shutil.copy2(os.path.join(repair_img, f), os.path.join(output_dir, "images", f))
    if os.path.isdir(repair_lab):
        for f in os.listdir(repair_lab):
            shutil.copy2(os.path.join(repair_lab, f), os.path.join(output_dir, "labels", f))

    n_original = len(os.listdir(os.path.join(original_dir, "images", subset))) if os.path.isdir(src_img) else 0
    n_repair = len(os.listdir(repair_img)) if os.path.isdir(repair_img) else 0
    print(f"数据集合并完成: 原始={n_original}, 修复={n_repair}, 总计={n_original + n_repair}")


# =========================
# 主流程
# =========================

def _run_single_round(args, YOLO, sr_engine, policy,
                      train_img, train_lab, val_img, val_lab,
                      class_names, base_out, round_num):
    """单轮闭环修复流程"""

    print("\n" + "#" * 60)
    print(f"第 {round_num} 轮闭环修复")
    print("#" * 60)

    round_out = os.path.join(base_out, f"round_{round_num}")
    ensure_dir(round_out)

    # 加载模型
    weights = args.repaired_weights if (round_num > 1 and args.repaired_weights) else args.weights
    model = YOLO(weights)
    print(f"模型加载: {weights}")

    # =========================
    # 第2阶段：错误诊断
    # =========================
    print("\n" + "=" * 60)
    print("第2阶段：错误诊断 (Error Diagnosis)")
    print("=" * 60)

    diagnosis_dir = os.path.join(round_out, "diagnosis")
    report, all_fn, all_fp = run_diagnosis(
        model, val_img, val_lab, class_names,
        imgsz=args.imgsz, conf=args.conf, iou_thr=args.iou,
        out_dir=diagnosis_dir)

    # =========================
    # 第4阶段：初始化修复算子库
    # =========================
    print("\n" + "=" * 60)
    print("第4阶段：修复算子库 (Repair Operator Library)")
    print("=" * 60)
    print(f"可用算子: {list(OPERATOR_REGISTRY.keys())}")

    # =========================
    # 第5阶段：错误→修复映射
    # =========================
    print("\n" + "=" * 60)
    print("第5阶段：错误→修复映射 (Error→Repair Mapping)")
    print("=" * 60)

    policy_summary = policy.get_policy_summary()
    for error_type, info in policy_summary.items():
        print(f"  {error_type}: {info['operators']} (权重: {info['weights']})")

    policy.save(os.path.join(round_out, "repair_policy.json"))

    # =========================
    # 第6阶段：样本生成
    # =========================
    print("\n" + "=" * 60)
    print("第6阶段：样本生成 (Sample Generation)")
    print("=" * 60)

    repair_dir = os.path.join(round_out, "repair_samples")
    generated, gen_stats = generate_repair_samples(
        all_fn, all_fp, policy, train_img, train_lab,
        sr_engine, args.imgsz, repair_dir)

    # 生成随机对照组
    random_generated = 0
    if args.random_baseline:
        random_dir = os.path.join(round_out, "random_baseline_samples")
        random_generated = generate_random_baseline(
            all_fn, all_fp, train_img, train_lab,
            args.imgsz, random_dir, target_count=generated)

    # =========================
    # 第8阶段：修复验证（需要重新训练后的模型）
    # =========================
    fn_validation = None
    fp_validation = None

    if args.repaired_weights and round_num > 1:
        print("\n" + "=" * 60)
        print("第8阶段：修复验证 (Repair Evaluation)")
        print("=" * 60)

        print(f"使用修复后模型: {args.repaired_weights}")
        repaired_model = YOLO(args.repaired_weights)

        _, all_fn_after, all_fp_after = run_diagnosis(
            repaired_model, val_img, val_lab, class_names,
            imgsz=args.imgsz, conf=args.conf, iou_thr=args.iou,
            out_dir=os.path.join(round_out, "diagnosis_after"))

        # 第9阶段：计算修复收益
        print("\n" + "=" * 60)
        print("第9阶段：修复收益 (Repair Rate)")
        print("=" * 60)

        fn_validation = RepairValidator.validate_fn_repair(all_fn, all_fn_after)
        fp_validation = RepairValidator.validate_fp_reduction(all_fp, all_fp_after)

        print("\nFN 修复效果:")
        for error_type, stats in fn_validation.items():
            print(f"  {error_type}: {stats['before']} → {stats['after']} "
                  f"(修复: {stats['repaired']}, 修复率: {stats['repair_rate']:.1%})")

        print("\nFP 减少效果:")
        for error_type, stats in fp_validation.items():
            print(f"  {error_type}: {stats['before']} → {stats['after']} "
                  f"(减少: {stats['reduced']}, 减少率: {stats['reduction_rate']:.1%})")

        # 随机对照组对比
        if args.random_baseline and random_generated > 0:
            print("\n" + "-" * 40)
            print("随机增强对照组对比")
            print("-" * 40)
            random_weights_path = args.random_repaired_weights or os.path.join(round_out, "random_repaired_best.pt")
            if os.path.exists(random_weights_path):
                random_model = YOLO(random_weights_path)
                _, random_fn_after, random_fp_after = run_diagnosis(
                    random_model, val_img, val_lab, class_names,
                    imgsz=args.imgsz, conf=args.conf, iou_thr=args.iou,
                    out_dir=os.path.join(round_out, "diagnosis_random_after"))
                random_fn_val = RepairValidator.validate_fn_repair(all_fn, random_fn_after)
                comparison = RepairValidator.compare_with_random(fn_validation, random_fn_val)
                print("\nFN 修复率对比 (针对性修复 vs 随机增强):")
                for error_type, comp in comparison.items():
                    print(f"  {error_type}: 针对性={comp['repair_rate']:.1%} "
                          f"随机={comp['random_rate']:.1%} "
                          f"增益={comp['improvement']:+.1%}")
            else:
                print(f"  随机对照模型未找到: {random_weights_path}")
                print("  请用 random_baseline_samples 训练后将权重放到上述路径")

        # 第10阶段：策略更新
        print("\n" + "=" * 60)
        print("第10阶段：策略更新 (Policy Update)")
        print("=" * 60)

        for error_type, stats in fn_validation.items():
            total = sum(v for k, v in gen_stats.items() if k.startswith(error_type + "+"))
            if total == 0:
                continue
            for key, count in gen_stats.items():
                if key.startswith(error_type + "+"):
                    op_name = key.split("+", 1)[1]
                    share = count / total
                    policy.record_repair(error_type, op_name, stats["repair_rate"] * share)

        updates = policy.update_policy(learning_rate=0.2)
        for error_type, info in updates.items():
            print(f"  {error_type}: {info['old_weights']} → {info['new_weights']}")

        policy.save(os.path.join(round_out, "repair_policy_updated.json"))
        policy.save_history(os.path.join(round_out, "repair_history.json"))

    else:
        print("\n未提供修复后模型，跳过验证阶段")
        print("请完成以下步骤后重新运行（或使用 --rounds 自动进入下一轮）：")
        print("  1. 将 repair_samples 合并到训练集")
        print("  2. 重新训练模型")
        print("  3. 使用 --repaired_weights 指定新模型")

    # 生成汇总报告
    _generate_final_report(round_out, report, generated, gen_stats, args)

    return report, generated, gen_stats, fn_validation, fp_validation


def run_closed_loop(args):
    """运行完整的闭环修复流程（支持多轮迭代）"""

    # 初始化
    YOLO, backend = init_ultralytics(args.stf_yolo, args.backend)
    sr_engine = init_sr_engine(args.sr_mode, args.sr_weights, args.device)

    data, train_img, train_lab, val_img, val_lab = resolve_data_paths(args.data)
    class_names = data.get("names", {})
    if isinstance(class_names, list):
        class_names = {i: n for i, n in enumerate(class_names)}

    print(f"训练集: {train_img}")
    print(f"验证集: {val_img}")
    print(f"类别: {class_names}")
    print(f"后端: {backend}")

    # 输出目录
    base_out = args.out
    ensure_dir(base_out)

    # 加载或恢复策略
    policy = RepairPolicy()
    policy_path = os.path.join(base_out, "repair_policy_updated.json")
    if os.path.exists(policy_path):
        policy.load(policy_path)
        print(f"已加载策略: {policy_path}")

    # 多轮迭代
    total_rounds = args.rounds
    for round_num in range(1, total_rounds + 1):
        _run_single_round(
            args, YOLO, sr_engine, policy,
            train_img, train_lab, val_img, val_lab,
            class_names, base_out, round_num)

        if round_num < total_rounds:
            if not args.repaired_weights:
                print(f"\n第 {round_num} 轮完成。请重新训练后通过 --repaired_weights 指定模型以继续下一轮。")
                break
            else:
                print(f"\n第 {round_num} 轮完成，即将进入第 {round_num + 1} 轮...")

    print("\n" + "=" * 60)
    print("闭环修复流程结束")
    print("=" * 60)


def _generate_final_report(base_out, diagnosis_report, generated, gen_stats, args):
    """生成最终汇总报告"""
    summary = {
        "框架": "Model Feedback Driven Error Repair Framework",
        "配置": {
            "data": args.data,
            "weights": args.weights,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "sr_mode": args.sr_mode,
        },
        "诊断结果": diagnosis_report,
        "修复样本数": generated,
        "策略使用": dict(gen_stats),
    }

    with open(os.path.join(base_out, "framework_report.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n所有结果已保存到: {base_out}/")


# =========================
# CLI
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="Model Feedback Driven Error Repair Framework (v4)")

    # 基本参数
    parser.add_argument("--data", type=str, required=True, help="data.yaml 路径")
    parser.add_argument("--weights", type=str, required=True, help="初始模型权重")
    parser.add_argument("--out", type=str, default="repair_output", help="输出目录")
    parser.add_argument("--imgsz", type=int, default=720, help="图像尺寸")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU 匹配阈值")
    parser.add_argument("--device", type=str, default="cuda:0", help="GPU 设备")

    # 后端
    parser.add_argument("--stf_yolo", type=str, default=None, help="STF-YOLO 路径")
    parser.add_argument("--backend", type=str, default=None, choices=["stf-yolo", "yolov8"])

    # SR
    parser.add_argument("--sr_mode", type=str, default="none",
                        choices=["none", "conservative", "realesrgan"])
    parser.add_argument("--sr_weights", type=str, default=None)

    # 闭环验证
    parser.add_argument("--repaired_weights", type=str, default=None,
                        help="修复后模型权重（用于验证修复效果）")
    parser.add_argument("--rounds", type=int, default=1,
                        help="闭环迭代轮数 (默认 1)")
    parser.add_argument("--random_baseline", action="store_true",
                        help="生成随机增强对照组用于对比实验")
    parser.add_argument("--random_repaired_weights", type=str, default=None,
                        help="随机增强对照组的修复后模型权重")

    args = parser.parse_args()
    run_closed_loop(args)


if __name__ == "__main__":
    main()

    """
    用法示例：

    # 第一轮：诊断 + 生成修复样本
    python closed_loop_framework.py \\
      --data data.yaml \\
      --weights yolov8n.pt \\
      --sr_mode realesrgan \\
      --imgsz 720 \\
      --out repair_round1

    # 合并数据集并重新训练后，验证修复效果
    python closed_loop_framework.py \\
      --data data.yaml \\
      --weights yolov8n.pt \\
      --repaired_weights runs/train/weights/best.pt \\
      --sr_mode realesrgan \\
      --imgsz 720 \\
      --out repair_round1_validation
    """
