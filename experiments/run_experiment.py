"""
消融实验运行器 — Ablation Experiment Runner

支持的实验组: B0/B1/B2, A1-A5, O1-O6, I1-I3

用法:
  # 诊断 + 生成修复样本
  python experiments/run_experiment.py --config A5 --data data.yaml \
    --weights best.pt --imgsz 640 --mode generate

  # 验证修复效果（重新训练后）
  python experiments/run_experiment.py --config A5 --data data.yaml \
    --weights best.pt --repaired_weights repaired_best.pt --mode validate

  # 完整运行（generate + 等待训练 + validate）
  python experiments/run_experiment.py --config A5 --data data.yaml \
    --weights best.pt --repaired_weights repaired_best.pt --mode full

  # 按类别批量运行
  python experiments/run_experiment.py --category component --data data.yaml \
    --weights best.pt --mode generate
"""

import os
import sys
import cv2
import json
import random
import shutil
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from error_diagnosis import ErrorType, run_diagnosis, parse_label_file
from repair_operators import create_operator, OPERATOR_REGISTRY, transform_labels_to_crop
from repair_policy import RepairValidator
from experiments.ablation_configs import (
    get_experiment, list_experiments, SHARED_CONFIG, ALL_OPERATORS, FIXED_MAPPING
)

IMGSZ_SUFFIX = (".jpg", ".jpeg", ".png", ".bmp")


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def load_yaml(path):
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def count_train_images(train_img_dir):
    if not os.path.isdir(train_img_dir):
        return 0
    return sum(1 for f in os.listdir(train_img_dir)
               if os.path.splitext(f)[1].lower() in IMGSZ_SUFFIX)


def collect_train_images(train_img_dir):
    """收集训练集所有图像路径"""
    if not os.path.isdir(train_img_dir):
        return []
    return sorted([
        f for f in os.listdir(train_img_dir)
        if os.path.splitext(f)[1].lower() in IMGSZ_SUFFIX
    ])


# =========================
# 实验样本生成
# =========================

def generate_experiment_samples(fn_list, fp_list, policy, config,
                                 train_img_dir, train_lab_dir,
                                 sr_engine, imgsz, out_dir, sample_budget):
    """
    根据实验配置生成修复/增强样本

    - targeted_sampling=True: 从 FN/FP 列表中采样
    - targeted_sampling=False: 从全部训练图像随机采样（B1 用）
    - strip_taxonomy: 生成前将 error_type 统一设为 OTHER_FN
    """
    ensure_dir(os.path.join(out_dir, "images"))
    ensure_dir(os.path.join(out_dir, "labels"))

    if config.strip_taxonomy:
        for fn in fn_list:
            fn["error_type"] = ErrorType.OTHER_FN
        for fp in fp_list:
            fp["error_type"] = ErrorType.BACKGROUND_FP

    # 准备样本源
    if config.targeted_sampling:
        all_items = []
        for fn in fn_list:
            all_items.append(("fn", fn))
        for fp in fp_list:
            all_items.append(("fp", fp))
    else:
        all_train = collect_train_images(train_img_dir)
        random.shuffle(all_train)
        all_items = []
        for img_name in all_train[:sample_budget * 3]:
            all_items.append(("random", {
                "image": img_name,
                "error_type": ErrorType.OTHER_FN,
                "box_px": None,
            }))

    if not all_items:
        print("无可生成的样本源")
        return 0, {}

    random.shuffle(all_items)
    items = all_items[:sample_budget]

    generated = 0
    failed = 0
    stats = defaultdict(int)

    desc = "生成实验样本"
    for tag, item in tqdm(items, desc=desc):
        if tag == "fn":
            error_type = item["error_type"]
        elif tag == "fp":
            error_type = item["error_type"]
        else:
            error_type = ErrorType.OTHER_FN

        operator_name = policy.select_operator(error_type)

        # 算子限制
        if operator_name not in config.enabled_operators:
            operator_name = config.enabled_operators[0] if config.enabled_operators else "zoom_crop"

        # 对于 FP，非 hard_negative 跳过
        if tag == "fp" and operator_name != "hard_negative":
            continue

        img_path = os.path.join(train_img_dir, item["image"])
        img = cv2.imread(img_path)
        if img is None:
            continue

        stem = os.path.splitext(item["image"])[0]
        lab_path = os.path.join(train_lab_dir, f"{stem}.txt")
        with open(lab_path, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        operator = create_operator(operator_name, sr_engine=sr_engine, imgsz=imgsz)

        # 确定目标框
        box_px = item.get("box_px")
        if box_px is None and tag == "random":
            h, w = img.shape[:2]
            cx = random.randint(0, w - 1)
            cy = random.randint(0, h - 1)
            box_px = [cx, cy, min(cx + 32, w), min(cy + 32, h)]

        try:
            result_img, result_labels = operator.apply(img, lines, box_px)
        except Exception:
            result_img, result_labels = None, None

        if result_img is not None and (result_labels or operator_name == "hard_negative"):
            prefix = "repair_fn" if tag in ("fn", "random") else "repair_fp"
            name = f"{prefix}_{generated:06d}"
            cv2.imwrite(os.path.join(out_dir, "images", f"{name}.jpg"), result_img)
            with open(os.path.join(out_dir, "labels", f"{name}.txt"), "w") as f:
                if result_labels:
                    f.write("\n".join(result_labels))
            generated += 1
            stats[f"{error_type}+{operator_name}"] += 1
        else:
            failed += 1

    print(f"\n实验样本生成: 成功={generated}, 失败={failed}")
    for k, v in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    return generated, stats


# =========================
# 单次实验运行
# =========================

def run_experiment_single(config, args, YOLO, sr_engine,
                           train_img, train_lab, val_img, val_lab,
                           class_names, base_out, round_num, round_budget):
    """运行单次实验（一轮诊断+生成+可选验证）"""

    round_out = os.path.join(base_out, f"round_{round_num}")
    ensure_dir(round_out)

    policy = config.create_policy()

    # 尝试加载已有策略
    policy_path = os.path.join(base_out, "policy.json")
    if os.path.exists(policy_path):
        try:
            policy.load(policy_path)
            print(f"已加载策略: {policy_path}")
        except Exception:
            pass

    # ===== 诊断 =====
    diagnosis_dir = os.path.join(round_out, "diagnosis")
    report, all_fn, all_fp = {}, [], []

    if config.num_rounds > 0 or config.targeted_sampling:
        weights = args.repaired_weights if (
            round_num > 1 and args.repaired_weights) else args.weights
        model = YOLO(weights)

        print(f"\n  [诊断] 使用模型: {weights}")
        report, all_fn, all_fp = run_diagnosis(
            model, val_img, val_lab, class_names,
            imgsz=args.imgsz, conf=args.conf, iou_thr=args.iou,
            out_dir=diagnosis_dir)

        print(f"  FN={len(all_fn)}, FP={len(all_fp)}")

        # B2 OHEM: 只诊断不生成
        if config.name == "B2":
            _save_ohem_report(all_fn, all_fp, round_out)
            return report, all_fn, all_fp, 0, {}

    # ===== 样本生成 =====
    if round_budget <= 0 or not config.enabled_operators:
        print("  [跳过] 无样本预算或无启用算子")
        return report, all_fn, all_fp, 0, {}

    repair_dir = os.path.join(round_out, "repair_samples")
    generated, gen_stats = generate_experiment_samples(
        all_fn, all_fp, policy, config,
        train_img, train_lab,
        sr_engine, args.imgsz, repair_dir, round_budget)

    # ===== 验证（仅在提供了 repaired_weights 时） =====
    fn_validation = None
    fp_validation = None

    if args.repaired_weights and round_num == config.num_rounds:
        print(f"\n  [验证] 使用修复后模型: {args.repaired_weights}")
        repaired_model = YOLO(args.repaired_weights)

        _, all_fn_after, all_fp_after = run_diagnosis(
            repaired_model, val_img, val_lab, class_names,
            imgsz=args.imgsz, conf=args.conf, iou_thr=args.iou,
            out_dir=os.path.join(round_out, "diagnosis_after"))

        # 使用原始基线 FN/FP 做 before
        baseline_results_path = os.path.join(base_out, "baseline_results.json")
        if os.path.exists(baseline_results_path):
            with open(baseline_results_path, "r") as f:
                baseline = json.load(f)
            all_fn_before = _reconstruct_fn_list(baseline.get("fn_summary", {}))
            all_fp_before = _reconstruct_fp_list(baseline.get("fp_summary", {}))
        else:
            all_fn_before = all_fn
            all_fp_before = all_fp

        fn_validation = RepairValidator.validate_fn_repair(all_fn_before, all_fn_after)
        fp_validation = RepairValidator.validate_fp_reduction(all_fp_before, all_fp_after)

        print("\n  FN 修复效果:")
        for error_type, stats in fn_validation.items():
            if stats["before"] > 0:
                print(f"    {error_type}: {stats['before']} → {stats['after']}"
                      f" (修复率: {stats['repair_rate']:.1%})")

        # 策略更新
        if config.use_bandit or config.operator_selection in ("weighted_random", "bandit"):
            for error_type, stats in fn_validation.items():
                for op_name in gen_stats:
                    if error_type in op_name:
                        success = stats["repair_rate"] > 0.1
                        policy.record_repair(error_type, op_name.split("+")[1], success)

            updates = policy.update_policy(learning_rate=0.2)
            if updates:
                print("\n  策略更新:")
                for error_type, info in updates.items():
                    print(f"    {error_type}: {info['old_weights']} → {info['new_weights']}")

        policy.save(os.path.join(base_out, "policy.json"))
        policy.save_history(os.path.join(base_out, "repair_history.json"))

    # 保存 round 结果
    round_result = {
        "round": round_num,
        "generated": generated,
        "gen_stats": dict(gen_stats),
        "fn_count": len(all_fn),
        "fp_count": len(all_fp),
        "fn_validation": fn_validation,
        "fp_validation": fp_validation,
    }
    with open(os.path.join(round_out, "round_result.json"), "w", encoding="utf-8") as f:
        json.dump(round_result, f, ensure_ascii=False, indent=2)

    return report, all_fn, all_fp, generated, gen_stats


# =========================
# 主实验流程
# =========================

def run_experiment(args, config):
    """运行一个完整的实验组"""

    print("\n" + "=" * 70)
    print(f"实验组: {config.name} — {config.description}")
    print(f"类别: {config.category}")
    print(f"算子选择: {config.operator_selection}")
    print(f"分类: {'✓' if config.use_taxonomy else '✗'}")
    print(f"轮数: {config.num_rounds}")
    print("=" * 70)

    # 初始化
    from closed_loop_framework import init_ultralytics, init_sr_engine, resolve_data_paths
    YOLO, backend = init_ultralytics(args.stf_yolo, args.backend)
    sr_engine = init_sr_engine(args.sr_mode, args.sr_weights, args.device)

    data, train_img, train_lab, val_img, val_lab = resolve_data_paths(args.data)
    class_names = data.get("names", {})
    if isinstance(class_names, list):
        class_names = {i: n for i, n in enumerate(class_names)}

    print(f"\n训练集: {train_img} ({count_train_images(train_img)} 张)")
    print(f"验证集: {val_img}")
    print(f"后端: {backend}")

    # 计算样本预算
    n_train = count_train_images(train_img)
    budget_ratio = getattr(args, 'sample_budget_ratio', 0.35)
    total_budget = max(1, int(n_train * budget_ratio))
    per_round_budget = total_budget // max(1, config.num_rounds)

    # 输出目录
    base_out = os.path.join(args.out, config.name)
    ensure_dir(base_out)

    # 保存实验配置
    exp_meta = {
        "name": config.name,
        "description": config.description,
        "category": config.category,
        "use_taxonomy": config.use_taxonomy,
        "operator_selection": config.operator_selection,
        "enabled_operators": config.enabled_operators,
        "num_rounds": config.num_rounds,
        "targeted_sampling": config.targeted_sampling,
        "sample_budget_ratio": budget_ratio,
        "total_budget": total_budget,
        "per_round_budget": per_round_budget,
        "shared": SHARED_CONFIG,
    }
    with open(os.path.join(base_out, "experiment_config.json"), "w", encoding="utf-8") as f:
        json.dump(exp_meta, f, ensure_ascii=False, indent=2)

    # B0: 仅诊断基线
    if config.name == "B0":
        print("\n[B0] 仅运行基线诊断，不生成样本")
        model = YOLO(args.weights)
        diagnosis_dir = os.path.join(base_out, "diagnosis")
        report, all_fn, all_fp = run_diagnosis(
            model, val_img, val_lab, class_names,
            imgsz=args.imgsz, conf=args.conf, iou_thr=args.iou,
            out_dir=diagnosis_dir)
        _save_baseline_results(base_out, report, all_fn, all_fp)
        _save_exp_result(base_out, config.name, report, 0, {})
        return

    # B2: OHEM 风格 — 只输出困难样本清单
    if config.name == "B2":
        print("\n[B2] OHEM 风格：挖掘困难样本，不生成新数据")
        model = YOLO(args.weights)
        diagnosis_dir = os.path.join(base_out, "diagnosis")
        report, all_fn, all_fp = run_diagnosis(
            model, val_img, val_lab, class_names,
            imgsz=args.imgsz, conf=args.conf, iou_thr=args.iou,
            out_dir=diagnosis_dir)
        _save_ohem_report(all_fn, all_fp, base_out)
        _save_baseline_results(base_out, report, all_fn, all_fp)
        _save_exp_result(base_out, config.name, report, 0, {})
        return

    # ===== 多轮迭代 =====
    if config.name == "I1":
        effective_rounds = 1
        args.repaired_weights = None
    elif config.name == "I2":
        effective_rounds = 2
    elif config.name == "I3":
        effective_rounds = 3
    else:
        effective_rounds = config.num_rounds

    for round_num in range(1, effective_rounds + 1):
        print(f"\n{'#' * 60}")
        print(f"第 {round_num}/{effective_rounds} 轮")
        print(f"样本预算: {per_round_budget}")
        print(f"{'#' * 60}")

        run_experiment_single(
            config, args, YOLO, sr_engine,
            train_img, train_lab, val_img, val_lab,
            class_names, base_out, round_num, per_round_budget)

        if round_num < effective_rounds:
            print(f"\n第 {round_num} 轮完成。请重新训练后再运行下一轮。")
            print(f"下次命令: python experiments/run_experiment.py "
                  f"--config {config.name} --data {args.data} "
                  f"--weights {args.weights} "
                  f"--repaired_weights <repaired_best.pt> "
                  f"--mode validate --out {args.out}")
            break

    print(f"\n实验组 {config.name} 完成")
    print(f"结果保存在: {base_out}/")


# =========================
# 结果保存工具
# =========================

def _save_baseline_results(base_out, report, all_fn, all_fp):
    """保存基线诊断结果"""
    baseline = {
        "report": report,
        "fn_summary": _summarize_fn_list(all_fn),
        "fp_summary": _summarize_fp_list(all_fp),
        "fn_count": len(all_fn),
        "fp_count": len(all_fp),
    }
    with open(os.path.join(base_out, "baseline_results.json"), "w", encoding="utf-8") as f:
        json.dump(baseline, f, ensure_ascii=False, indent=2)


def _save_exp_result(base_out, name, report, generated, gen_stats):
    """保存实验结果"""
    result = {
        "experiment": name,
        "diagnosis_report": report,
        "generated_samples": generated,
        "strategy_usage": dict(gen_stats),
    }
    with open(os.path.join(base_out, "experiment_result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def _save_ohem_report(all_fn, all_fp, out_dir):
    """保存 OHEM 风格困难样本报告"""
    hard_samples = []
    for fn in all_fn:
        hard_samples.append({
            "image": fn["image"],
            "type": "FN",
            "error_type": fn["error_type"],
            "class": fn.get("class_name", "?"),
            "box": fn.get("box_px", []),
        })
    for fp in all_fp:
        hard_samples.append({
            "image": fp["image"],
            "type": "FP",
            "error_type": fp["error_type"],
            "confidence": fp.get("confidence", 0),
        })

    # 按错误类型统计
    fn_types = defaultdict(int)
    fp_types = defaultdict(int)
    for fn in all_fn:
        fn_types[fn["error_type"]] += 1
    for fp in all_fp:
        fp_types[fp["error_type"]] += 1

    ohem_report = {
        "method": "OHEM-style",
        "note": "仅挖掘困难样本，不生成新数据。外接训练时建议使用样本权重。",
        "total_hard": len(hard_samples),
        "fn_by_type": dict(fn_types),
        "fp_by_type": dict(fp_types),
        "hard_sample_weight_suggestions": {
            "scale_fn": 2.0,
            "boundary_fn": 1.5,
            "occlusion_fn": 2.0,
            "crowding_fn": 1.5,
            "blur_fn": 1.5,
            "low_contrast_fn": 1.5,
        },
        "hard_samples": hard_samples[:100],  # 仅保存前 100 作为样例
    }
    with open(os.path.join(out_dir, "ohem_report.json"), "w", encoding="utf-8") as f:
        json.dump(ohem_report, f, ensure_ascii=False, indent=2)


def _summarize_fn_list(fn_list):
    """将 FN 列表压缩为 JSON 摘要"""
    summary = {}
    for fn in fn_list:
        et = fn["error_type"]
        if et not in summary:
            summary[et] = []
        summary[et].append({
            "image": fn["image"],
            "box_px": fn.get("box_px"),
            "class_name": fn.get("class_name", "?"),
        })
    return {k: len(v) for k, v in summary.items()}


def _summarize_fp_list(fp_list):
    summary = {}
    for fp in fp_list:
        et = fp["error_type"]
        if et not in summary:
            summary[et] = []
        summary[et].append({
            "image": fp["image"],
            "box_px": fp.get("box_px"),
            "confidence": fp.get("confidence", 0),
        })
    return {k: len(v) for k, v in summary.items()}


def _reconstruct_fn_list(summary):
    results = []
    for et, count in summary.items():
        for _ in range(count):
            results.append({"error_type": et})
    return results


def _reconstruct_fp_list(summary):
    results = []
    for et, count in summary.items():
        for _ in range(count):
            results.append({"error_type": et})
    return results


# =========================
# CLI
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="消融实验运行器 — Ablation Experiment Runner")

    # 实验选择
    parser.add_argument("--config", type=str, default=None,
                        help="实验组名称 (B0/B1/B2/A1/A2/A3/A4/A5/O1~O6/I1/I2/I3)")
    parser.add_argument("--category", type=str, default=None,
                        choices=["baseline", "component", "operator", "iteration"],
                        help="运行整个类别的所有实验")
    parser.add_argument("--list", action="store_true",
                        help="列出所有实验组")

    # 通用参数
    parser.add_argument("--data", type=str, help="data.yaml 路径")
    parser.add_argument("--weights", type=str, help="模型权重")
    parser.add_argument("--out", type=str, default="ablation_results", help="输出根目录")
    parser.add_argument("--imgsz", type=int, default=640, help="图像尺寸")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU 匹配阈值")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--sample_budget_ratio", type=float, default=0.35,
                        help="新增样本占比")

    # 后端
    parser.add_argument("--stf_yolo", type=str, default=None)
    parser.add_argument("--backend", type=str, default=None,
                        choices=["stf-yolo", "yolov8"])
    parser.add_argument("--sr_mode", type=str, default="none",
                        choices=["none", "conservative", "realesrgan"])
    parser.add_argument("--sr_weights", type=str, default=None)

    # 验证
    parser.add_argument("--repaired_weights", type=str, default=None,
                        help="修复后模型权重")

    # 随机种子
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.list:
        print("\n可用实验组:\n")
        print(f"{'名称':<6} {'类别':<12} {'描述'}")
        print("-" * 60)
        for name, desc, cat in list_experiments():
            print(f"{name:<6} {cat:<12} {desc}")
        return

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.category:
        configs = [c for c in EXPERIMENTS.values() if c.category == args.category]
        if not configs:
            print(f"类别 '{args.category}' 下没有实验组")
            return
        print(f"批量运行 {args.category} 类别 ({len(configs)} 组实验)")

        for config in configs:
            run_experiment(args, config)
    elif args.config:
        all_configs = {name: config for name, config in EXPERIMENTS.items()}
        if args.config not in all_configs:
            print(f"未知实验: {args.config}")
            print(f"可用: {list(all_configs.keys())}")
            return
        run_experiment(args, all_configs[args.config])
    else:
        parser.print_help()
        print("\n提示: 使用 --config 指定实验组，或 --category 批量运行")


if __name__ == "__main__":
    from experiments.ablation_configs import EXPERIMENTS
    main()
