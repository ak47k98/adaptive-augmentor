"""
消融实验运行器 - Ablation Experiment Runner

用法:
  # 运行单个实验
  python run_ablation.py --data data.yaml --weights best.pt --experiment A3

  # 运行一组实验
  python run_ablation.py --data data.yaml --weights best.pt --group core

  # 运行所有实验
  python run_ablation.py --data data.yaml --weights best.pt --group all

  # 指定修复后模型（用于验证阶段）
  python run_ablation.py --data data.yaml --weights best.pt --group ablation \\
      --repaired_weights_map '{"A3":"runs/a3/best.pt","A4":"runs/a4/best.pt"}'

  # 生成对比表格
  python run_ablation.py --data data.yaml --weights best.pt --summary_only \\
      --results_dir ablation_results
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

from error_diagnosis import ErrorType, run_diagnosis
from repair_operators import create_operator, OPERATOR_REGISTRY, transform_labels_to_crop
from repair_policy import RepairPolicy, RepairValidator
from closed_loop_framework import (
    init_ultralytics, init_sr_engine, resolve_data_paths,
    generate_repair_samples, generate_random_baseline,
    merge_datasets, ensure_dir, load_yaml, _generate_final_report,
)
from ablation_config import (
    EXPERIMENTS, EXPERIMENT_GROUPS, ALL_OPERATORS,
    FIXED_RULE_POLICY, OPERATOR_ABLATION_POLICIES,
)


# =========================
# B1: 从全部训练图随机采样 + 随机算子
# =========================

def generate_b1_samples(train_img_dir, train_lab_dir, sr_engine, imgsz,
                        out_dir, target_count):
    """B1 基线：从全部训练图随机采样，随机选择算子"""

    ensure_dir(os.path.join(out_dir, "images"))
    ensure_dir(os.path.join(out_dir, "labels"))

    IMG_SUFFIX = (".jpg", ".jpeg", ".png", ".bmp")
    all_imgs = [f for f in os.listdir(train_img_dir)
                if f.lower().endswith(IMG_SUFFIX)]

    if not all_imgs:
        print("训练集为空")
        return 0, {}

    random.shuffle(all_imgs)
    selected = all_imgs[:target_count]

    generated = 0
    stats = defaultdict(int)
    available_ops = [op for op in ALL_OPERATORS if op in OPERATOR_REGISTRY]

    for fname in tqdm(selected, desc="B1 随机增强样本生成"):
        img_path = os.path.join(train_img_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        stem = os.path.splitext(fname)[0]
        lab_path = os.path.join(train_lab_dir, f"{stem}.txt")

        if not os.path.exists(lab_path):
            continue

        with open(lab_path, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        if not lines:
            continue

        # 随机选择一个包含目标的 GT box 作为修复焦点
        boxes = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                px = [
                    (cx - bw / 2) * w, (cy - bh / 2) * h,
                    (cx + bw / 2) * w, (cy + bh / 2) * h,
                ]
                boxes.append(px)

        if not boxes:
            continue

        target_box = random.choice(boxes)

        # 随机选择算子
        op_name = random.choice(available_ops)
        operator = create_operator(op_name, sr_engine=sr_engine, imgsz=imgsz)

        try:
            result_img, result_labels = operator.apply(img, lines, target_box)
        except Exception:
            result_img, result_labels = None, None

        if result_img is not None and result_labels:
            name = f"b1_{generated:06d}"
            cv2.imwrite(os.path.join(out_dir, "images", f"{name}.jpg"), result_img)
            with open(os.path.join(out_dir, "labels", f"{name}.txt"), "w") as f:
                f.write("\n".join(result_labels))
            generated += 1
            stats[op_name] += 1
        # hard_negative 可能返回无标签的纯背景图
        elif result_img is not None and op_name == "hard_negative":
            name = f"b1_{generated:06d}"
            cv2.imwrite(os.path.join(out_dir, "images", f"{name}.jpg"), result_img)
            with open(os.path.join(out_dir, "labels", f"{name}.txt"), "w") as f:
                if result_labels:
                    f.write("\n".join(result_labels))
            generated += 1
            stats[op_name] += 1

    print(f"B1 样本生成完成: {generated}/{target_count}")
    for k, v in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    return generated, dict(stats)


# =========================
# A1: 有诊断-无分类-随机修复
# =========================

def generate_a1_samples(fn_list, fp_list, train_img_dir, train_lab_dir,
                        sr_engine, imgsz, out_dir):
    """A1: 从 FN/FP 采样（误差引导），但不使用 error type，随机选算子"""

    ensure_dir(os.path.join(out_dir, "images"))
    ensure_dir(os.path.join(out_dir, "labels"))

    all_items = [(fn, "fn") for fn in fn_list] + [(fp, "fp") for fp in fp_list]
    if not all_items:
        print("无错误样本")
        return 0, {}

    generated = 0
    failed = 0
    stats = defaultdict(int)
    available_ops = [op for op in ALL_OPERATORS if op in OPERATOR_REGISTRY]

    for item, tag in tqdm(all_items, desc="A1 误差引导随机修复"):
        img_path = os.path.join(train_img_dir, item["image"])
        img = cv2.imread(img_path)
        if img is None:
            continue

        stem = os.path.splitext(item["image"])[0]
        lab_path = os.path.join(train_lab_dir, f"{stem}.txt")
        if not os.path.exists(lab_path):
            continue
        with open(lab_path, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        op_name = random.choice(available_ops)
        operator = create_operator(op_name, sr_engine=sr_engine, imgsz=imgsz)

        try:
            result_img, result_labels = operator.apply(img, lines, item["box_px"])
        except Exception:
            result_img, result_labels = None, None

        if result_img is not None and result_labels:
            name = f"a1_{generated:06d}"
            cv2.imwrite(os.path.join(out_dir, "images", f"{name}.jpg"), result_img)
            with open(os.path.join(out_dir, "labels", f"{name}.txt"), "w") as f:
                f.write("\n".join(result_labels))
            generated += 1
            stats[f"{tag}+{op_name}"] += 1
        elif result_img is not None and op_name == "hard_negative":
            name = f"a1_{generated:06d}"
            cv2.imwrite(os.path.join(out_dir, "images", f"{name}.jpg"), result_img)
            with open(os.path.join(out_dir, "labels", f"{name}.txt"), "w") as f:
                if result_labels:
                    f.write("\n".join(result_labels))
            generated += 1
            stats[f"{tag}+{op_name}"] += 1
        else:
            failed += 1

    print(f"A1 样本生成完成: 成功={generated}, 失败={failed}")
    for k, v in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    return generated, dict(stats)


# =========================
# A2: 有诊断-有分类-随机修复
# =========================

def create_uniform_policy():
    """创建所有 error type 共享全部算子的均匀策略（用于 A2）"""
    w = [1.0 / len(ALL_OPERATORS)] * len(ALL_OPERATORS)
    policy = {}
    for et in ErrorType.ALL_FN + ErrorType.ALL_FP:
        policy[et] = {"operators": list(ALL_OPERATORS), "weights": list(w)}
    return policy


# =========================
# 主运行逻辑
# =========================

def run_single_experiment(exp_id, exp_cfg, args, YOLO, sr_engine):
    """运行单个消融实验"""

    print("\n" + "=" * 70)
    print(f"实验 {exp_id}: {exp_cfg['name']}")
    print(f"  {exp_cfg['description']}")
    print("=" * 70)

    exp_out = os.path.join(args.out, exp_id)
    ensure_dir(exp_out)

    data, train_img, train_lab, val_img, val_lab = resolve_data_paths(args.data)
    class_names = data.get("names", {})
    if isinstance(class_names, list):
        class_names = {i: n for i, n in enumerate(class_names)}

    # B0: 无增强基线，不生成样本
    if exp_id == "B0":
        print("B0: 无增强基线，不生成修复样本。")
        report_path = os.path.join(exp_out, "ablation_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump({"experiment": exp_id, "name": exp_cfg["name"],
                        "generated": 0, "stats": {}}, f, ensure_ascii=False, indent=2)
        return {"experiment": exp_id, "generated": 0}

    # B2: OHEM 基线，需要外部实现
    if exp_id == "B2":
        print("B2: OHEM 基线需要在训练脚本中实现 (--bce_loss + OHEM)。")
        print("  请在 YOLO 训练时使用 OHEM 模式，然后将结果放入:", exp_out)
        report_path = os.path.join(exp_out, "ablation_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump({"experiment": exp_id, "name": exp_cfg["name"],
                        "note": "需要在训练脚本中手动实现 OHEM"}, f, ensure_ascii=False, indent=2)
        return {"experiment": exp_id, "generated": 0, "oheim": True}

    model = YOLO(args.weights)

    # ── 错误诊断 ───────────────────────────────
    all_fn, all_fp = [], []
    report = {}

    if exp_cfg["use_diagnosis"]:
        print("\n[诊断] 运行错误诊断...")
        diagnosis_dir = os.path.join(exp_out, "diagnosis")
        report, all_fn, all_fp = run_diagnosis(
            model, val_img, val_lab, class_names,
            imgsz=args.imgsz, conf=args.conf, iou_thr=args.iou,
            out_dir=diagnosis_dir)
    else:
        print("\n[诊断] 跳过错误诊断（不使用误差信息）")

    # ── 策略配置 ───────────────────────────────
    mapping = exp_cfg["mapping_type"]

    if mapping == "none":
        # A1/B1: 不使用 error type 信息
        policy = None
    elif mapping == "fixed_rule":
        # A3: 固定规则映射
        policy = RepairPolicy(policy=FIXED_RULE_POLICY)
    elif mapping == "default_adaptive":
        # A4: 默认策略（当前代码等价）
        policy = RepairPolicy()
    elif mapping == "bandit":
        # A5/I1-I3: 默认策略 + 支持更新
        policy = RepairPolicy()
    elif mapping == "operator_ablation":
        # O1-O6: 使用算子消融专用策略
        ak = exp_cfg["ablation_key"]
        policy = RepairPolicy(policy=OPERATOR_ABLATION_POLICIES[ak])
    else:
        policy = RepairPolicy()

    if policy:
        policy.save(os.path.join(exp_out, "repair_policy.json"))
        print(f"[策略] 映射类型: {mapping}")
        for et, info in policy.get_policy_summary().items():
            print(f"  {et}: {info['operators']} (权重: {info['weights']})")

    # ── 多轮迭代 ───────────────────────────────
    total_rounds = max(exp_cfg["rounds"], 1)
    all_gen_stats = {}

    for round_num in range(1, total_rounds + 1):
        if total_rounds > 1:
            print(f"\n--- 第 {round_num}/{total_rounds} 轮 ---")

        round_out = os.path.join(exp_out, f"round_{round_num}") if total_rounds > 1 else exp_out
        ensure_dir(round_out)

        # ── 样本生成 ───────────────────────────
        repair_dir = os.path.join(round_out, "repair_samples")
        generated = 0
        gen_stats = {}

        if exp_id == "B1":
            # B1: 从全部训练图随机采样
            n_target = int(len([f for f in os.listdir(train_img)
                                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))])
                         * args.budget_ratio)
            generated, gen_stats = generate_b1_samples(
                train_img, train_lab, sr_engine, args.imgsz,
                repair_dir, target_count=n_target)

        elif exp_id == "A1":
            # A1: 误差引导 + 随机算子
            generated, gen_stats = generate_a1_samples(
                all_fn, all_fp, train_img, train_lab,
                sr_engine, args.imgsz, repair_dir)

        elif exp_id == "A2":
            # A2: 有分类但随机选算子（均匀策略）
            uniform_policy = RepairPolicy(policy=create_uniform_policy())
            generated, gen_stats = generate_repair_samples(
                all_fn, all_fp, uniform_policy, train_img, train_lab,
                sr_engine, args.imgsz, repair_dir)

        else:
            # A3/A4/A5/O1-O6/I1-I3: 使用配置的策略
            if policy is None:
                policy = RepairPolicy()
            generated, gen_stats = generate_repair_samples(
                all_fn, all_fp, policy, train_img, train_lab,
                sr_engine, args.imgsz, repair_dir)

        all_gen_stats.update(gen_stats)

        # ── 策略更新（仅 bandit 类型）──────────
        if exp_cfg["policy_update"] and exp_cfg["repaired_weights"]:
            rw = exp_cfg["repaired_weights"]
            if isinstance(rw, dict):
                rw = rw.get(exp_id)
            if rw and os.path.exists(rw):
                print(f"\n[验证] 使用修复后模型: {rw}")
                repaired_model = YOLO(rw)
                _, fn_after, fp_after = run_diagnosis(
                    repaired_model, val_img, val_lab, class_names,
                    imgsz=args.imgsz, conf=args.conf, iou_thr=args.iou,
                    out_dir=os.path.join(round_out, "diagnosis_after"))

                fn_val = RepairValidator.validate_fn_repair(all_fn, fn_after)
                fp_val = RepairValidator.validate_fp_reduction(all_fp, fp_after)

                print("\nFN 修复效果:")
                for et, st in fn_val.items():
                    print(f"  {et}: {st['before']} → {st['after']} "
                          f"(修复率: {st['repair_rate']:.1%})")

                for et, st in fn_val.items():
                    total = sum(v for k, v in gen_stats.items() if k.startswith(et + "+"))
                    if total == 0:
                        continue
                    for key, count in gen_stats.items():
                        if key.startswith(et + "+"):
                            op_name = key.split("+", 1)[1]
                            share = count / total
                            policy.record_repair(et, op_name, st["repair_rate"] * share)

                updates = policy.update_policy(learning_rate=0.2)
                for et, info in updates.items():
                    print(f"  {et}: {info['old_weights']} → {info['new_weights']}")

                policy.save(os.path.join(round_out, "repair_policy_updated.json"))
                policy.save_history(os.path.join(round_out, "repair_history.json"))

                # 更新诊断结果供下一轮使用
                all_fn = fn_after
                all_fp = fp_after

    # ── 保存报告 ───────────────────────────────
    result = {
        "experiment": exp_id,
        "name": exp_cfg["name"],
        "description": exp_cfg["description"],
        "config": {
            "use_diagnosis": exp_cfg["use_diagnosis"],
            "use_taxonomy": exp_cfg["use_taxonomy"],
            "mapping_type": mapping,
            "policy_update": exp_cfg["policy_update"],
            "rounds": total_rounds,
            "operators": exp_cfg.get("operators"),
        },
        "generated": generated,
        "stats": all_gen_stats,
        "fn_count": len(all_fn),
        "fp_count": len(all_fp),
    }

    report_path = os.path.join(exp_out, "ablation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n实验 {exp_id} 完成: 生成 {generated} 样本, FN={len(all_fn)}, FP={len(all_fp)}")
    return result


# =========================
# 结果汇总
# =========================

def generate_comparison_tables(results_dir):
    """从各实验结果生成对比表格"""

    results = {}
    for exp_id in EXPERIMENTS:
        rpt_path = os.path.join(results_dir, exp_id, "ablation_report.json")
        if os.path.exists(rpt_path):
            with open(rpt_path, "r", encoding="utf-8") as f:
                results[exp_id] = json.load(f)

    if not results:
        print("未找到实验结果")
        return

    # ── 表 1: 全局指标（需要用户提供验证结果）──
    print("\n" + "=" * 80)
    print("表 1: 全局指标对比")
    print("=" * 80)
    print(f"{'组':<6} {'mAP@50':>8} {'mAP@50:95':>10} {'Recall':>8} {'Precision':>10} {'F1':>6}")
    print("-" * 52)
    for exp_id in results:
        r = results[exp_id]
        # 这些指标需要从 YOLO val 结果中获取，此处留空
        print(f"{exp_id:<6} {'—':>8} {'—':>10} {'—':>8} {'—':>10} {'—':>6}")
    print("\n  注: mAP/Recall/Precision 需要对每个实验的模型运行 val 后填入。")

    # ── 表 2: FN 类型数量（核心表）──
    print("\n" + "=" * 80)
    print("表 2: 各 FN 类型数量变化（核心表）")
    print("=" * 80)

    fn_types = ErrorType.ALL_FN
    header = f"{'组':<6}" + "".join(f"{et:>16}" for et in fn_types) + f"{'Total FN':>12}"
    print(header)
    print("-" * len(header))

    for exp_id, r in results.items():
        # 从诊断结果读取 FN 分布
        diag_path = os.path.join(results_dir, exp_id, "diagnosis", "diagnosis_report.json")
        fn_dist = {}
        total_fn = r.get("fn_count", 0)
        if os.path.exists(diag_path):
            with open(diag_path, "r", encoding="utf-8") as f:
                diag = json.load(f)
                fn_dist = diag.get("fn_by_type", {})

        row = f"{exp_id:<6}"
        for et in fn_types:
            cnt = fn_dist.get(et, "—")
            row += f"{str(cnt):>16}"
        row += f"{total_fn:>12}"
        print(row)

    # ── 表 3: 生成统计 ──
    print("\n" + "=" * 80)
    print("表 3: 样本生成统计")
    print("=" * 80)
    print(f"{'组':<6} {'名称':<30} {'生成数':>8} {'FN数':>8} {'FP数':>8}")
    print("-" * 64)
    for exp_id, r in results.items():
        name = r.get("name", "")[:28]
        gen = r.get("generated", 0)
        fn = r.get("fn_count", 0)
        fp = r.get("fp_count", 0)
        print(f"{exp_id:<6} {name:<30} {gen:>8} {fn:>8} {fp:>8}")

    # ── 表 4: 算子使用分布 ──
    print("\n" + "=" * 80)
    print("表 4: 算子使用分布")
    print("=" * 80)

    all_ops = set()
    for r in results.values():
        all_ops.update(r.get("stats", {}).keys())
    all_ops = sorted(all_ops)

    if all_ops:
        header = f"{'组':<6}" + "".join(f"{op:>18}" for op in all_ops)
        print(header)
        print("-" * len(header))
        for exp_id, r in results.items():
            stats = r.get("stats", {})
            row = f"{exp_id:<6}"
            for op in all_ops:
                cnt = stats.get(op, 0)
                row += f"{cnt:>18}"
            print(row)

    # ── 保存 JSON 汇总 ──
    summary_path = os.path.join(results_dir, "ablation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n汇总已保存: {summary_path}")


# =========================
# CLI
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="消融实验运行器 - Ablation Experiment Runner")

    # 基本参数
    parser.add_argument("--data", type=str, default=None, help="data.yaml 路径")
    parser.add_argument("--weights", type=str, default=None, help="初始模型权重")
    parser.add_argument("--out", type=str, default="ablation_results", help="输出根目录")
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

    # 实验选择
    parser.add_argument("--experiment", type=str, default=None,
                        help="运行单个实验 (如 A3, B1)")
    parser.add_argument("--group", type=str, default=None,
                        choices=list(EXPERIMENT_GROUPS.keys()),
                        help="运行一组实验 (如 baseline, ablation, all)")
    parser.add_argument("--list", action="store_true", help="列出所有实验配置")

    # 验证
    parser.add_argument("--repaired_weights", type=str, default=None,
                        help="修复后模型权重（单实验验证）")
    parser.add_argument("--repaired_weights_map", type=str, default=None,
                        help='JSON 映射: {"A3":"path/a3.pt","A4":"path/a4.pt"}')

    # 预算
    parser.add_argument("--budget_ratio", type=float, default=0.35,
                        help="新增样本占原训练集比例 (默认 0.35)")

    # 汇总
    parser.add_argument("--summary_only", action="store_true",
                        help="仅从已有结果生成对比表格")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="结果目录（用于 --summary_only）")

    args = parser.parse_args()

    # 列出实验
    if args.list:
        print("\n可用实验:")
        print("-" * 80)
        for exp_id, cfg in EXPERIMENTS.items():
            print(f"  {exp_id:<4} | {cfg['group']:<10} | {cfg['name']}")
            print(f"         {cfg['description']}")
        print("\n可用实验组:")
        for gname, gexp in EXPERIMENT_GROUPS.items():
            print(f"  {gname:<12} → {gexp}")
        return

    # 仅汇总
    if args.summary_only:
        rd = args.results_dir or args.out
        generate_comparison_tables(rd)
        return

    # 确定要运行的实验
    experiments_to_run = []
    if args.experiment:
        if args.experiment not in EXPERIMENTS:
            print(f"未知实验: {args.experiment}")
            print(f"可用实验: {list(EXPERIMENTS.keys())}")
            return
        experiments_to_run = [args.experiment]
    elif args.group:
        experiments_to_run = EXPERIMENT_GROUPS[args.group]
    else:
        print("请指定 --experiment 或 --group")
        return

    # 验证必需参数
    if not args.data or not args.weights:
        print("运行实验需要 --data 和 --weights 参数")
        return

    # 解析 repaired_weights_map
    rw_map = {}
    if args.repaired_weights_map:
        rw_map = json.loads(args.repaired_weights_map)
    if args.repaired_weights:
        # 单实验模式
        for eid in experiments_to_run:
            rw_map[eid] = args.repaired_weights

    # 初始化
    YOLO, backend = init_ultralytics(args.stf_yolo, args.backend)
    sr_engine = init_sr_engine(args.sr_mode, args.sr_weights, args.device)
    ensure_dir(args.out)

    print(f"\n将运行 {len(experiments_to_run)} 个实验: {experiments_to_run}")
    print(f"输出目录: {args.out}")
    print(f"预算比例: {args.budget_ratio}")

    # 运行实验
    all_results = []
    for exp_id in experiments_to_run:
        exp_cfg = dict(EXPERIMENTS[exp_id])

        # 注入 repaired_weights
        if exp_id in rw_map:
            exp_cfg["repaired_weights"] = rw_map[exp_id]
        else:
            exp_cfg["repaired_weights"] = None

        try:
            result = run_single_experiment(exp_id, exp_cfg, args, YOLO, sr_engine)
            all_results.append(result)
        except Exception as e:
            print(f"\n实验 {exp_id} 失败: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({"experiment": exp_id, "error": str(e)})

    # 生成汇总
    print("\n" + "=" * 70)
    print("所有实验完成，生成对比表格...")
    print("=" * 70)
    generate_comparison_tables(args.out)


if __name__ == "__main__":
    main()
