"""
消融实验结果收集与报告生成

从各实验组输出目录读取 JSON 结果，自动生成:
  - 表1: 全局指标对比
  - 表2: FN 类型数量变化（核心证据表）
  - 表3: RepairRate 对比
  - 表4: RelativeRepairRate（相对 B1）
  - 表5: 副作用监控
  - 表6: 算子消融结果
  - 表7: 迭代消融结果

用法:
  python experiments/collect_results.py --input ablation_results --output report.md
  python experiments/collect_results.py --input ablation_results --output report.json --format json
  python experiments/collect_results.py --input ablation_results --format csv
"""

import os
import sys
import json
import glob
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from error_diagnosis import ErrorType
from experiments.ablation_configs import get_experiment, EXPERIMENTS

ALL_FN_TYPES = [
    ErrorType.SCALE_FN, ErrorType.BOUNDARY_FN, ErrorType.OCCLUSION_FN,
    ErrorType.CROWDING_FN, ErrorType.BLUR_FN, ErrorType.LOW_CONTRAST_FN,
    ErrorType.OTHER_FN,
]

ALL_FP_TYPES = [
    ErrorType.BACKGROUND_FP, ErrorType.CLUSTER_FP, ErrorType.HIGH_CONF_FP,
]

FN_TYPE_SHORT = {
    ErrorType.SCALE_FN: "Scale",
    ErrorType.BOUNDARY_FN: "Boundary",
    ErrorType.OCCLUSION_FN: "Occlusion",
    ErrorType.CROWDING_FN: "Crowding",
    ErrorType.BLUR_FN: "Blur",
    ErrorType.LOW_CONTRAST_FN: "LowCon",
    ErrorType.OTHER_FN: "Other",
}


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def scan_experiments(input_dir):
    """扫描所有实验输出目录，收集结果"""
    experiments = []
    for name in sorted(os.listdir(input_dir)):
        exp_dir = os.path.join(input_dir, name)
        if not os.path.isdir(exp_dir):
            continue
        if name not in EXPERIMENTS and not name.startswith("_"):
            continue

        exp_data = {"name": name, "dir": exp_dir}

        # 实验配置
        config = load_json(os.path.join(exp_dir, "experiment_config.json"))
        if config:
            exp_data["config"] = config

        # 基线结果（before）
        baseline = load_json(os.path.join(exp_dir, "baseline_results.json"))
        if baseline:
            exp_data["baseline"] = baseline

        # 轮次结果
        rounds = []
        for rd in sorted(glob.glob(os.path.join(exp_dir, "round_*"))):
            rr = load_json(os.path.join(rd, "round_result.json"))
            if rr:
                rounds.append(rr)
        exp_data["rounds"] = rounds

        experiments.append(exp_data)

    return experiments


def get_fn_counts(baseline):
    """从 baseline 中提取每类 FN 数量"""
    counts = {}
    if baseline and "fn_summary" in baseline:
        fn_sum = baseline["fn_summary"]
        for etype in ALL_FN_TYPES:
            counts[etype] = fn_sum.get(etype, 0)
    return counts


def get_fp_counts(baseline):
    counts = {}
    if baseline and "fp_summary" in baseline:
        fp_sum = baseline["fp_summary"]
        for etype in ALL_FP_TYPES:
            counts[etype] = fp_sum.get(etype, 0)
    return counts


def get_fn_after(rounds):
    """从最后一轮的验证结果中获取修复后 FN 数量"""
    if not rounds:
        return None
    last = rounds[-1]
    fn_val = last.get("fn_validation", {})
    if not fn_val:
        return None
    after = {}
    for etype, stats in fn_val.items():
        after[etype] = stats.get("after", stats.get("fn_count", 0))
    return after


def compute_repair_rate(before_counts, after_counts):
    """计算每类修复率"""
    rates = {}
    for etype in ALL_FN_TYPES:
        b = before_counts.get(etype, 0)
        a = after_counts.get(etype, 0) if after_counts else b
        repaired = max(0, b - a)
        rate = repaired / max(b, 1)
        rates[etype] = {"before": b, "after": a, "repaired": repaired, "rate": rate}
    return rates


def generate_table1(exps):
    """表1: 全局指标对比"""
    header = ["组", "mAP@50", "mAP@50:95", "Recall", "Precision", "F1",
              "Total FN", "Total FP", "生成样本数"]
    rows = [header]

    for exp in exps:
        baseline = exp.get("baseline", {})
        report = baseline.get("report", {}) if baseline else {}
        rounds = exp.get("rounds", [])
        generated = sum(r.get("generated", 0) for r in rounds)

        row = [
            exp["name"],
            f"{report.get('mAP50', '—')}",
            f"{report.get('mAP50_95', '—')}",
            f"{report.get('recall', '—')}",
            f"{report.get('precision', '—')}",
            f"{report.get('f1', '—')}",
            str(baseline.get("fn_count", "—")) if baseline else "—",
            str(baseline.get("fp_count", "—")) if baseline else "—",
            str(generated),
        ]
        rows.append(row)

    return rows


def generate_table2(exps):
    """表2: 各 FN 类型数量变化（核心证据表）"""
    header = ["组"] + [FN_TYPE_SHORT[e] for e in ALL_FN_TYPES] + ["Total FN"]
    rows = [header]

    for exp in exps:
        baseline = exp.get("baseline", {})
        fn_before = get_fn_counts(baseline)

        # 尝试获取 after
        after = get_fn_after(exp.get("rounds", []))
        label = exp["name"]
        if after:
            label += " (after)"

        row = [label]
        counts = after if after else fn_before
        total = 0
        for etype in ALL_FN_TYPES:
            v = counts.get(etype, 0)
            row.append(str(v))
            total += v
        row.append(str(total))
        rows.append(row)

    return rows


def generate_table3(exps):
    """表3: RepairRate 对比"""
    header = ["组"] + [f"RR({FN_TYPE_SHORT[e]})" for e in ALL_FN_TYPES] + ["RR(均值)"]
    rows = [header]

    for exp in exps:
        baseline = exp.get("baseline", {})
        fn_before = get_fn_counts(baseline)
        after = get_fn_after(exp.get("rounds", []))
        if not after:
            continue

        rates = compute_repair_rate(fn_before, after)
        row = [exp["name"]]
        rr_values = []
        for etype in ALL_FN_TYPES:
            r = rates[etype]["rate"]
            row.append(f"{r:.1%}")
            rr_values.append(r)
        avg_rr = sum(rr_values) / max(len(rr_values), 1)
        row.append(f"{avg_rr:.1%}")
        rows.append(row)

    return rows


def generate_table4(exps):
    """表4: RelativeRepairRate（相对 B1）"""
    # 找 B1 基线的修复率
    b1_rates = {}
    for exp in exps:
        if exp["name"] == "B1":
            baseline = exp.get("baseline", {})
            fn_before = get_fn_counts(baseline)
            after = get_fn_after(exp.get("rounds", []))
            if after:
                b1_rates = compute_repair_rate(fn_before, after)
            break

    header = ["组"] + [f"RRR({FN_TYPE_SHORT[e]})" for e in ALL_FN_TYPES] + ["RRR(均值)"]
    rows = [header]

    for exp in exps:
        if exp["name"] == "B1":
            continue
        baseline = exp.get("baseline", {})
        fn_before = get_fn_counts(baseline)
        after = get_fn_after(exp.get("rounds", []))
        if not after or not b1_rates:
            continue

        rates = compute_repair_rate(fn_before, after)
        row = [exp["name"]]
        rrr_values = []
        for etype in ALL_FN_TYPES:
            our_repair = rates[etype]["repaired"]
            b1_repair = b1_rates[etype]["repaired"]
            rrr = our_repair / max(b1_repair, 1) if b1_repair > 0 else float("inf")
            row.append(f"{rrr:.2f}" if rrr != float("inf") else "∞")
            if rrr != float("inf"):
                rrr_values.append(rrr)
        avg = sum(rrr_values) / max(len(rrr_values), 1) if rrr_values else 0
        row.append(f"{avg:.2f}")
        rows.append(row)

    return rows


def generate_table5(exps):
    """表5: 副作用监控 — 修复某类 FN 是否导致其他类型增加"""
    header = ["目标修复类型"] + [FN_TYPE_SHORT[e] for e in ALL_FN_TYPES]
    rows = [header]

    for exp in exps:
        if exp["name"] not in ("A3", "A4", "A5"):
            continue
        baseline = exp.get("baseline", {})
        fn_before = get_fn_counts(baseline)
        after = get_fn_after(exp.get("rounds", []))
        if not after:
            continue

        diff = {}
        for etype in ALL_FN_TYPES:
            diff[etype] = after.get(etype, 0) - fn_before.get(etype, 0)

        # 主要针对的类型（减少最多的）
        repairs = sorted(diff.items(), key=lambda x: x[1])
        for target_etype, target_diff in repairs[:3]:  # 取减少最多的 3 类
            row = [f"{exp['name']}→{FN_TYPE_SHORT[target_etype]}"]
            for etype in ALL_FN_TYPES:
                d = diff[etype]
                if etype == target_etype:
                    row.append(f"↓{abs(d)}")
                else:
                    sign = "↑" if d > 0 else ("↓" if d < 0 else "—")
                    row.append(f"{sign}{abs(d)}" if d != 0 else "—")
            rows.append(row)

    return rows


def generate_table6(exps):
    """表6: 算子消融结果"""
    header = ["组", "Scale FN↓", "Boundary FN↓", "Occlusion FN↓",
              "Crowding FN↓", "Total FN", "生成数", "主要覆盖场景"]
    rows = [header]

    for exp in exps:
        if not exp["name"].startswith("O"):
            continue
        baseline = exp.get("baseline", {})
        fn_before = get_fn_counts(baseline)
        after = get_fn_after(exp.get("rounds", []))
        generated = sum(r.get("generated", 0) for r in exp.get("rounds", []))

        config = exp.get("config", {})
        operators = config.get("enabled_operators", [])

        row = [exp["name"]]
        for etype in ALL_FN_TYPES[:4]:  # Scale, Boundary, Occlusion, Crowding
            if after:
                d = after.get(etype, 0) - fn_before.get(etype, 0)
                row.append(f"↓{abs(d)}" if d < 0 else (f"↑{d}" if d > 0 else "—"))
            else:
                row.append(str(fn_before.get(etype, 0)))
        row.append(str(baseline.get("fn_count", "—")) if baseline else "—")
        row.append(str(generated))
        row.append(", ".join(operators[:3]) if operators else "—")
        rows.append(row)

    return rows


def generate_table7(exps):
    """表7: 迭代消融结果"""
    header = ["组", "Scale FN", "Boundary FN", "Total FN",
              "RepairRate均值", "训练开销"]
    rows = [header]

    for exp in exps:
        if not exp["name"].startswith("I"):
            continue
        baseline = exp.get("baseline", {})
        fn_before = get_fn_counts(baseline)
        after = get_fn_after(exp.get("rounds", []))
        rounds = exp.get("rounds", [])
        num_rounds = len(rounds)

        rates = compute_repair_rate(fn_before, after) if after else {}
        avg_rr = sum(r["rate"] for r in rates.values()) / max(len(rates), 1) if rates else 0

        row = [exp["name"]]
        row.append(str(after.get(ErrorType.SCALE_FN, "—")) if after else str(fn_before.get(ErrorType.SCALE_FN, "—")))
        row.append(str(after.get(ErrorType.BOUNDARY_FN, "—")) if after else str(fn_before.get(ErrorType.BOUNDARY_FN, "—")))
        row.append(str(sum(after.values())) if after else str(sum(fn_before.values())))
        row.append(f"{avg_rr:.1%}")
        row.append(f"{1.0 + 0.3 * (num_rounds - 1):.1f}x")
        rows.append(row)

    return rows


def generate_experiment_matrix(exps):
    """输出实验矩阵"""
    from experiments.ablation_configs import get_experiment_matrix
    matrix = get_experiment_matrix()

    header = ["组", "Error Diagnosis", "Error Taxonomy",
              "Targeted Mapping", "Policy Update", "Iterations"]
    rows = [header]

    for name in sorted(matrix.keys(), key=lambda x: (0 if x.startswith("B") else 1 if x.startswith("A") else 2 if x.startswith("O") else 3, x)):
        info = matrix[name]
        rows.append([
            name,
            "✓" if info["Error Diagnosis"] else "✗",
            "✓" if info["Error Taxonomy"] else "✗",
            "✓" if info["Targeted Mapping"] else "✗",
            "✓" if info["Policy Update"] else "✗",
            str(info["Iterations"]),
        ])
    return rows


def format_markdown_table(rows):
    """将行列表格式化为 Markdown 表格"""
    if not rows:
        return ""
    lines = []
    header = rows[0]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for row in rows[1:]:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def format_csv(rows):
    import csv
    import io
    output = io.StringIO()
    writer = csv.writer(output)
    for row in rows:
        writer.writerow(row)
    return output.getvalue()


def generate_report(exps, output_path, fmt="md"):
    """生成完整报告"""
    sections = []

    sections.append(("# 消融实验报告\n", True))
    sections.append(("## 实验矩阵\n", True))
    sections.append((format_markdown_table(generate_experiment_matrix(exps)), False))
    sections.append(("\n## 表1: 全局指标对比\n", True))
    sections.append((format_markdown_table(generate_table1(exps)), False))
    sections.append(("\n## 表2: 各 FN 类型数量变化（核心证据表）\n", True))
    sections.append((format_markdown_table(generate_table2(exps)), False))
    sections.append(("\n## 表3: RepairRate 对比\n", True))
    sections.append((format_markdown_table(generate_table3(exps)), False))
    sections.append(("\n## 表4: RelativeRepairRate（相对 B1 随机增强基线）\n", True))
    sections.append(("> RRR > 1.0 表示定向修复优于随机增强\n", True))
    sections.append((format_markdown_table(generate_table4(exps)), False))
    sections.append(("\n## 表5: 副作用监控\n", True))
    sections.append((format_markdown_table(generate_table5(exps)), False))
    sections.append(("\n## 表6: 算子消融结果\n", True))
    sections.append((format_markdown_table(generate_table6(exps)), False))
    sections.append(("\n## 表7: 迭代消融结果\n", True))
    sections.append((format_markdown_table(generate_table7(exps)), False))

    # 汇总统计
    sections.append(("\n## 汇总统计\n", True))
    sections.append((f"- 实验组数量: {len(exps)}\n", True))
    total_generated = sum(
        sum(r.get("generated", 0) for r in exp.get("rounds", []))
        for exp in exps)
    sections.append((f"- 总生成样本数: {total_generated}\n", True))

    with open(output_path, "w", encoding="utf-8") as f:
        for content, is_header in sections:
            f.write(content)

    print(f"报告已保存到: {output_path}")


def generate_json_report(exps, output_path):
    """生成 JSON 格式报告"""
    report = {
        "matrix": generate_experiment_matrix(exps)[1:],  # 去掉 header
        "table1_global": generate_table1(exps),
        "table2_fn_counts": generate_table2(exps),
        "table3_repair_rate": generate_table3(exps),
        "table4_relative_repair_rate": generate_table4(exps),
        "table5_side_effects": generate_table5(exps),
        "table6_operator_ablation": generate_table6(exps),
        "table7_iteration_ablation": generate_table7(exps),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"JSON 报告已保存到: {output_path}")


def generate_csv_report(exps, output_dir):
    """生成 CSV 格式报告"""
    tables = {
        "matrix": generate_experiment_matrix(exps),
        "table1_global": generate_table1(exps),
        "table2_fn_counts": generate_table2(exps),
        "table3_repair_rate": generate_table3(exps),
        "table4_rrr": generate_table4(exps),
        "table5_side_effects": generate_table5(exps),
        "table6_operator": generate_table6(exps),
        "table7_iteration": generate_table7(exps),
    }
    for name, rows in tables.items():
        path = os.path.join(output_dir, f"{name}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write(format_csv(rows))
    print(f"CSV 报告已保存到: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="消融实验结果收集与报告生成")
    parser.add_argument("--input", type=str, required=True,
                        help="实验输出根目录 (ablation_results/)")
    parser.add_argument("--output", type=str, default="ablation_report.md",
                        help="输出文件路径")
    parser.add_argument("--format", type=str, default="md",
                        choices=["md", "json", "csv"],
                        help="输出格式")
    args = parser.parse_args()

    exps = scan_experiments(args.input)
    if not exps:
        print(f"未在 {args.input} 中找到实验输出")
        return

    print(f"发现 {len(exps)} 个实验组: {[e['name'] for e in exps]}")

    if args.format == "md":
        generate_report(exps, args.output, "md")
    elif args.format == "json":
        generate_json_report(exps, args.output)
    elif args.format == "csv":
        csv_dir = args.output.rsplit(".", 1)[0] if "." in args.output else args.output
        os.makedirs(csv_dir, exist_ok=True)
        generate_csv_report(exps, csv_dir)


if __name__ == "__main__":
    main()
