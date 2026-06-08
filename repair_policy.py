"""
修复策略模块 - Repair Policy
错误类型 → 修复算子映射 + 闭环策略更新
"""

import os
import json
import numpy as np
from collections import defaultdict
from error_diagnosis import ErrorType


# =========================
# 默认策略映射
# =========================

DEFAULT_POLICY = {
    # FN 类型 → 候选算子及其初始权重
    ErrorType.SCALE_FN: {
        "operators": ["zoom_crop", "sr", "zoom_sr"],
        "weights": [0.3, 0.3, 0.4],
    },
    ErrorType.BOUNDARY_FN: {
        "operators": ["context_crop", "altitude_sim"],
        "weights": [0.6, 0.4],
    },
    ErrorType.OCCLUSION_FN: {
        "operators": ["copy_paste", "context_crop"],
        "weights": [0.5, 0.5],
    },
    ErrorType.CROWDING_FN: {
        "operators": ["mosaic", "context_crop"],
        "weights": [0.5, 0.5],
    },
    ErrorType.BLUR_FN: {
        "operators": ["sr", "zoom_crop"],
        "weights": [0.6, 0.4],
    },
    ErrorType.LOW_CONTRAST_FN: {
        "operators": ["sr", "zoom_crop"],
        "weights": [0.5, 0.5],
    },
    ErrorType.OTHER_FN: {
        "operators": ["zoom_crop", "context_crop"],
        "weights": [0.5, 0.5],
    },

    # FP 类型 → 候选算子
    ErrorType.BACKGROUND_FP: {
        "operators": ["hard_negative"],
        "weights": [1.0],
    },
    ErrorType.CLUSTER_FP: {
        "operators": ["hard_negative"],
        "weights": [1.0],
    },
    ErrorType.HIGH_CONF_FP: {
        "operators": ["hard_negative"],
        "weights": [1.0],
    },
}


# =========================
# 修复记录
# =========================

class RepairRecord:
    """记录单次修复"""
    def __init__(self, error_type, operator, image, target_box):
        self.error_type = error_type
        self.operator = operator
        self.image = image
        self.target_box = target_box
        self.repaired = False
        self.repair_verified = False


# =========================
# 修复策略
# =========================

class RepairPolicy:
    def __init__(self, policy=None, policy_path=None):
        if policy_path and os.path.exists(policy_path):
            with open(policy_path, "r", encoding="utf-8") as f:
                self.policy = json.load(f)
        else:
            self.policy = policy or DEFAULT_POLICY.copy()

        # 修复历史记录
        self.repair_history = defaultdict(lambda: defaultdict(lambda: {
            "attempts": 0,
            "successes": 0,
            "repair_rate": 0.0,
        }))

    def select_operator(self, error_type):
        """根据错误类型选择修复算子"""
        if error_type not in self.policy:
            return "zoom_crop"  # 默认算子

        entry = self.policy[error_type]
        operators = entry["operators"]
        weights = entry["weights"]

        # 归一化权重
        total = sum(weights)
        if total <= 0:
            weights = [1.0 / len(operators)] * len(operators)
        else:
            weights = [w / total for w in weights]

        return np.random.choice(operators, p=weights)

    def record_repair(self, error_type, operator, success):
        """
        记录修复结果
        success: float in [0, 1]，连续值表示该算子的修复贡献份额
        """
        self.repair_history[error_type][operator]["attempts"] += 1
        self.repair_history[error_type][operator]["successes"] += success

        record = self.repair_history[error_type][operator]
        if record["attempts"] > 0:
            record["repair_rate"] = record["successes"] / record["attempts"]

    def update_policy(self, learning_rate=0.1, min_weight=0.05):
        """根据修复历史更新策略权重"""
        updates = {}

        for error_type, operators in self.repair_history.items():
            if error_type not in self.policy:
                continue

            entry = self.policy[error_type]
            op_names = entry["operators"]
            old_weights = entry["weights"]

            # 计算新权重：修复率高的算子获得更高权重
            repair_rates = []
            for op in op_names:
                if op in operators and operators[op]["attempts"] > 0:
                    repair_rates.append(operators[op]["repair_rate"])
                else:
                    repair_rates.append(0.5)  # 默认

            # 加权更新
            new_weights = []
            for old_w, rate in zip(old_weights, repair_rates):
                new_w = old_w * (1 - learning_rate) + rate * learning_rate
                new_w = max(new_w, min_weight)
                new_weights.append(new_w)

            # 归一化
            total = sum(new_weights)
            new_weights = [w / total for w in new_weights]

            entry["weights"] = new_weights
            updates[error_type] = {
                "operators": op_names,
                "old_weights": old_weights,
                "new_weights": new_weights,
            }

        return updates

    def get_policy_summary(self):
        """获取当前策略摘要"""
        summary = {}
        for error_type, entry in self.policy.items():
            summary[error_type] = {
                "operators": entry["operators"],
                "weights": [round(w, 3) for w in entry["weights"]],
            }
        return summary

    def save(self, path):
        """保存策略到文件"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.policy, f, ensure_ascii=False, indent=2)

    def load(self, path):
        """从文件加载策略"""
        with open(path, "r", encoding="utf-8") as f:
            self.policy = json.load(f)

    def save_history(self, path):
        """保存修复历史"""
        history = {}
        for error_type, operators in self.repair_history.items():
            history[error_type] = {}
            for op, record in operators.items():
                history[error_type][op] = {
                    "attempts": record["attempts"],
                    "successes": record["successes"],
                    "repair_rate": round(record["repair_rate"], 4),
                }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)


# =========================
# 修复验证器
# =========================

class RepairValidator:
    """验证修复是否有效"""

    @staticmethod
    def validate_fn_repair(fn_before, fn_after):
        """
        验证 FN 修复效果
        Returns: dict with repair stats per error type
        """
        before_counts = defaultdict(int)
        after_counts = defaultdict(int)

        for fn in fn_before:
            before_counts[fn["error_type"]] += 1
        for fn in fn_after:
            after_counts[fn["error_type"]] += 1

        results = {}
        for error_type in set(list(before_counts.keys()) + list(after_counts.keys())):
            b = before_counts.get(error_type, 0)
            a = after_counts.get(error_type, 0)
            repaired = max(0, b - a)
            rate = repaired / max(b, 1)
            results[error_type] = {
                "before": b,
                "after": a,
                "repaired": repaired,
                "repair_rate": round(rate, 4),
            }

        return results

    @staticmethod
    def validate_fp_reduction(fp_before, fp_after):
        """
        验证 FP 减少效果
        """
        before_counts = defaultdict(int)
        after_counts = defaultdict(int)

        for fp in fp_before:
            before_counts[fp["error_type"]] += 1
        for fp in fp_after:
            after_counts[fp["error_type"]] += 1

        results = {}
        for error_type in set(list(before_counts.keys()) + list(after_counts.keys())):
            b = before_counts.get(error_type, 0)
            a = after_counts.get(error_type, 0)
            reduced = max(0, b - a)
            rate = reduced / max(b, 1)
            results[error_type] = {
                "before": b,
                "after": a,
                "reduced": reduced,
                "reduction_rate": round(rate, 4),
            }

        return results

    @staticmethod
    def compare_with_random(repair_results, random_results):
        """
        与随机增强对比
        """
        comparison = {}
        for error_type in repair_results:
            if error_type in random_results:
                repair_rate = repair_results[error_type].get("repair_rate", 0)
                random_rate = random_results[error_type].get("repair_rate", 0)
                comparison[error_type] = {
                    "repair_rate": repair_rate,
                    "random_rate": random_rate,
                    "improvement": repair_rate - random_rate,
                }
        return comparison
