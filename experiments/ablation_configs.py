"""
消融实验配置定义
基于 Model Feedback Driven Error Repair Framework

所有 22 组实验定义:
  B0, B1, B2   — 基线组
  A1, A2, A3, A4, A5 — 框架组件消融
  O1~O6        — 修复算子消融
  I1, I2, I3   — 迭代次数消融
"""

import os
import json
import random
import numpy as np
from collections import defaultdict
from copy import deepcopy

from error_diagnosis import ErrorType
from repair_policy import DEFAULT_POLICY, RepairPolicy

ALL_OPERATORS = [
    "zoom_crop", "sr", "zoom_sr", "context_crop",
    "copy_paste", "hard_negative", "altitude_sim", "mosaic"
]

FIXED_MAPPING = {
    ErrorType.SCALE_FN:        "zoom_sr",
    ErrorType.BOUNDARY_FN:     "context_crop",
    ErrorType.OCCLUSION_FN:    "copy_paste",
    ErrorType.CROWDING_FN:     "mosaic",
    ErrorType.BLUR_FN:         "sr",
    ErrorType.LOW_CONTRAST_FN: "sr",
    ErrorType.OTHER_FN:        "zoom_crop",
    ErrorType.BACKGROUND_FP:   "hard_negative",
    ErrorType.CLUSTER_FP:      "hard_negative",
    ErrorType.HIGH_CONF_FP:    "hard_negative",
}


class ExperimentConfig:
    """单个实验组的配置"""

    def __init__(self, name, description, category,
                 use_taxonomy=True,
                 operator_selection="weighted_random",
                 use_bandit=False,
                 enabled_operators=None,
                 num_rounds=1,
                 targeted_sampling=True,
                 strip_taxonomy=False,
                 fixed_mapping=None):
        self.name = name
        self.description = description
        self.category = category
        self.use_taxonomy = use_taxonomy
        self.operator_selection = operator_selection  # "fixed_rule" | "weighted_random" | "uniform_random" | "bandit"
        self.use_bandit = use_bandit
        self.enabled_operators = enabled_operators or ALL_OPERATORS
        self.num_rounds = num_rounds
        self.targeted_sampling = targeted_sampling
        self.strip_taxonomy = strip_taxonomy
        self.fixed_mapping = fixed_mapping or {}

    def create_policy(self):
        """根据配置创建对应的策略对象"""
        if self.operator_selection == "uniform_random":
            return UniformRandomPolicy(self.enabled_operators)
        elif self.operator_selection == "fixed_rule":
            return FixedMappingPolicy(self.fixed_mapping)
        elif self.operator_selection == "bandit":
            return BanditPolicy()
        else:  # weighted_random (default)
            return RepairPolicy()


# =========================
# 专用策略类
# =========================

class UniformRandomPolicy:
    """统一随机选择（A1, A2 用）"""

    def __init__(self, operators=None):
        self.operators = operators or ALL_OPERATORS
        self.policy = {"_uniform": {"operators": self.operators, "weights": None}}
        self.repair_history = defaultdict(lambda: defaultdict(lambda: {
            "attempts": 0, "successes": 0, "repair_rate": 0.0,
        }))

    def select_operator(self, error_type):
        return random.choice(self.operators)

    def record_repair(self, error_type, operator, success):
        h = self.repair_history[error_type][operator]
        h["attempts"] += 1
        if success:
            h["successes"] += 1
        if h["attempts"] > 0:
            h["repair_rate"] = h["successes"] / h["attempts"]

    def update_policy(self, learning_rate=0.1, min_weight=0.05):
        return {}

    def get_policy_summary(self):
        return {"_uniform": {
            "operators": self.operators,
            "weights": [1.0 / len(self.operators)] * len(self.operators),
        }}

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type": "uniform_random", "operators": self.operators}, f, ensure_ascii=False, indent=2)

    def save_history(self, path):
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

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if data.get("type") == "uniform_random":
                self.operators = data.get("operators", ALL_OPERATORS)


class FixedMappingPolicy:
    """固定 1:1 映射（A3, O1~O6 用）"""

    def __init__(self, mapping=None):
        self.mapping = mapping or FIXED_MAPPING
        self.repair_history = defaultdict(lambda: defaultdict(lambda: {
            "attempts": 0, "successes": 0, "repair_rate": 0.0,
        }))

    def select_operator(self, error_type):
        return self.mapping.get(error_type, "zoom_crop")

    def record_repair(self, error_type, operator, success):
        h = self.repair_history[error_type][operator]
        h["attempts"] += 1
        if success:
            h["successes"] += 1
        if h["attempts"] > 0:
            h["repair_rate"] = h["successes"] / h["attempts"]

    def update_policy(self, learning_rate=0.1, min_weight=0.05):
        return {}

    def get_policy_summary(self):
        summary = {}
        for etype, op in self.mapping.items():
            summary[etype] = {"operators": [op], "weights": [1.0]}
        return summary

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type": "fixed_mapping", "mapping": self.mapping}, f, ensure_ascii=False, indent=2)

    def save_history(self, path):
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

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if data.get("type") == "fixed_mapping":
                self.mapping = data.get("mapping", FIXED_MAPPING)


class BanditPolicy:
    """
    Thompson Sampling 策略更新（A5 用）
    每个 (error_type, operator) 对维护 Beta(alpha, beta) 分布
    选择时按采样值最大的算子
    """

    def __init__(self):
        self.policy = deepcopy(DEFAULT_POLICY)
        self.repair_history = defaultdict(lambda: defaultdict(lambda: {
            "attempts": 0, "successes": 0, "repair_rate": 0.0,
        }))
        # Beta 分布参数
        self.beta_params = defaultdict(lambda: defaultdict(lambda: {"alpha": 1.0, "beta": 1.0}))

    def select_operator(self, error_type):
        """Thompson Sampling 选择"""
        entry = self.policy.get(error_type)
        if not entry:
            return "zoom_crop"
        operators = entry["operators"]
        samples = []
        for op_name in operators:
            params = self.beta_params[error_type][op_name]
            samples.append(np.random.beta(params["alpha"], params["beta"]))
        best_idx = int(np.argmax(samples))
        return operators[best_idx]

    def record_repair(self, error_type, operator, success):
        h = self.repair_history[error_type][operator]
        h["attempts"] += 1
        if success:
            h["successes"] += 1
        if h["attempts"] > 0:
            h["repair_rate"] = h["successes"] / h["attempts"]

        # 更新 Beta 参数
        bp = self.beta_params[error_type][operator]
        if success:
            bp["alpha"] += 1.0
        else:
            bp["beta"] += 1.0

    def update_policy(self, learning_rate=0.1, min_weight=0.05):
        updates = {}
        for error_type in self.repair_history:
            entry = self.policy[error_type]
            operators = entry["operators"]
            old_weights = entry["weights"]
            new_weights = []
            for op_name in operators:
                params = self.beta_params[error_type][op_name]
                # Beta 均值作为经验修复率
                mean = params["alpha"] / (params["alpha"] + params["beta"])
                new_w = old_weights[operators.index(op_name)] * (1 - learning_rate) + mean * learning_rate
                new_w = max(new_w, min_weight)
                new_weights.append(new_w)
            total = sum(new_weights)
            new_weights = [w / total for w in new_weights]
            entry["weights"] = [round(w, 3) for w in new_weights]
            updates[error_type] = {
                "old_weights": old_weights,
                "new_weights": new_weights,
            }
        return updates

    def get_policy_summary(self):
        summary = {}
        for error_type, entry in self.policy.items():
            summary[error_type] = {
                "operators": entry["operators"],
                "weights": entry["weights"],
            }
        return summary

    def save(self, path):
        data = {"type": "bandit", "policy": self.policy}
        # 也保存 beta 参数
        beta_serializable = {}
        for et, ops in dict(self.beta_params).items():
            beta_serializable[et] = dict(ops)
        data["beta_params"] = beta_serializable
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def save_history(self, path):
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

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if data.get("type") == "bandit":
                self.policy = data.get("policy", deepcopy(DEFAULT_POLICY))
                bp = data.get("beta_params", {})
                for et, ops in bp.items():
                    for op_name, params in ops.items():
                        if isinstance(params, dict):
                            self.beta_params[et][op_name] = {
                                "alpha": params.get("alpha", 1.0),
                                "beta": params.get("beta", 1.0),
                            }


# =========================
# 所有实验组定义
# =========================

SHARED_CONFIG = {
    "detector": "YOLOv11-m",
    "dataset": "VisDrone val split",
    "epochs": 100,
    "imgsz": 640,
    "sample_budget_ratio": 0.35,  # 新增样本占比
}

EXPERIMENTS = {
    # ========== 基线组 ==========
    "B0": ExperimentConfig(
        name="B0",
        description="无增强基线",
        category="baseline",
        use_taxonomy=False,
        operator_selection="uniform_random",
        enabled_operators=[],
        num_rounds=0,
        targeted_sampling=False,
    ),

    "B1": ExperimentConfig(
        name="B1",
        description="随机增强基线（最关键对照组）",
        category="baseline",
        use_taxonomy=False,
        operator_selection="uniform_random",
        enabled_operators=ALL_OPERATORS,
        num_rounds=1,
        targeted_sampling=False,  # 从全部训练集随机采样
    ),

    "B2": ExperimentConfig(
        name="B2",
        description="OHEM 基线",
        category="baseline",
        use_taxonomy=False,
        operator_selection="uniform_random",
        enabled_operators=[],       # OHEM 不生成新样本
        num_rounds=0,
        targeted_sampling=True,     # 挖掘困难样本，仅记录
    ),

    # ========== 框架组件消融 ==========
    "A1": ExperimentConfig(
        name="A1",
        description="有诊断，无分类，随机修复",
        category="component",
        use_taxonomy=False,
        operator_selection="uniform_random",
        strip_taxonomy=True,        # 诊断得到 FN/FP 后剥离分类
        enabled_operators=ALL_OPERATORS,
        num_rounds=1,
        targeted_sampling=True,
    ),

    "A2": ExperimentConfig(
        name="A2",
        description="有诊断，有分类，随机修复",
        category="component",
        use_taxonomy=True,
        operator_selection="uniform_random",
        enabled_operators=ALL_OPERATORS,
        num_rounds=1,
        targeted_sampling=True,
    ),

    "A3": ExperimentConfig(
        name="A3",
        description="有诊断，有分类，固定规则映射（无迭代）",
        category="component",
        use_taxonomy=True,
        operator_selection="fixed_rule",
        fixed_mapping=FIXED_MAPPING,
        enabled_operators=ALL_OPERATORS,
        num_rounds=1,
        targeted_sampling=True,
    ),

    "A4": ExperimentConfig(
        name="A4",
        description="当前代码等价组（加权随机映射 + 单次迭代）",
        category="component",
        use_taxonomy=True,
        operator_selection="weighted_random",
        enabled_operators=ALL_OPERATORS,
        num_rounds=1,
        targeted_sampling=True,
    ),

    "A5": ExperimentConfig(
        name="A5",
        description="完整框架（Thompson Sampling 策略更新）",
        category="component",
        use_taxonomy=True,
        operator_selection="bandit",
        use_bandit=True,
        enabled_operators=ALL_OPERATORS,
        num_rounds=3,
        targeted_sampling=True,
    ),

    # ========== 修复算子消融 ==========
    "O1": ExperimentConfig(
        name="O1",
        description="只有 Zoom",
        category="operator",
        use_taxonomy=True,
        operator_selection="fixed_rule",
        fixed_mapping={
            k: "zoom_crop" for k in FIXED_MAPPING
        },
        enabled_operators=["zoom_crop"],
        num_rounds=1,
        targeted_sampling=True,
    ),

    "O2": ExperimentConfig(
        name="O2",
        description="只有 SR",
        category="operator",
        use_taxonomy=True,
        operator_selection="fixed_rule",
        fixed_mapping={
            k: "sr" for k in FIXED_MAPPING
        },
        enabled_operators=["sr"],
        num_rounds=1,
        targeted_sampling=True,
    ),

    "O3": ExperimentConfig(
        name="O3",
        description="只有 Context Crop",
        category="operator",
        use_taxonomy=True,
        operator_selection="fixed_rule",
        fixed_mapping={
            k: "context_crop" for k in FIXED_MAPPING
        },
        enabled_operators=["context_crop"],
        num_rounds=1,
        targeted_sampling=True,
    ),

    "O4": ExperimentConfig(
        name="O4",
        description="只有 Hard Negative",
        category="operator",
        use_taxonomy=True,
        operator_selection="fixed_rule",
        fixed_mapping={
            k: "hard_negative" for k in FIXED_MAPPING
        },
        enabled_operators=["hard_negative"],
        num_rounds=1,
        targeted_sampling=True,
    ),

    "O5": ExperimentConfig(
        name="O5",
        description="Zoom + SR（无其他）",
        category="operator",
        use_taxonomy=True,
        operator_selection="fixed_rule",
        fixed_mapping={
            k: "zoom_sr" if "scale" in k else "hard_negative" for k in FIXED_MAPPING
        },
        enabled_operators=["zoom_crop", "sr", "zoom_sr", "hard_negative"],
        num_rounds=1,
        targeted_sampling=True,
    ),

    "O6": ExperimentConfig(
        name="O6",
        description="完整算子库（=A3 重复确认）",
        category="operator",
        use_taxonomy=True,
        operator_selection="fixed_rule",
        fixed_mapping=FIXED_MAPPING,
        enabled_operators=ALL_OPERATORS,
        num_rounds=1,
        targeted_sampling=True,
    ),

    # ========== 迭代次数消融 ==========
    "I1": ExperimentConfig(
        name="I1",
        description="1 轮迭代",
        category="iteration",
        use_taxonomy=True,
        operator_selection="bandit",
        use_bandit=True,
        enabled_operators=ALL_OPERATORS,
        num_rounds=1,
        targeted_sampling=True,
    ),

    "I2": ExperimentConfig(
        name="I2",
        description="2 轮迭代",
        category="iteration",
        use_taxonomy=True,
        operator_selection="bandit",
        use_bandit=True,
        enabled_operators=ALL_OPERATORS,
        num_rounds=2,
        targeted_sampling=True,
    ),

    "I3": ExperimentConfig(
        name="I3",
        description="3 轮迭代（=A5）",
        category="iteration",
        use_taxonomy=True,
        operator_selection="bandit",
        use_bandit=True,
        enabled_operators=ALL_OPERATORS,
        num_rounds=3,
        targeted_sampling=True,
    ),
}


def get_experiment(name):
    """获取实验配置"""
    if name not in EXPERIMENTS:
        raise ValueError(f"未知实验: {name}。可用: {list(EXPERIMENTS.keys())}")
    return EXPERIMENTS[name]


def list_experiments(category=None):
    """列出所有实验"""
    result = []
    for name, exp in EXPERIMENTS.items():
        if category is None or exp.category == category:
            result.append((name, exp.description, exp.category))
    return result


def get_experiment_matrix():
    """返回实验矩阵（Feature × Experiment）"""
    matrix = {}
    for name, exp in EXPERIMENTS.items():
        matrix[name] = {
            "Error Diagnosis": exp.targeted_sampling or exp.num_rounds > 0,
            "Error Taxonomy": exp.use_taxonomy,
            "Targeted Mapping": exp.operator_selection in ("fixed_rule", "weighted_random", "bandit") and not exp.strip_taxonomy,
            "Policy Update": exp.use_bandit,
            "Iterations": exp.num_rounds,
        }
    return matrix
