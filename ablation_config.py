"""
消融实验配置 - Ablation Experiment Configurations
(兼容层 — 规范定义见 experiments/ablation_configs.py)

实验分组：
  基线组:   B0 (无增强), B1 (随机增强), B2 (OHEM)
  框架消融: A1→A5 (逐步增加模块)
  算子消融: O1→O6 (限制算子库)
  迭代消融: I1→I3 (不同迭代轮数)
"""

from error_diagnosis import ErrorType
from experiments.ablation_configs import (
    ALL_OPERATORS as _ALL_OPERATORS,
    FIXED_MAPPING as _FIXED_MAPPING,
)

ALL_OPERATORS = list(_ALL_OPERATORS)

# =========================
# 固定规则映射 (A3 使用) — 由 FIXED_MAPPING 自动生成
# =========================

FIXED_RULE_POLICY = {
    k: {"operators": [v], "weights": [1.0]}
    for k, v in _FIXED_MAPPING.items()
}


# =========================
# 算子消融专用策略
# =========================

def _make_uniform_policy(operators):
    """为给定算子列表创建均匀权重策略（所有 error type 共享同一算子池）"""
    w = [1.0 / len(operators)] * len(operators)
    policy = {}
    for et in ErrorType.ALL_FN + ErrorType.ALL_FP:
        policy[et] = {"operators": list(operators), "weights": list(w)}
    return policy


OPERATOR_ABLATION_POLICIES = {
    "O1": _make_uniform_policy(["zoom_crop"]),
    "O2": _make_uniform_policy(["sr"]),
    "O3": _make_uniform_policy(["context_crop"]),
    "O4": _make_uniform_policy(["hard_negative"]),
    "O5": _make_uniform_policy(["zoom_crop", "sr"]),
    "O6": _make_uniform_policy(ALL_OPERATORS),
}


# =========================
# 实验定义 (dict 兼容格式, run_ablation.py 使用)
# =========================

# mapping_type 说明:
#   "none"              → 不使用 error type 信息，算子随机选
#   "fixed_rule"        → 固定规则映射 (FIXED_RULE_POLICY)
#   "default_adaptive"  → 默认策略 + 一次性自适应权重
#   "bandit"            → 默认策略 + 多轮 Bandit 更新
#   "operator_ablation" → 使用 OPERATOR_ABLATION_POLICIES 中的策略

EXPERIMENTS = {
    "B0": {
        "name": "无增强基线",
        "group": "baseline",
        "use_diagnosis": False,
        "use_taxonomy": False,
        "mapping_type": "none",
        "policy_update": False,
        "rounds": 0,
        "operators": None,
        "random_baseline": False,
        "description": "原始训练集，不生成任何新样本。用于建立性能下限。",
    },
    "B1": {
        "name": "随机增强基线",
        "group": "baseline",
        "use_diagnosis": False,
        "use_taxonomy": False,
        "mapping_type": "none",
        "policy_update": False,
        "rounds": 1,
        "operators": ALL_OPERATORS,
        "random_baseline": False,
        "description": "从全部训练图随机采样，随机选择算子。用于排除'数据量增加'解释。",
    },
    "B2": {
        "name": "OHEM 基线",
        "group": "baseline",
        "use_diagnosis": False,
        "use_taxonomy": False,
        "mapping_type": "none",
        "policy_update": False,
        "rounds": 0,
        "operators": None,
        "random_baseline": False,
        "oheim": True,
        "description": "在线困难样本挖掘。不生成新样本，仅重加权困难样本。",
    },

    "A1": {
        "name": "有诊断-无分类-随机修复",
        "group": "ablation",
        "use_diagnosis": True,
        "use_taxonomy": False,
        "mapping_type": "none",
        "policy_update": False,
        "rounds": 1,
        "operators": ALL_OPERATORS,
        "random_baseline": False,
        "description": "从 FN/FP 样本采样，但不分类，随机选算子。验证误差引导采样的价值。",
    },
    "A2": {
        "name": "有诊断-有分类-随机修复",
        "group": "ablation",
        "use_diagnosis": True,
        "use_taxonomy": True,
        "mapping_type": "none",
        "policy_update": False,
        "rounds": 1,
        "operators": ALL_OPERATORS,
        "random_baseline": False,
        "description": "FN 分类后仍随机选算子。验证分类+定向映射的联合价值。",
    },
    "A3": {
        "name": "有诊断-有分类-固定规则映射",
        "group": "ablation",
        "use_diagnosis": True,
        "use_taxonomy": True,
        "mapping_type": "fixed_rule",
        "policy_update": False,
        "rounds": 1,
        "operators": None,
        "random_baseline": False,
        "description": "固定规则映射，单次迭代。验证定向映射的基础效果。",
    },
    "A4": {
        "name": "当前代码等价（一次性自适应权重）",
        "group": "ablation",
        "use_diagnosis": True,
        "use_taxonomy": True,
        "mapping_type": "default_adaptive",
        "policy_update": False,
        "rounds": 1,
        "operators": None,
        "random_baseline": False,
        "description": "默认策略 + 一次性自适应权重，等价于当前代码行为。",
    },
    "A5": {
        "name": "完整框架（统计反馈更新策略）",
        "group": "ablation",
        "use_diagnosis": True,
        "use_taxonomy": True,
        "mapping_type": "bandit",
        "policy_update": True,
        "rounds": 3,
        "operators": None,
        "random_baseline": False,
        "description": "完整闭环：3 轮迭代，每轮根据 RepairRate 更新策略权重。",
    },

    "O1": {
        "name": "算子消融: Zoom only",
        "group": "operator",
        "use_diagnosis": True,
        "use_taxonomy": True,
        "mapping_type": "operator_ablation",
        "ablation_key": "O1",
        "policy_update": False,
        "rounds": 1,
        "operators": ["zoom_crop"],
        "random_baseline": False,
        "description": "只用 Zoom Crop 算子。",
    },
    "O2": {
        "name": "算子消融: SR only",
        "group": "operator",
        "use_diagnosis": True,
        "use_taxonomy": True,
        "mapping_type": "operator_ablation",
        "ablation_key": "O2",
        "policy_update": False,
        "rounds": 1,
        "operators": ["sr"],
        "random_baseline": False,
        "description": "只用 SR 超分辨率算子。",
    },
    "O3": {
        "name": "算子消融: Context Crop only",
        "group": "operator",
        "use_diagnosis": True,
        "use_taxonomy": True,
        "mapping_type": "operator_ablation",
        "ablation_key": "O3",
        "policy_update": False,
        "rounds": 1,
        "operators": ["context_crop"],
        "random_baseline": False,
        "description": "只用 Context Crop 算子。",
    },
    "O4": {
        "name": "算子消融: Hard Negative only",
        "group": "operator",
        "use_diagnosis": True,
        "use_taxonomy": True,
        "mapping_type": "operator_ablation",
        "ablation_key": "O4",
        "policy_update": False,
        "rounds": 1,
        "operators": ["hard_negative"],
        "random_baseline": False,
        "description": "只用 Hard Negative 算子。",
    },
    "O5": {
        "name": "算子消融: Zoom + SR",
        "group": "operator",
        "use_diagnosis": True,
        "use_taxonomy": True,
        "mapping_type": "operator_ablation",
        "ablation_key": "O5",
        "policy_update": False,
        "rounds": 1,
        "operators": ["zoom_crop", "sr"],
        "random_baseline": False,
        "description": "Zoom + SR 组合，Scale FN 修复的最小组合。",
    },
    "O6": {
        "name": "算子消融: 完整算子库",
        "group": "operator",
        "use_diagnosis": True,
        "use_taxonomy": True,
        "mapping_type": "operator_ablation",
        "ablation_key": "O6",
        "policy_update": False,
        "rounds": 1,
        "operators": ALL_OPERATORS,
        "random_baseline": False,
        "description": "完整算子库，与 A3 对照确认。",
    },

    "I1": {
        "name": "迭代消融: 1 轮",
        "group": "iteration",
        "use_diagnosis": True,
        "use_taxonomy": True,
        "mapping_type": "bandit",
        "policy_update": True,
        "rounds": 1,
        "operators": None,
        "random_baseline": False,
        "description": "完整框架 1 轮迭代。",
    },
    "I2": {
        "name": "迭代消融: 2 轮",
        "group": "iteration",
        "use_diagnosis": True,
        "use_taxonomy": True,
        "mapping_type": "bandit",
        "policy_update": True,
        "rounds": 2,
        "operators": None,
        "random_baseline": False,
        "description": "完整框架 2 轮迭代。",
    },
    "I3": {
        "name": "迭代消融: 3 轮",
        "group": "iteration",
        "use_diagnosis": True,
        "use_taxonomy": True,
        "mapping_type": "bandit",
        "policy_update": True,
        "rounds": 3,
        "operators": None,
        "random_baseline": False,
        "description": "完整框架 3 轮迭代。",
    },
}


EXPERIMENT_GROUPS = {
    "baseline":  ["B0", "B1", "B2"],
    "ablation":  ["A1", "A2", "A3", "A4", "A5"],
    "operator":  ["O1", "O2", "O3", "O4", "O5", "O6"],
    "iteration": ["I1", "I2", "I3"],
    "all":       list(EXPERIMENTS.keys()),
    "quick":     ["B0", "B1", "A3", "A5"],
    "core":      ["B0", "B1", "A1", "A2", "A3", "A4", "A5"],
}
