# Adaptive Augmentor - Model Feedback Driven Error Repair Framework

基于模型反馈的闭环错误修复框架。不是简单的数据增强，而是：**利用模型反馈持续发现弱点，并自动生成最有价值的新训练样本。**

## 核心思想

```
传统数据增强：随机变换 → 祈祷有效
本框架：      诊断错误 → 针对修复 → 验证效果 → 更新策略
```

## 项目结构

```
adaptive-augmentor/
├── closed_loop_framework.py   # v4 主入口（10阶段闭环）
├── error_diagnosis.py         # 错误诊断 + 错误分类
├── repair_operators.py        # 修复算子库（8个算子）
├── repair_policy.py           # 错误→修复映射 + 策略更新
├── val_analyzer.py            # val 详细分析工具
├── adaptive_augmentor.py      # v3 增强（保留兼容）
├── sr/                        # 超分模块（Real-ESRGAN）
│   ├── srvgg_arch.py          # SRVGGNetCompact 架构
│   ├── upsampler.py           # 推理封装
│   └── weights/               # 预训练权重
├── configs/                   # 配置模板
└── scripts/                   # 工具脚本
```

## 安装

```bash
# 克隆项目
git clone <repo-url> adaptive-augmentor
cd adaptive-augmentor

# 依赖（已有 conda 环境则跳过）
pip install ultralytics opencv-python pandas matplotlib tqdm pyyaml

# 下载超分权重（可选，用于 SR 算子）
bash scripts/download_weights.sh
```

## 快速开始

### 1. 标准 val 分析

```bash
python val_analyzer.py \
  --data data.yaml \
  --weights best.pt \
  --imgsz 720
```

输出：
- `val_results/standard_val_results.json` — 标准 mAP/P/R
- `val_results/defect_summary.json` — 漏检/误检分析
- `val_results/dataset_features_summary.json` — 数据集特征
- `val_results/fn_samples/` — 漏检样例图片

### 2. 闭环修复（第一轮）

```bash
python closed_loop_framework.py \
  --data data.yaml \
  --weights best.pt \
  --sr_mode realesrgan \
  --imgsz 720 \
  --out repair_round1
```

输出：
- `repair_round1/diagnosis/` — 错误诊断报告
- `repair_round1/repair_samples/` — 修复样本
- `repair_round1/repair_policy.json` — 策略映射

### 3. 合并数据集 + 重新训练

```bash
# 将 repair_samples 合并到训练集
cp -r repair_round1/repair_samples/images/* /path/to/train/images/
cp -r repair_round1/repair_samples/labels/* /path/to/train/labels/

# 重新训练
yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=100 imgsz=720
```

### 4. 验证修复效果

```bash
python closed_loop_framework.py \
  --data data.yaml \
  --weights best.pt \
  --repaired_weights runs/train/weights/best.pt \
  --out repair_round1_validation
```

输出：
- 各错误类型的修复率（Repair Rate）
- 策略权重更新记录
- 与随机增强的对比

## 10 阶段流程

```
阶段1: 建立基线模型（使用任意检测器）
阶段2: 错误诊断（FN/FP 分析）
阶段3: 错误分类（7种FN + 3种FP）
阶段4: 修复算子库（8个算子）
阶段5: 错误→修复映射（策略选择）
阶段6: 样本生成（针对性修复）
阶段7: 重新训练（原始+修复数据）
阶段8: 修复验证（重新诊断）
阶段9: 计算修复收益（Repair Rate）
阶段10: 策略更新（闭环）
```

## 错误分类体系

### FN（漏检）类型

| 类型 | 含义 | 判断条件 |
|------|------|----------|
| `scale_fn` | 目标太小 | 等效边长 < 32px |
| `boundary_fn` | 目标在边界 | 超出图像边界 > 30% |
| `occlusion_fn` | 遮挡 | 与相邻目标 IoU > 0.3 |
| `crowding_fn` | 目标过密 | 半径内目标数 ≥ 3 |
| `blur_fn` | 模糊 | Laplacian 方差 < 50 |
| `low_contrast_fn` | 低对比度 | 区域标准差 < 20 |
| `other_fn` | 其他 | 以上均不满足 |

### FP（误检）类型

| 类型 | 含义 | 判断条件 |
|------|------|----------|
| `background_fp` | 背景误检 | 与任何 GT IoU < 0.1 |
| `cluster_fp` | 重复检测 | 与 GT 有部分重叠 |
| `high_conf_fp` | 高置信度误检 | 置信度 ≥ 0.7 |

## 修复算子库

| 算子 | 功能 | 适用场景 |
|------|------|----------|
| `zoom_crop` | 裁剪+放大 | 小目标 |
| `sr` | 超分辨率 | 小目标、模糊 |
| `zoom_sr` | 放大+超分 | 极小目标 |
| `context_crop` | 保留上下文 | 边界目标、遮挡 |
| `copy_paste` | 复制粘贴 | 构造遮挡场景 |
| `hard_negative` | 硬负样本 | 背景误检 |
| `altitude_sim` | 高度模拟 | 尺度变化 |
| `mosaic` | 拼接 | 密集场景 |

## 策略映射

默认映射关系（可通过闭环学习更新）：

```json
{
  "scale_fn":       {"operators": ["zoom_crop", "sr", "zoom_sr"],  "weights": [0.3, 0.3, 0.4]},
  "boundary_fn":    {"operators": ["context_crop", "altitude_sim"], "weights": [0.6, 0.4]},
  "occlusion_fn":   {"operators": ["copy_paste", "context_crop"],  "weights": [0.5, 0.5]},
  "background_fp":  {"operators": ["hard_negative"],                "weights": [1.0]}
}
```

## 双后端支持

框架自动检测 ultralytics 后端：

1. **优先 STF-YOLO**：支持自定义模块（SwinTransformer 等）
2. **降级 YOLOv8**：标准 ultralytics，自动下载权重

```bash
# 自动检测
python closed_loop_framework.py --data data.yaml --weights best.pt

# 强制 YOLOv8
python closed_loop_framework.py --data data.yaml --weights yolov8n.pt --backend yolov8

# 指定 STF-YOLO 路径
python closed_loop_framework.py --data data.yaml --weights best.pt --stf_yolo ~/PycharmProjects/STF-YOLO
```

## 超分模块

基于 Real-ESRGAN (realesr-general-x4v3)，零外部依赖：

```bash
# 下载权重
bash scripts/download_weights.sh

# 使用
python closed_loop_framework.py --data data.yaml --weights best.pt --sr_mode realesrgan
```

性能（CPU）：
- 40×40 → 160×160：~54ms
- 100×100 → 400×400：~173ms

## CLI 参数

### closed_loop_framework.py

| 参数 | 默认 | 说明 |
|------|------|------|
| `--data` | 必填 | data.yaml 路径 |
| `--weights` | 必填 | 模型权重 |
| `--out` | `repair_output` | 输出目录 |
| `--imgsz` | 720 | 图像尺寸 |
| `--conf` | 0.25 | 置信度阈值 |
| `--iou` | 0.5 | IoU 匹配阈值 |
| `--sr_mode` | none | 超分模式 (none/conservative/realesrgan) |
| `--repaired_weights` | 无 | 修复后模型（用于验证） |
| `--backend` | auto | 后端 (stf-yolo/yolov8) |

### val_analyzer.py

| 参数 | 默认 | 说明 |
|------|------|------|
| `--data` | 必填 | data.yaml 路径 |
| `--weights` | 必填 | 模型权重 |
| `--imgsz` | 720 | 图像尺寸 |
| `--conf` | 0.25 | 置信度阈值 |
| `--skip_standard_val` | false | 跳过标准 val |

## 引用

如果这个框架对你的研究有帮助，请引用：

```
Model Feedback Driven Error Repair Framework for Small Object Detection
```

## License

本项目代码仅供研究使用。
