"""
修复算子库 - Repair Operator Library
每个算子接收图像+标注，输出修复后的图像+标注
"""

import os
import cv2
import random
import numpy as np
from pathlib import Path


# =========================
# 基础工具
# =========================

def xywh2xyxy_px(box, w, h):
    cx, cy, bw, bh = box
    return [
        (cx - bw / 2.0) * w,
        (cy - bh / 2.0) * h,
        (cx + bw / 2.0) * w,
        (cy + bh / 2.0) * h,
    ]


def xyxy2xywh_norm(box, w, h):
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    cx, cy = x1 + bw / 2.0, y1 + bh / 2.0
    return [cx / w, cy / h, bw / w, bh / h]


def parse_label_lines(lines):
    out = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        arr = s.split()
        if len(arr) < 5:
            continue
        cls, x, y, bw, bh = map(float, arr[:5])
        out.append((int(cls), x, y, bw, bh))
    return out


def transform_labels_to_crop(lines, img_shape, crop_box, out_size, min_size_px=3):
    """将标签投影到裁剪区域"""
    orig_h, orig_w = img_shape[:2]
    cx1, cy1, cx2, cy2 = crop_box
    cw, ch = max(1, cx2 - cx1), max(1, cy2 - cy1)
    ow, oh = out_size

    scale = min(ow / max(1, cw), oh / max(1, ch))
    nw, nh = int(round(cw * scale)), int(round(ch * scale))
    dx = (ow - nw) // 2
    dy = (oh - nh) // 2

    labels = []
    parsed = parse_label_lines(lines)

    for cls, x, y, bw, bh in parsed:
        px_box = xywh2xyxy_px([x, y, bw, bh], orig_w, orig_h)
        nx1 = max(0.0, px_box[0] - cx1)
        ny1 = max(0.0, px_box[1] - cy1)
        nx2 = min(float(cw), px_box[2] - cx1)
        ny2 = min(float(ch), px_box[3] - cy1)
        if nx2 <= nx1 or ny2 <= ny1:
            continue

        bw_after = (nx2 - nx1) * scale
        bh_after = (ny2 - ny1) * scale
        if bw_after < min_size_px or bh_after < min_size_px:
            continue

        lnx1 = nx1 * scale + dx
        lny1 = ny1 * scale + dy
        lnx2 = nx2 * scale + dx
        lny2 = ny2 * scale + dy

        norm = xyxy2xywh_norm([lnx1, lny1, lnx2, lny2], ow, oh)
        norm = [np.clip(v, 0.0, 1.0) for v in norm]
        labels.append(f"{int(cls)} {norm[0]:.6f} {norm[1]:.6f} {norm[2]:.6f} {norm[3]:.6f}")

    return labels


def letterbox_resize(img, out_size, color=(0, 0, 0)):
    ow, oh = out_size
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((oh, ow, 3), dtype=np.uint8)
    scale = min(ow / w, oh / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    canvas = np.full((oh, ow, 3), color, dtype=np.uint8)
    dx = (ow - nw) // 2
    dy = (oh - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = resized
    return canvas


# =========================
# 修复算子基类
# =========================

class RepairOperator:
    """修复算子基类"""
    name = "base"

    def __init__(self, sr_engine=None, imgsz=720):
        self.sr_engine = sr_engine
        self.imgsz = imgsz

    def apply(self, img, lines, target_box, **kwargs):
        """
        执行修复
        Args:
            img: 原图 (BGR)
            lines: 原始标签行
            target_box: 目标框 [x1, y1, x2, y2] 像素坐标
        Returns:
            (修复后图像, 修复后标签) 或 (None, None)
        """
        raise NotImplementedError


# =========================
# 算子1: Zoom Crop - 放大小目标
# =========================

class ZoomCropOperator(RepairOperator):
    """围绕目标做 crop + 放大"""
    name = "zoom_crop"

    def __init__(self, sr_engine=None, imgsz=720, scale_range=(3.0, 6.0)):
        super().__init__(sr_engine, imgsz)
        self.scale_range = scale_range

    def apply(self, img, lines, target_box, **kwargs):
        h, w = img.shape[:2]
        x1, y1, x2, y2 = target_box
        bw_t, bh_t = x2 - x1, y2 - y1
        if bw_t <= 0 or bh_t <= 0:
            return None, None

        scale = random.uniform(*self.scale_range)
        crop_w, crop_h = bw_t * scale, bh_t * scale
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        c_x1 = int(max(0, cx - crop_w / 2))
        c_y1 = int(max(0, cy - crop_h / 2))
        c_x2 = int(min(w, cx + crop_w / 2))
        c_y2 = int(min(h, cy + crop_h / 2))

        if c_x2 <= c_x1 or c_y2 <= c_y1:
            return None, None

        crop = img[c_y1:c_y2, c_x1:c_x2]
        if crop.size == 0:
            return None, None

        out_size = (self.imgsz, self.imgsz)
        out = letterbox_resize(crop, out_size)
        labels = transform_labels_to_crop(lines, img.shape, [c_x1, c_y1, c_x2, c_y2], out_size)

        if not labels:
            return None, None
        return out, labels


# =========================
# 算子2: SR - 超分辨率放大
# =========================

class SROperator(RepairOperator):
    """先 crop 再用超分放大"""
    name = "sr"

    def __init__(self, sr_engine=None, imgsz=720, crop_scale_range=(2.0, 4.0)):
        super().__init__(sr_engine, imgsz)
        self.crop_scale_range = crop_scale_range

    def apply(self, img, lines, target_box, **kwargs):
        if self.sr_engine is None:
            return None, None

        h, w = img.shape[:2]
        x1, y1, x2, y2 = target_box
        bw_t, bh_t = x2 - x1, y2 - y1
        if bw_t <= 0 or bh_t <= 0:
            return None, None

        scale = random.uniform(*self.crop_scale_range)
        crop_w, crop_h = bw_t * scale, bh_t * scale
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        c_x1 = int(max(0, cx - crop_w / 2))
        c_y1 = int(max(0, cy - crop_h / 2))
        c_x2 = int(min(w, cx + crop_w / 2))
        c_y2 = int(min(h, cy + crop_h / 2))

        if c_x2 <= c_x1 or c_y2 <= c_y1:
            return None, None

        crop = img[c_y1:c_y2, c_x1:c_x2]
        if crop.size == 0:
            return None, None

        # 超分放大
        upscaled = self.sr_engine.enhance(crop)
        # 缩放到目标尺寸
        out_size = (self.imgsz, self.imgsz)
        out = letterbox_resize(upscaled, out_size)
        labels = transform_labels_to_crop(lines, img.shape, [c_x1, c_y1, c_x2, c_y2], out_size)

        if not labels:
            return None, None
        return out, labels


# =========================
# 算子3: Zoom+SR 组合
# =========================

class ZoomSROperator(RepairOperator):
    """先 Zoom crop 再 SR"""
    name = "zoom_sr"

    def __init__(self, sr_engine=None, imgsz=720, zoom_range=(4.0, 8.0)):
        super().__init__(sr_engine, imgsz)
        self.zoom_range = zoom_range

    def apply(self, img, lines, target_box, **kwargs):
        if self.sr_engine is None:
            return None, None

        h, w = img.shape[:2]
        x1, y1, x2, y2 = target_box
        bw_t, bh_t = x2 - x1, y2 - y1
        if bw_t <= 0 or bh_t <= 0:
            return None, None

        scale = random.uniform(*self.zoom_range)
        crop_w = min(bw_t * scale, self.imgsz * 0.6)
        crop_h = min(bh_t * scale, self.imgsz * 0.6)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        c_x1 = int(max(0, cx - crop_w / 2))
        c_y1 = int(max(0, cy - crop_h / 2))
        c_x2 = int(min(w, cx + crop_w / 2))
        c_y2 = int(min(h, cy + crop_h / 2))

        if c_x2 <= c_x1 or c_y2 <= c_y1:
            return None, None

        crop = img[c_y1:c_y2, c_x1:c_x2]
        if crop.size == 0:
            return None, None

        upscaled = self.sr_engine.enhance(crop)
        out_size = (self.imgsz, self.imgsz)
        out = letterbox_resize(upscaled, out_size)
        labels = transform_labels_to_crop(lines, img.shape, [c_x1, c_y1, c_x2, c_y2], out_size)

        if not labels:
            return None, None
        return out, labels


# =========================
# 算子4: Context Crop - 保留上下文
# =========================

class ContextCropOperator(RepairOperator):
    """保留目标周围上下文的裁剪"""
    name = "context_crop"

    def __init__(self, sr_engine=None, imgsz=720, context_ratio=0.5):
        super().__init__(sr_engine, imgsz)
        self.context_ratio = context_ratio

    def apply(self, img, lines, target_box, **kwargs):
        h, w = img.shape[:2]
        x1, y1, x2, y2 = target_box
        bw_t, bh_t = x2 - x1, y2 - y1
        if bw_t <= 0 or bh_t <= 0:
            return None, None

        # 以目标框为中心，扩展上下文
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        expand = 1 + self.context_ratio * 2
        crop_w = bw_t * expand
        crop_h = bh_t * expand

        c_x1 = int(max(0, cx - crop_w / 2))
        c_y1 = int(max(0, cy - crop_h / 2))
        c_x2 = int(min(w, cx + crop_w / 2))
        c_y2 = int(min(h, cy + crop_h / 2))

        if c_x2 <= c_x1 or c_y2 <= c_y1:
            return None, None

        crop = img[c_y1:c_y2, c_x1:c_x2]
        if crop.size == 0:
            return None, None

        out_size = (self.imgsz, self.imgsz)
        out = letterbox_resize(crop, out_size)
        labels = transform_labels_to_crop(lines, img.shape, [c_x1, c_y1, c_x2, c_y2], out_size)

        if not labels:
            return None, None
        return out, labels


# =========================
# 算子5: Copy-Paste - 构造遮挡
# =========================

class CopyPasteOperator(RepairOperator):
    """将目标复制到其他位置，构造遮挡场景"""
    name = "copy_paste"

    def apply(self, img, lines, target_box, all_gt_boxes=None, **kwargs):
        h, w = img.shape[:2]
        x1, y1, x2, y2 = map(int, target_box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None, None

        target_crop = img[y1:y2, x1:x2].copy()
        if target_crop.size == 0:
            return None, None

        # 随机选择粘贴位置（靠近原目标）
        offset_x = random.randint(-int(w * 0.2), int(w * 0.2))
        offset_y = random.randint(-int(h * 0.2), int(h * 0.2))
        nx1 = max(0, min(w - (x2 - x1), x1 + offset_x))
        ny1 = max(0, min(h - (y2 - y1), y1 + offset_y))
        nx2 = nx1 + (x2 - x1)
        ny2 = ny1 + (y2 - y1)

        # 粘贴
        out_img = img.copy()
        out_img[ny1:ny2, nx1:nx2] = target_crop

        # 更新标签（添加新目标）
        parsed = parse_label_lines(lines)
        new_label = xyxy2xywh_norm([nx1, ny1, nx2, ny2], w, h)
        cls_id = 0
        if parsed:
            cls_id = parsed[0][0]  # 使用第一个目标的类别

        out_lines = list(lines)
        out_lines.append(f"{cls_id} {new_label[0]:.6f} {new_label[1]:.6f} {new_label[2]:.6f} {new_label[3]:.6f}")

        out_size = (self.imgsz, self.imgsz)
        out = letterbox_resize(out_img, out_size)
        # 重新计算所有标签
        out_labels = transform_labels_to_crop(out_lines, img.shape, [0, 0, w, h], out_size)

        if not out_labels:
            return None, None
        return out, out_labels


# =========================
# 算子6: Hard Negative - FP 背景学习
# =========================

class HardNegativeOperator(RepairOperator):
    """裁剪 FP 区域作为负样本"""
    name = "hard_negative"

    def apply(self, img, lines, fp_box, strict_negative=True, **kwargs):
        h, w = img.shape[:2]
        x1, y1, x2, y2 = fp_box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bw, bh = (x2 - x1), (y2 - y1)
        scale = random.uniform(2.5, 5.0)

        rw, rh = bw * scale, bh * scale
        c_x1 = int(max(0, cx - rw / 2))
        c_y1 = int(max(0, cy - rh / 2))
        c_x2 = int(min(w, cx + rw / 2))
        c_y2 = int(min(h, cy + rh / 2))

        if c_x2 <= c_x1 or c_y2 <= c_y1:
            return None, None

        crop = img[c_y1:c_y2, c_x1:c_x2]
        if crop.size == 0:
            return None, None

        out_size = (self.imgsz, self.imgsz)
        out = letterbox_resize(crop, out_size)
        labels = transform_labels_to_crop(lines, img.shape, [c_x1, c_y1, c_x2, c_y2], out_size)

        if strict_negative and labels:
            return None, None

        return out, labels


# =========================
# 算子7: Altitude Simulation - 尺度变化
# =========================

class AltitudeSimOperator(RepairOperator):
    """模拟不同飞行高度"""
    name = "altitude_sim"

    def apply(self, img, lines, target_box=None, mode="low", **kwargs):
        h, w = img.shape[:2]

        if mode == "low":
            # 模拟低空（center crop 50%）
            x1, y1 = int(0.25 * w), int(0.25 * h)
            x2, y2 = int(0.75 * w), int(0.75 * h)
        elif mode == "high":
            # 模拟高空（直接返回原图）
            return img, lines
        else:
            return None, None

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return None, None

        out_size = (w, h)
        out = letterbox_resize(crop, out_size)
        labels = transform_labels_to_crop(lines, img.shape, [x1, y1, x2, y2], out_size)

        return out, labels


# =========================
# 算子8: Mosaic - 密集场景
# =========================

class MosaicOperator(RepairOperator):
    """将多张图拼接成密集场景"""
    name = "mosaic"

    def apply(self, img, lines, target_box=None, other_imgs=None, other_lines_list=None, **kwargs):
        if other_imgs is None or len(other_imgs) < 3:
            return None, None

        out_size = self.imgsz
        mosaic_img = np.zeros((out_size, out_size, 3), dtype=np.uint8)

        # 4 张图拼接
        imgs = [img] + list(other_imgs[:3])
        lines_list = [lines] + list(other_lines_list[:3])

        positions = [
            (0, 0, out_size // 2, out_size // 2),
            (out_size // 2, 0, out_size, out_size // 2),
            (0, out_size // 2, out_size // 2, out_size),
            (out_size // 2, out_size // 2, out_size, out_size),
        ]

        all_labels = []
        for i, (src_img, src_lines) in enumerate(zip(imgs, lines_list)):
            x1, y1, x2, y2 = positions[i]
            pw, ph = x2 - x1, y2 - y1

            resized = cv2.resize(src_img, (pw, ph))
            mosaic_img[y1:y2, x1:x2] = resized

            # 调整标签
            parsed = parse_label_lines(src_lines)
            for cls, cx, cy, bw, bh in parsed:
                new_cx = (cx * pw + x1) / out_size
                new_cy = (cy * ph + y1) / out_size
                new_bw = bw * pw / out_size
                new_bh = bh * ph / out_size
                if 0 < new_cx < 1 and 0 < new_cy < 1:
                    all_labels.append(f"{cls} {new_cx:.6f} {new_cy:.6f} {new_bw:.6f} {new_bh:.6f}")

        if not all_labels:
            return None, None

        # 检查目标尺寸退化：Mosaic 将每张图缩小至 imgsz/2，可能导致目标过小
        small_count = 0
        for label in all_labels:
            parts = label.split()
            if len(parts) >= 5:
                bw, bh = float(parts[3]), float(parts[4])
                side = out_size * np.sqrt(bw * bh)
                if side < 32:
                    small_count += 1
        if small_count > 0:
            print(f"  [Mosaic] 警告: {small_count}/{len(all_labels)} 个目标尺寸 < 32px，"
                  f"YOLO 自带 Mosaic 训练时会进一步 resize，该退化通常在训练 pipeline 中缓解")

        return mosaic_img, all_labels


# =========================
# 算子工厂
# =========================

OPERATOR_REGISTRY = {
    "zoom_crop": ZoomCropOperator,
    "sr": SROperator,
    "zoom_sr": ZoomSROperator,
    "context_crop": ContextCropOperator,
    "copy_paste": CopyPasteOperator,
    "hard_negative": HardNegativeOperator,
    "altitude_sim": AltitudeSimOperator,
    "mosaic": MosaicOperator,
}


def create_operator(name, sr_engine=None, imgsz=720, **kwargs):
    cls = OPERATOR_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"未知算子: {name}")
    return cls(sr_engine=sr_engine, imgsz=imgsz, **kwargs)
