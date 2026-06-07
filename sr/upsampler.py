"""
RealESRGAN 推理封装 - 支持 tile 模式，控制显存
依赖：仅 PyTorch，无需 basicsr/gfpgan
"""

import numpy as np
import torch
from .srvgg_arch import SRVGGNetCompact


class RealESRGANUpsampler:
    def __init__(self, model_path, scale=4, tile=128, device='cuda:0'):
        self.scale = scale
        self.tile = tile
        self.device = device

        self.model = SRVGGNetCompact(
            num_in_ch=3, num_out_ch=3,
            num_feat=64, num_conv=32,
            upscale=scale, act_type='prelu'
        )
        state = torch.load(model_path, map_location='cpu')
        if 'params_ema' in state:
            state = state['params_ema']
        elif 'params' in state:
            state = state['params']
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

        try:
            self.model.to(device)
            self.device = device
        except Exception as e:
            print(f"CUDA 加载失败 ({e})，降级到 CPU")
            self.device = 'cpu'
            self.model.to('cpu')

    def enhance(self, img_bgr, outscale=None):
        img_rgb = img_bgr[:, :, ::-1].copy()
        img_tensor = torch.from_numpy(img_rgb).float().div_(255.0)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        if self.tile > 0:
            output = self._tile_process(img_tensor)
        else:
            with torch.no_grad():
                output = self.model(img_tensor)

        output = output.squeeze(0).clamp_(0, 1).permute(1, 2, 0).cpu().numpy()
        output = (output * 255.0).round().astype(np.uint8)
        output_bgr = output[:, :, ::-1].copy()

        if outscale is not None and outscale != self.scale:
            import cv2
            target_h = int(img_bgr.shape[0] * outscale)
            target_w = int(img_bgr.shape[1] * outscale)
            output_bgr = cv2.resize(output_bgr, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        return output_bgr

    def _tile_process(self, img_tensor):
        _, _, h, w = img_tensor.shape
        tile = self.tile
        sf = self.scale

        if h <= tile and w <= tile:
            with torch.no_grad():
                return self.model(img_tensor)

        h_pad = (tile - h % tile) % tile
        w_pad = (tile - w % tile) % tile
        img_padded = torch.nn.functional.pad(img_tensor, (0, w_pad, 0, h_pad), mode='reflect')

        _, _, H, W = img_padded.shape
        output = torch.zeros(1, 3, H * sf, W * sf, device=self.device)

        for y in range(0, H, tile):
            for x in range(0, W, tile):
                y2 = min(y + tile, H)
                x2 = min(x + tile, W)
                patch = img_padded[:, :, y:y2, x:x2]
                with torch.no_grad():
                    out_patch = self.model(patch)
                output[:, :, y*sf:y2*sf, x*sf:x2*sf] = out_patch

        output = output[:, :, :h*sf, :w*sf]
        return output
