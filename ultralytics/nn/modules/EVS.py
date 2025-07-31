import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

from torchvision.transforms.functional import resize, to_pil_image

import warnings
warnings.filterwarnings('ignore')

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.norm(to_3d(x)), h, w)

class EDFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=3, bias=False):
        super().__init__()
        self.patch_size = 8  # 分块大小

    def forward(self, x):
        b, c, h, w = x.shape

        # 计算需要填充的量（使h和w能被patch_size整除）
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size

        # 对称填充（reflect/pad）
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        # 现在分块操作可以正常执行
        x_patch = rearrange(x, "b c (h p1) (w p2) -> b c h w p1 p2",
                          p1=self.patch_size, p2=self.patch_size)

        # ...（后续FFT操作）

        # 最后裁剪回原始尺寸
        x = x[:, :, :h, :w]
        return x

class PseudoMambaScan(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Conv2d(dim, dim, 1)
        self.state = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.update = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x):
        gate = torch.sigmoid(self.gate(x))
        state = self.update(x)
        return gate * x + (1 - gate) * state

class EVSS(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=3, bias=False, att=False, idx=3, patch=128):
        super().__init__()
        self.att = att
        self.idx = idx
        self.kernel_size = (patch, patch)

        if self.att:
            self.norm1 = LayerNorm(dim)
            self.attn = PseudoMambaScan(dim)

        self.norm2 = LayerNorm(dim)
        self.ffn = EDFFN(dim, ffn_expansion_factor, bias)

    def grids(self, x):
        b, c, h, w = x.shape
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)

        step_i = max(1, k1)
        step_j = max(1, k2)

        parts = []
        self.idxes = []
        for i in range(0, h - k1 + 1, step_i):
            for j in range(0, w - k2 + 1, step_j):
                parts.append(x[:, :, i:i + k1, j:j + k2])
                self.idxes.append((i, j))
        return torch.cat(parts, dim=0)

    def grids_inverse(self, outs, original_shape):
        b, c, h, w = original_shape
        preds = torch.zeros((b, c, h, w), device=outs.device)
        counts = torch.zeros((b, 1, h, w), device=outs.device)
        k1, k2 = self.kernel_size

        for idx, (i, j) in enumerate(self.idxes):
            preds[0, :, i:i + k1, j:j + k2] += outs[idx]
            counts[0, :, i:i + k1, j:j + k2] += 1

        return preds / counts

    def forward(self, x):
        if self.att:
            if self.idx % 2 == 1:
                x = torch.flip(x, dims=(-2, -1))
            if self.idx % 2 == 0:
                x = x.transpose(-2, -1)

            x_split = self.grids(x)
            x_split = x_split + self.attn(self.norm1(x_split))
            x = self.grids_inverse(x_split, x.shape)

        x = x + self.ffn(self.norm2(x))
        return x

