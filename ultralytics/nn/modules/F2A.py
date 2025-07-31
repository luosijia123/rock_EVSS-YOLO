import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

class F2A(nn.Module):
    """
    Frequency-Filtered Attention (F2A)
    输入 (B, C, H, W)，输出 (B, C, H, W)
    """
    def __init__(self, freq_keep_ratio=0.5):
        super(F2A, self).__init__()
        self.freq_keep_ratio = freq_keep_ratio
        self.scale = nn.Parameter(torch.ones(1))  # 可学习缩放因子
        self.bias = nn.Parameter(torch.zeros(1))  # 可学习偏置项

    def forward(self, x):
        b, c, h, w = x.shape
        input_dtype = x.dtype  # 保存输入的数据类型（可能是 float16/float32）

        # 强制使用 float32 计算 FFT
        x_fft = torch.fft.fft2(x.to(torch.float32), norm='ortho')  # 使用 float32 避免 ComplexHalf
        x_fft_amp = torch.abs(x_fft)  # 幅度谱是实数

        # 计算频率掩码（和之前一样）
        flat = x_fft_amp.view(b, c, -1)
        threshold_idx = int(flat.size(-1) * self.freq_keep_ratio)
        threshold_values, _ = torch.topk(flat, threshold_idx, dim=-1)
        min_threshold = threshold_values.min(dim=-1, keepdim=True)[0].unsqueeze(-1)
        freq_mask = (x_fft_amp >= min_threshold.view(b, c, 1, 1)).to(x_fft.dtype)  # 保持复数类型

        # 滤波 + 反变换
        x_fft_filtered = x_fft * freq_mask
        x_filtered = torch.fft.ifft2(x_fft_filtered, norm='ortho').real

        # 恢复原始数据类型
        x_filtered = x_filtered.to(input_dtype)
        out = self.scale * x_filtered + self.bias + x
        return out


