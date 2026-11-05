# Ultralytics 🚀 AGPL-3.0 License - https://www.ultralytics.com/license
"""Water-surface small-object attention modules.

三个面向水面垃圾分类小目标检测的轻量注意力模块，用于替换检测头前的
LSK 注意力（原 C3k2_LSK）。设计动机与思想来源：

- WFSA (Wavelet Frequency Saliency Attention)
  水面背景在大尺度上是低频平滑区域，漂浮垃圾表现为局部高频异常；
  波浪反光虽然也产生高频，但能量沿固定方向弥散分布，而垃圾目标的
  高频能量集中且方向接近各向同性。模块用固定正交 Haar 变换（零参数）
  提取 LH/HL/HH 三个方向的高频能量，经可学习的方向融合（抑制方向性
  波纹噪声）后做组对比度归一化（借鉴 SGE 的组显著性思想），生成
  空间显著性门控；低频子带经 ECA 式一维卷积（借鉴 ECA/MLCA）生成
  通道门控。模糊目标即使边缘退化，残存高频能量仍集中在目标局部，
  因此该门控对模糊目标同样敏感。

- LCPA (Local-Contrast Polarized Attention)
  借鉴极化自注意力（PSA）「先通道极化、再空间极化」的顺序结构
  （有利于像素级定位回归），但将空间分支的全局 matmul 替换为
  局部对比度：每个位置与其 k x k 邻域均值之差的能量决定门控。
  水面场景中背景占绝对主导，全局归一化会淹没小目标的显著信号，
  局部对比让目标从「周边小块水面」中弹出，同时对全局光照/反光
  变化不敏感。

- CGMA (Context-Gated Modulation Attention)
  借鉴 CoT（上下文作为 key）与 LSK/SK（可选感受野）的思想：
  模糊目标的判别线索更多来自「与周边水域的上下文差异」而非自身
  清晰纹理。模块用两个深度卷积（d=1 局部上下文、d=3 扩张上下文）
  构造静态上下文，将「当前特征 − 扩张上下文」的异常信号与局部
  上下文拼接后生成空间门控，并用 ECA 式通道门控补偿类别判别性。
  相比 LSK 的大核空间门控，参数量与计算量下降约一个数量级。

三个模块共用「零初始化调制」（ReZero 式）设计：所有门控分支在
初始化时严格输出单位因子（乘性恒等），因此替换进已有结构后，
初始前向行为与无注意力基线完全一致，训练稳定，模块只会在确实
有效时才逐步发挥作用。所有模块输入输出形状不变。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("WFSA", "LCPA", "CGMA")


def _largest_divisor(value: int, upper: int) -> int:
    """Return the largest divisor of ``value`` that is <= ``upper``."""
    g = min(upper, value)
    while g > 1 and value % g != 0:
        g -= 1
    return g


def _eca_kernel(channels: int, gamma: float = 2.0, b: int = 1) -> int:
    """Adaptive 1D kernel size following the ECA rule."""
    t = int(abs(math.log2(channels) + b) / gamma)
    return t if t % 2 else t + 1


class WFSA(nn.Module):
    """Wavelet Frequency Saliency Attention.

    Data flow:
        input
        -> fixed orthonormal Haar DWT (LL, LH, HL, HH at half resolution)
        -> per-group directional high-frequency energy -> learnable fusion
        -> upsample + SGE-style group contrast normalization -> spatial gate
        -> LL global descriptor -> ECA-style 1D conv -> channel gate
        -> input * spatial gate * channel gate
    """

    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        if channels < 2:
            raise ValueError(f"WFSA requires channels >= 2, got {channels}")

        self.channels = int(channels)
        self.groups = _largest_divisor(self.channels, groups)

        # 固定正交 Haar 滤波器，依次为 LL、LH、HL、HH。
        # buffer 不参与训练，也不计入模型参数量。
        haar = torch.tensor(
            [
                [[0.5, 0.5], [0.5, 0.5]],
                [[-0.5, 0.5], [-0.5, 0.5]],
                [[-0.5, -0.5], [0.5, 0.5]],
                [[0.5, -0.5], [-0.5, 0.5]],
            ],
            dtype=torch.float32,
        ).unsqueeze(1)
        self.register_buffer(
            "haar_filters",
            haar.repeat(self.channels, 1, 1, 1),
            persistent=False,
        )

        # 可学习方向融合：三个方向高频能量 -> 1。
        # 初始化为等权平均，等价于无方向偏置的能量求和。
        self.dir_fuse = nn.Conv2d(3, 1, kernel_size=1, bias=True)
        nn.init.constant_(self.dir_fuse.weight, 1.0 / 3.0)
        nn.init.zeros_(self.dir_fuse.bias)

        # 组对比度归一化后的零初始化缩放/偏置：初始恒等。
        self.sp_scale = nn.Parameter(torch.zeros(self.groups))
        self.sp_bias = nn.Parameter(torch.zeros(self.groups))

        # 低频通道门控（ECA 式一维卷积，零初始化：初始恒等）。
        k = _eca_kernel(self.channels)
        self.ch_conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        nn.init.zeros_(self.ch_conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        g = self.groups

        if h % 2 != 0 or w % 2 != 0:
            x_pad = F.pad(x, (0, w % 2, 0, h % 2), mode="replicate")
        else:
            x_pad = x
        h2, w2 = x_pad.shape[2] // 2, x_pad.shape[3] // 2

        filters = self.haar_filters.to(device=x.device, dtype=x.dtype)
        coeffs = F.conv2d(x_pad, filters, stride=2, groups=c)
        coeffs = coeffs.reshape(b, c, 4, h2, w2)
        ll, lh, hl, hh = coeffs.unbind(dim=2)

        # 每组三个方向的高频能量 (b, g, 3, h2, w2)。
        def _group_energy(band: torch.Tensor) -> torch.Tensor:
            return (
                band.abs()
                .view(b, g, c // g, h2, w2)
                .mean(dim=2)
            )

        energy = torch.stack(
            (_group_energy(lh), _group_energy(hl), _group_energy(hh)),
            dim=2,
        ).reshape(b * g, 3, h2, w2)
        energy = self.dir_fuse(energy)  # (b*g, 1, h2, w2)
        energy = F.interpolate(energy, size=(h, w), mode="bilinear", align_corners=False)

        # SGE 式组对比度归一化 + 零初始化缩放。
        t = energy.flatten(1)
        t = (t - t.mean(dim=1, keepdim=True)) / (t.std(dim=1, keepdim=True) + 1e-5)
        t = t.view(b * g, 1, h, w)
        t = t * self.sp_scale.view(1, g, 1, 1).expand(b, g, 1, 1).reshape(b * g, 1, 1, 1)
        t = t + self.sp_bias.view(1, g, 1, 1).expand(b, g, 1, 1).reshape(b * g, 1, 1, 1)
        spatial = 1.0 + torch.tanh(t)  # (b*g, 1, h, w) in (0, 2)

        spatial = (
            spatial.view(b, g, 1, h, w)
            .expand(b, g, c // g, h, w)
            .reshape(b, c, h, w)
        )

        # 低频通道门控。
        z = ll.flatten(2).mean(dim=-1)  # (b, c)
        channel = 1.0 + torch.tanh(self.ch_conv(z.unsqueeze(1)).squeeze(1))
        channel = channel.view(b, c, 1, 1)

        return x * spatial * channel


class LCPA(nn.Module):
    """Local-Contrast Polarized Attention.

    Data flow:
        input
        -> channel polarization: low-rank value @ softmax(spatial query)
           -> 1x1 lift -> zero-init tanh gate  (PSA-inspired)
        -> spatial polarization: local window contrast energy
           -> normalization -> zero-init tanh gate  (novel)
        -> input * channel gate * spatial gate
    """

    def __init__(self, channels: int, r: int = 2, k: int = 7):
        super().__init__()
        if channels < 4:
            raise ValueError(f"LCPA requires channels >= 4, got {channels}")

        self.channels = int(channels)
        self.r = _largest_divisor(self.channels, max(2, r))
        self.k = int(k)
        if self.k < 3 or self.k % 2 == 0:
            raise ValueError(f"k must be an odd integer >= 3, got {k}")

        c_low = self.channels // self.r

        # 通道极化分支（PSA 式低秩结构，去掉了 LayerNorm 以配合零初始化）。
        self.ch_wv = nn.Conv2d(self.channels, c_low, kernel_size=1)
        self.ch_wq = nn.Conv2d(self.channels, 1, kernel_size=1)
        self.ch_wz = nn.Conv2d(c_low, self.channels, kernel_size=1)
        nn.init.zeros_(self.ch_wz.weight)
        nn.init.zeros_(self.ch_wz.bias)

        # 空间局部对比分支。
        self.sp_proj = nn.Conv2d(self.channels, c_low, kernel_size=1)
        self.sp_scale = nn.Parameter(torch.zeros(1))
        self.sp_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        c_low = c // self.r

        # ---- 通道极化 ----
        wv = self.ch_wv(x).reshape(b, c_low, h * w)  # (b, c_low, hw)
        wq = self.ch_wq(x).reshape(b, h * w, 1).softmax(dim=1)  # (b, hw, 1)
        wz = torch.matmul(wv, wq).unsqueeze(-1)  # (b, c_low, 1, 1)
        cw = 1.0 + torch.tanh(self.ch_wz(wz)).view(b, c, 1, 1)  # (b, c, 1, 1)
        x = x * cw

        # ---- 空间局部对比 ----
        v = self.sp_proj(x)  # (b, c_low, h, w)
        ctx = F.avg_pool2d(
            v,
            kernel_size=self.k,
            stride=1,
            padding=self.k // 2,
            count_include_pad=False,
        )
        energy = (v - ctx).pow(2).mean(dim=1, keepdim=True)  # (b, 1, h, w)
        t = energy.flatten(1)
        t = (t - t.mean(dim=1, keepdim=True)) / (t.std(dim=1, keepdim=True) + 1e-5)
        sw = 1.0 + torch.tanh(t.view(b, 1, h, w) * self.sp_scale + self.sp_bias)

        return x * sw


class CGMA(nn.Module):
    """Context-Gated Modulation Attention.

    Data flow:
        input
        -> depthwise local context (3x3, d=1)
        -> depthwise wide context (3x3, d=3)
        -> anomaly = input - wide context
        -> concat(anomaly, local context) -> conv gate -> zero-init tanh
        -> global pooled input -> ECA-style 1D conv -> zero-init tanh
        -> input * spatial gate * channel gate
    """

    def __init__(self, channels: int, k: int = 3, dilation: int = 3, r: int = 4):
        super().__init__()
        if channels < 4:
            raise ValueError(f"CGMA requires channels >= 4, got {channels}")

        self.channels = int(channels)
        self.r = _largest_divisor(self.channels, max(2, r))

        if k < 3 or k % 2 == 0:
            raise ValueError(f"k must be an odd integer >= 3, got {k}")

        # 静态上下文（CoT 思想）：局部 + 扩张两个感受野。
        self.ctx_local = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=k,
            padding=k // 2,
            groups=self.channels,
            bias=False,
        )
        self.ctx_wide = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=k,
            padding=(k // 2) * dilation,
            dilation=dilation,
            groups=self.channels,
            bias=False,
        )

        # 空间门控：异常信号 + 局部上下文 -> 门控。
        c_low = self.channels // self.r
        self.gate = nn.Sequential(
            nn.Conv2d(2 * self.channels, c_low, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_low),
            nn.SiLU(),
            nn.Conv2d(c_low, 1, kernel_size=1, bias=False),
        )
        nn.init.zeros_(self.gate[-1].weight)

        # 通道门控（ECA 式，零初始化）。
        k_eca = _eca_kernel(self.channels)
        self.ch_conv = nn.Conv1d(
            1, 1, kernel_size=k_eca, padding=(k_eca - 1) // 2, bias=False
        )
        nn.init.zeros_(self.ch_conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape

        ctx_l = self.ctx_local(x)
        ctx_w = self.ctx_wide(x)
        anomaly = x - ctx_w
        gate_in = torch.cat((anomaly, ctx_l), dim=1)
        s = 1.0 + torch.tanh(self.gate(gate_in))  # (b, 1, h, w)

        z = x.flatten(2).mean(dim=-1)  # (b, c)
        c_att = 1.0 + torch.tanh(self.ch_conv(z.unsqueeze(1)).squeeze(1))
        c_att = c_att.view(b, c, 1, 1)

        return x * s * c_att
