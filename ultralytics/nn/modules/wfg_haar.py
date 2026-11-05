# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Water-aware frequency-gated Haar downsampling module."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv

__all__ = ("WFGHaarDown",)


class WFGHaarDown(nn.Module):
    """Water-aware Frequency-Gated Haar stride-2 downsampling.

    Data flow:
        input
        -> fixed orthonormal Haar DWT
        -> LL low-frequency bypass
        -> adaptive soft-thresholding of LH/HL/HH
        -> residual depthwise high-frequency refinement
        -> bounded local cross-band gate
        -> full-channel 1x1 projection
        -> output
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 3,
        gate_scale: float = 0.5,
        threshold_init: float = 0.0025,
        threshold_max: float = 0.25,
        act: bool | nn.Module = True,
    ):
        super().__init__()

        if c1 <= 0 or c2 <= 0:
            raise ValueError(
                f"c1 and c2 must be positive, got c1={c1}, c2={c2}"
            )

        if k < 1 or k % 2 == 0:
            raise ValueError(
                f"k must be a positive odd integer, got k={k}"
            )

        if not 0.0 <= gate_scale < 1.0:
            raise ValueError(
                "gate_scale must be in [0, 1), "
                f"got gate_scale={gate_scale}"
            )

        if not 0.0 < threshold_init < threshold_max <= 1.0:
            raise ValueError(
                "threshold values must satisfy "
                "0 < threshold_init < threshold_max <= 1, "
                f"got threshold_init={threshold_init}, "
                f"threshold_max={threshold_max}"
            )

        self.c1 = int(c1)
        self.c2 = int(c2)
        self.gate_scale = float(gate_scale)
        self.threshold_max = float(threshold_max)

        # 固定正交 Haar 滤波器，依次为 LL、LH、HL、HH。
        # 它们是 buffer，不参与训练，也不计入模型参数量。
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
            haar.repeat(self.c1, 1, 1, 1),
            persistent=False,
        )

        high_channels = 3 * self.c1

        # 对 LH/HL/HH 进行低成本逐通道空间细化。
        # act=False 非常重要：不能用 SiLU/ReLU 破坏 Haar 高频系数正负号。
        self.high_refine = Conv(
            high_channels,
            high_channels,
            k=k,
            s=1,
            g=high_channels,
            act=False,
        )

        # 零初始化 BN，使残差分支初始严格为 0：
        # high + high_refine(high) = high。
        nn.init.zeros_(self.high_refine.bn.weight)
        nn.init.zeros_(self.high_refine.bn.bias)

        # 根据三个高频子带的局部响应联合生成空间门控。
        self.band_gate = nn.Conv2d(
            3,
            3,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        nn.init.zeros_(self.band_gate.weight)
        nn.init.zeros_(self.band_gate.bias)

        # 三个可学习阈值，分别对应 LH、HL、HH。
        # 参数使用一维形式，并包含 logit_scale：
        # 兼容当前 optimizer=auto 可能选择的 MuSGD 参数分组。
        initial_probability = threshold_init / threshold_max
        initial_logit = math.log(
            initial_probability / (1.0 - initial_probability)
        )

        self.logit_scale_threshold = nn.Parameter(
            torch.full(
                (3,),
                initial_logit,
                dtype=torch.float32,
            )
        )

        # 负责跨通道、跨频带融合。
        self.project = Conv(
            4 * self.c1,
            self.c2,
            k=1,
            s=1,
            act=act,
        )

    def _haar_dwt(self, x: torch.Tensor) -> torch.Tensor:
        """Return Haar coefficients shaped [B, C, 4, H/2, W/2]."""
        if x.ndim != 4:
            raise ValueError(
                f"WFGHaarDown expects BCHW input, got {tuple(x.shape)}"
            )

        b, c, h, w = x.shape

        if c != self.c1:
            raise ValueError(
                f"WFGHaarDown expected C={self.c1}, got C={c}"
            )

        if h % 2 != 0 or w % 2 != 0:
            raise ValueError(
                "WFGHaarDown requires even input height and width, "
                f"got H={h}, W={w}"
            )

        filters = self.haar_filters.to(
            device=x.device,
            dtype=x.dtype,
        )

        coefficients = F.conv2d(
            x,
            filters,
            bias=None,
            stride=2,
            padding=0,
            groups=self.c1,
        )

        return coefficients.reshape(
            b,
            self.c1,
            4,
            h // 2,
            w // 2,
        )

    def _soft_threshold(
        self,
        high: torch.Tensor,
    ) -> torch.Tensor:
        """Suppress weak repetitive high-frequency interference."""
        # FP16/BF16 下使用 FP32 计算统计量，提高数值稳定性。
        statistics = (
            high.float()
            if high.dtype in (torch.float16, torch.bfloat16)
            else high
        )

        # 每张图、每个通道、每个频带分别计算参考能量。
        # 形状：[B, C, 3, 1, 1]。
        reference = (
            statistics.abs()
            .mean(dim=(-2, -1), keepdim=True)
            .detach()
            .to(dtype=high.dtype)
        )

        threshold_fraction = (
            self.threshold_max
            * torch.sigmoid(self.logit_scale_threshold)
        ).to(dtype=high.dtype)

        threshold_fraction = threshold_fraction.view(
            1,
            1,
            3,
            1,
            1,
        )

        threshold = reference * threshold_fraction

        # 对称软阈值，不使用 sign，方便后续 TorchScript/ONNX 导出。
        return (
            F.relu(high - threshold)
            - F.relu(-high - threshold)
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        coefficients = self._haar_dwt(x)

        # LL 低频语义信息直接旁路。
        low = coefficients[:, :, :1]

        # LH、HL、HH 高频细节。
        high = coefficients[:, :, 1:]
        high = self._soft_threshold(high)

        b, _, _, h2, w2 = high.shape

        # 高频逐通道残差细化。
        high_flat = high.reshape(
            b,
            3 * self.c1,
            h2,
            w2,
        )

        high_flat = (
            high_flat
            + self.high_refine(high_flat)
        )

        high = high_flat.reshape(
            b,
            self.c1,
            3,
            h2,
            w2,
        )

        # 将通道压缩成三个频带描述图。
        descriptor = high.abs().mean(dim=1)

        # 零初始化时 band_weight=1；
        # gate_scale=0.5 时范围为 [0.5, 1.5]。
        band_weight = (
            1.0
            + self.gate_scale
            * torch.tanh(self.band_gate(descriptor))
        )

        high = high * band_weight.unsqueeze(1)

        # 保持每个原始通道的 LL/LH/HL/HH 相邻排列。
        fused = torch.cat(
            (low, high),
            dim=2,
        ).flatten(1, 2)

        return self.project(fused)