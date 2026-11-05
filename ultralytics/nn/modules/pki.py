from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from .block import C3k2
from .conv import Conv

__all__ = ("PKIEnhancer", "C3k2_PKI_R")


class PKIEnhancer(nn.Module):
    """
    Lightweight Poly-Kernel Inception feature enhancer.

    Adapted from the multi-kernel InceptionBottleneck in PKINet
    (CVPR 2024). CAA is intentionally omitted because LSK attention
    is already used separately in the detection network.
    """

    def __init__(
        self,
        channels: int,
        expansion: float = 0.5,
        kernel_sizes: Sequence[int] = (3, 5, 7, 9, 11),
    ):
        super().__init__()

        if channels <= 0:
            raise ValueError("channels must be positive.")

        if not 0.0 < expansion <= 1.0:
            raise ValueError("expansion must be in the range (0, 1].")

        if len(kernel_sizes) != 5:
            raise ValueError("kernel_sizes must contain exactly five kernels.")

        if any(k <= 0 or k % 2 == 0 for k in kernel_sizes):
            raise ValueError("All kernel sizes must be positive odd integers.")

        # Compress channels before running large depthwise kernels.
        hidden = max(8, ((int(channels * expansion) + 7) // 8) * 8)

        self.pre_conv = Conv(channels, hidden, 1, 1)

        # First local 3x3 depthwise convolution.
        self.dw_conv = nn.Conv2d(
            hidden,
            hidden,
            kernel_size=kernel_sizes[0],
            stride=1,
            padding=kernel_sizes[0] // 2,
            groups=hidden,
            bias=False,
        )

        # Parallel poly-kernel depthwise branches: 5x5, 7x7, 9x9, 11x11.
        self.dw_branches = nn.ModuleList(
            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=k,
                stride=1,
                padding=k // 2,
                groups=hidden,
                bias=False,
            )
            for k in kernel_sizes[1:]
        )

        self.pw_conv = Conv(hidden, hidden, 1, 1)
        self.post_conv = Conv(hidden, channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_conv(x)
        identity = x

        local = self.dw_conv(x)

        # Official PKI-style multi-kernel aggregation:
        # all large-kernel branches process the local 3x3 feature.
        multi_scale = local
        for branch in self.dw_branches:
            multi_scale = multi_scale + branch(local)

        multi_scale = self.pw_conv(multi_scale)

        # Multiplicative local interaction retained from PKI bottleneck.
        multi_scale = multi_scale + multi_scale * identity

        return self.post_conv(multi_scale)


class C3k2_PKI_R(C3k2):
    """
    Original YOLO11 C3k2 followed by a zero-initialized residual
    Poly-Kernel Inception enhancement branch.

    F_out = F_base + alpha * PKI(F_base)
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        pki_expansion: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
    ):
        super().__init__(
            c1=c1,
            c2=c2,
            n=n,
            c3k=c3k,
            e=e,
            attn=False,
            g=g,
            shortcut=shortcut,
        )

        self.pki = PKIEnhancer(
            channels=c2,
            expansion=pki_expansion,
        )

        # alpha=0 means the initial forward is exactly the original C3k2.
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = super().forward(x)
        return base + self.alpha * self.pki(base)

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        base = super().forward_split(x)
        return base + self.alpha * self.pki(base)