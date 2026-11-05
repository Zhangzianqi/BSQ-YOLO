from __future__ import annotations

import torch
import torch.nn as nn

from .block import C3k2

__all__ = ("LSKAttention", "C3k2_LSK")


class _LSKSpatialGate(nn.Module):
    """Large Selective Kernel spatial gate."""

    def __init__(self, channels: int):
        super().__init__()
        if channels < 2:
            raise ValueError("LSKAttention requires channels >= 2.")

        hidden = channels // 2

        # Local 5x5 context
        self.local_dw = nn.Conv2d(
            channels,
            channels,
            kernel_size=5,
            padding=2,
            groups=channels,
        )

        # Large context: effective receptive field is about 23x23
        self.context_dw = nn.Conv2d(
            channels,
            channels,
            kernel_size=7,
            padding=9,
            dilation=3,
            groups=channels,
        )

        self.local_reduce = nn.Conv2d(channels, hidden, kernel_size=1)
        self.context_reduce = nn.Conv2d(channels, hidden, kernel_size=1)

        # Select local or large-range context at each position
        self.selector = nn.Conv2d(
            2,
            2,
            kernel_size=7,
            padding=3,
        )

        self.expand = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_feature = self.local_dw(x)
        context_feature = self.context_dw(local_feature)

        local_feature = self.local_reduce(local_feature)
        context_feature = self.context_reduce(context_feature)

        merged = torch.cat(
            (local_feature, context_feature),
            dim=1,
        )

        spatial_statistics = torch.cat(
            (
                merged.mean(dim=1, keepdim=True),
                merged.amax(dim=1, keepdim=True),
            ),
            dim=1,
        )

        weights = torch.sigmoid(self.selector(spatial_statistics))

        selected_feature = (
            local_feature * weights[:, 0:1]
            + context_feature * weights[:, 1:2]
        )

        attention = self.expand(selected_feature)
        return x * attention


class LSKAttention(nn.Module):
    """Residual LSK attention with unchanged input/output shape."""

    def __init__(self, channels: int):
        super().__init__()
        self.proj_in = nn.Conv2d(channels, channels, kernel_size=1)
        self.activation = nn.GELU()
        self.spatial_gate = _LSKSpatialGate(channels)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        x = self.proj_in(x)
        x = self.activation(x)
        x = self.spatial_gate(x)
        x = self.proj_out(x)

        return shortcut + x


class C3k2_LSK(C3k2):
    """Original C3k2 followed by LSK Attention."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
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
        self.lsk = LSKAttention(c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lsk(super().forward(x))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        return self.lsk(super().forward_split(x))