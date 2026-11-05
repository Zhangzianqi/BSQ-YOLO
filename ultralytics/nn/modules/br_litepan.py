# Ultralytics 🚀 AGPL-3.0 License
"""BSQ-aware Residual Lightweight PAN module."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("BRLitePAN",)


def _valid_gn_groups(
    channels: int,
    requested_groups: int,
) -> int:
    """Return a valid GroupNorm group count."""

    groups = max(
        1,
        min(int(channels), int(requested_groups)),
    )

    while channels % groups != 0:
        groups -= 1

    return groups


class _BRProject(nn.Module):
    """Level-specific 1x1 projection followed by GroupNorm."""

    def __init__(
        self,
        c1: int,
        c2: int,
        gn_groups: int = 8,
    ):
        super().__init__()

        self.conv = (
            nn.Identity()
            if c1 == c2
            else nn.Conv2d(
                c1,
                c2,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
        )

        self.norm = nn.GroupNorm(
            _valid_gn_groups(c2, gn_groups),
            c2,
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class _BRDownProject(nn.Module):
    """
    Anti-aliased lightweight downsampling.

    AvgPool suppresses high-frequency water noise.
    DW-PW convolution performs inexpensive channel transformation.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        gn_groups: int = 8,
    ):
        super().__init__()

        self.pool = nn.AvgPool2d(
            kernel_size=2,
            stride=2,
        )

        self.dw = nn.Conv2d(
            c1,
            c1,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=c1,
            bias=False,
        )
        self.dw_norm = nn.GroupNorm(
            _valid_gn_groups(c1, gn_groups),
            c1,
        )

        self.pw = nn.Conv2d(
            c1,
            c2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.pw_norm = nn.GroupNorm(
            _valid_gn_groups(c2, gn_groups),
            c2,
        )

        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)

        x = self.dw(x)
        x = self.dw_norm(x)
        x = self.act(x)

        x = self.pw(x)
        x = self.pw_norm(x)

        return self.act(x)


class _BRResidualRefine(nn.Module):
    """DW-PW residual refinement without another full convolution."""

    def __init__(
        self,
        channels: int,
        gn_groups: int = 8,
    ):
        super().__init__()

        groups = _valid_gn_groups(
            channels,
            gn_groups,
        )

        self.dw = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.dw_norm = nn.GroupNorm(
            groups,
            channels,
        )

        self.pw = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.pw_norm = nn.GroupNorm(
            groups,
            channels,
        )

        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.dw_norm(x)
        x = self.act(x)

        x = self.pw(x)
        return self.pw_norm(x)


class BRLitePAN(nn.Module):
    """
    BSQ-aware Residual Lightweight PAN stage.

    Input order:
        0: high-resolution bottom-up feature
        1: same-scale lateral/top-down feature
        2: BSQ semantic support feature

    Output:
        Fused feature at the lateral feature resolution.
    """

    def __init__(
        self,
        c_high: int,
        c_lateral: int,
        c_bsq: int,
        c_out: int,
        reduction: int = 4,
        gn_groups: int = 8,
    ):
        super().__init__()

        if reduction < 1:
            raise ValueError(
                f"reduction must be >= 1, got {reduction}."
            )

        c_mid = max(c_out // reduction, 8)

        # Bottom-up lightweight downsampling.
        self.down = _BRDownProject(
            c_high,
            c_out,
            gn_groups,
        )

        # Same-scale lateral feature.
        self.lateral = _BRProject(
            c_lateral,
            c_out,
            gn_groups,
        )

        # BSQ semantic support.
        self.support = _BRProject(
            c_bsq,
            c_out,
            gn_groups,
        )

        # Learnable normalized weights for bottom-up/lateral fusion.
        self.fusion_logits = nn.Parameter(
            torch.zeros(2)
        )

        # Channel gate for BSQ residual correction.
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                2 * c_out,
                c_mid,
                kernel_size=1,
                bias=True,
            ),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                c_mid,
                c_out,
                kernel_size=1,
                bias=True,
            ),
            nn.Sigmoid(),
        )

        # Spatial agreement statistics:
        # fused, support, difference, correlation.
        self.spatial_gate = nn.Conv2d(
            4,
            1,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        # BSQ correction begins as an identity residual.
        self.bsq_scale = nn.Parameter(
            torch.zeros(1)
        )

        self.refine = _BRResidualRefine(
            c_out,
            gn_groups,
        )
        self.refine_scale = nn.Parameter(
            torch.tensor(0.1)
        )

        self.out_act = nn.SiLU(inplace=True)

        # Initial gates are neutral.
        nn.init.zeros_(
            self.channel_gate[-2].weight
        )
        nn.init.zeros_(
            self.channel_gate[-2].bias
        )
        nn.init.zeros_(
            self.spatial_gate.weight
        )
        nn.init.zeros_(
            self.spatial_gate.bias
        )

    def forward(
        self,
        x: list[torch.Tensor] | tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        if not isinstance(x, (list, tuple)) or len(x) != 3:
            raise ValueError(
                "BRLitePAN expects "
                "[high_res, lateral, bsq_support]."
            )

        high, lateral, support = x
        target_size = lateral.shape[-2:]

        # P2->P3, P3->P4 or P4->P5.
        down = self.down(high)

        if down.shape[-2:] != target_size:
            down = F.interpolate(
                down,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )

        # S4 is downsampled when it supports the P5 stage.
        if support.shape[-2:] != target_size:
            support = F.adaptive_avg_pool2d(
                support,
                target_size,
            )

        lateral = self.lateral(lateral)
        support = self.support(support)

        # Normalized bottom-up and lateral fusion.
        weights = self.fusion_logits.softmax(dim=0)
        fused = (
            weights[0] * down
            + weights[1] * lateral
        )

        # Correct toward BSQ support rather than blindly adding it.
        bsq_delta = support - lateral

        channel_gate = self.channel_gate(
            torch.cat(
                (fused, bsq_delta),
                dim=1,
            )
        )

        spatial_statistics = torch.cat(
            (
                fused.mean(
                    dim=1,
                    keepdim=True,
                ),
                support.mean(
                    dim=1,
                    keepdim=True,
                ),
                (fused - support)
                .abs()
                .mean(
                    dim=1,
                    keepdim=True,
                ),
                (fused * support)
                .mean(
                    dim=1,
                    keepdim=True,
                ),
            ),
            dim=1,
        )

        spatial_gate = torch.sigmoid(
            self.spatial_gate(
                spatial_statistics
            )
        )

        fused = (
            fused
            + self.bsq_scale.tanh()
            * channel_gate
            * spatial_gate
            * bsq_delta
        )

        refined = self.refine(fused)

        return self.out_act(
            fused
            + self.refine_scale.tanh()
            * refined
        )