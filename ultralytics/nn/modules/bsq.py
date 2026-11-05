# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Blur-aware Semantic Query Feature Pyramid modules."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, DWConv

__all__ = ("BSQGather", "BSQFeedback")


class BSQGather(nn.Module):
    """
    Gather B4/P4 and B5/P5 into a semantic feature at B4 resolution.

    Input:
        x[0]: B4, stride 16
        x[1]: B5, stride 32

    Output:
        S4, same shape and channels as B4
    """

    def __init__(
        self,
        c4: int,
        c5: int,
        reduction: int = 4,
    ):
        super().__init__()

        if reduction < 1:
            raise ValueError(
                f"reduction must be >= 1, got {reduction}"
            )

        c_mid = max(c4 // reduction, 16)
        c_gate = max(c_mid // 2, 8)

        self.b4_reduce = Conv(
            c4,
            c_mid,
            k=1,
            s=1,
        )

        self.b5_reduce = Conv(
            c5,
            c_mid,
            k=1,
            s=1,
        )

        # 为B4、B5生成逐通道尺度权重
        self.scale_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                2 * c_mid,
                c_gate,
                kernel_size=1,
                bias=True,
            ),
            nn.SiLU(),
            nn.Conv2d(
                c_gate,
                2 * c_mid,
                kernel_size=1,
                bias=True,
            ),
        )

        self.refine = nn.Sequential(
            DWConv(
                c_mid,
                c_mid,
                k=3,
                s=1,
            ),
            Conv(
                c_mid,
                c4,
                k=1,
                s=1,
                act=False,
            ),
        )

        # 零初始化：刚开始训练时等价于原始B4
        self.gamma = nn.Parameter(torch.zeros(1))

        # 初始时B4、B5权重相同
        nn.init.zeros_(self.scale_gate[-1].weight)
        nn.init.zeros_(self.scale_gate[-1].bias)

    def forward(
        self,
        x: list[torch.Tensor] | tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError(
                "BSQGather expects two inputs: [B4, B5]."
            )

        b4, b5 = x

        z4 = self.b4_reduce(b4)
        z5 = self.b5_reduce(b5)

        # B5/32 -> B4/16
        z5 = F.interpolate(
            z5,
            size=z4.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        scale_input = torch.cat(
            (z4, z5),
            dim=1,
        )

        weights = self.scale_gate(scale_input)

        batch = weights.shape[0]
        channels = z4.shape[1]

        weights = weights.view(
            batch,
            2,
            channels,
            1,
            1,
        )

        # 在B4/B5尺度维度归一化
        weights = weights.softmax(dim=1)

        fused = (
            weights[:, 0] * z4
            + weights[:, 1] * z5
        )

        semantic = self.refine(fused)

        # 初始gamma=0，因此初始输出严格等于B4
        return (
            b4
            + self.gamma.tanh() * semantic
        )


class BSQFeedback(nn.Module):
    """
    Progressively feed deep semantic information into a shallower feature.

    Inputs:
        x[0]: local high-resolution feature, e.g. B3 or B2
        x[1]: deeper semantic feature, e.g. S4 or refined B3

    Output:
        Refined local feature with unchanged shape/channels.
    """

    def __init__(
        self,
        c_local: int,
        c_semantic: int,
        reduction: int = 8,
        max_offset: float = 1.0,
        use_detail: bool = True,
    ):
        super().__init__()

        if reduction < 1:
            raise ValueError(
                f"reduction must be >= 1, got {reduction}"
            )

        if max_offset < 0:
            raise ValueError(
                f"max_offset must be >= 0, got {max_offset}"
            )

        c_mid = max(c_local // reduction, 16)

        self.max_offset = float(max_offset)
        self.use_detail = bool(use_detail)

        # 浅层位置/细节特征
        self.local_reduce = Conv(
            c_local,
            c_mid,
            k=1,
            s=1,
        )

        # 深层语义特征
        self.semantic_reduce = Conv(
            c_semantic,
            c_mid,
            k=1,
            s=1,
        )

        self.semantic_refine = DWConv(
            c_mid,
            c_mid,
            k=3,
            s=1,
            act=False,
        )

        # 学习语义特征的空间校正偏移
        self.offset = nn.Sequential(
            Conv(
                2 * c_mid,
                c_mid,
                k=3,
                s=1,
            ),
            nn.Conv2d(
                c_mid,
                2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
        )

        # 输入四张统计图：
        # local、semantic、difference、correlation
        self.query_gate = nn.Sequential(
            nn.Conv2d(
                4,
                1,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.Sigmoid(),
        )

        self.semantic_expand = Conv(
            c_mid,
            c_local,
            k=1,
            s=1,
            act=False,
        )

        if self.use_detail:
            # 根据高频强度和语义支持程度判断细节是否可靠
            self.reliability_gate = nn.Sequential(
                nn.Conv2d(
                    2,
                    1,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                ),
                nn.Sigmoid(),
            )

            self.detail_reduce = Conv(
                c_local,
                c_mid,
                k=1,
                s=1,
            )

            self.detail_refine = DWConv(
                c_mid,
                c_mid,
                k=3,
                s=1,
                act=False,
            )

            self.detail_expand = Conv(
                c_mid,
                c_local,
                k=1,
                s=1,
                act=False,
            )

            self.beta = nn.Parameter(torch.zeros(1))

            nn.init.zeros_(
                self.reliability_gate[0].weight
            )
            nn.init.zeros_(
                self.reliability_gate[0].bias
            )

        self.alpha = nn.Parameter(torch.zeros(1))

        # 初始offset=0，不移动任何像素
        nn.init.zeros_(self.offset[-1].weight)
        nn.init.zeros_(self.offset[-1].bias)

        # 初始query约为0.5，但alpha=0，所以仍是恒等映射
        nn.init.zeros_(self.query_gate[0].weight)
        nn.init.zeros_(self.query_gate[0].bias)

    def _align_semantic(
        self,
        semantic: torch.Tensor,
        local_q: torch.Tensor,
    ) -> torch.Tensor:
        """Upsample and spatially align semantic features."""

        semantic = F.interpolate(
            semantic,
            size=local_q.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        semantic = self.semantic_refine(semantic)

        if self.max_offset == 0.0:
            return semantic

        offset_input = torch.cat(
            (local_q, semantic),
            dim=1,
        )

        # 单位为当前特征图上的像素
        offset = (
            self.max_offset
            * torch.tanh(self.offset(offset_input))
        )

        batch, _, height, width = semantic.shape

        yy, xx = torch.meshgrid(
            torch.linspace(
                -1.0,
                1.0,
                height,
                device=semantic.device,
                dtype=semantic.dtype,
            ),
            torch.linspace(
                -1.0,
                1.0,
                width,
                device=semantic.device,
                dtype=semantic.dtype,
            ),
            indexing="ij",
        )

        base_grid = torch.stack(
            (xx, yy),
            dim=-1,
        )

        base_grid = (
            base_grid
            .unsqueeze(0)
            .expand(batch, -1, -1, -1)
        )

        norm_x = (
            offset[:, 0]
            * (2.0 / max(width - 1, 1))
        )

        norm_y = (
            offset[:, 1]
            * (2.0 / max(height - 1, 1))
        )

        sampling_grid = (
            base_grid
            + torch.stack(
                (norm_x, norm_y),
                dim=-1,
            )
        )

        return F.grid_sample(
            semantic,
            sampling_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

    def forward(
        self,
        x: list[torch.Tensor] | tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError(
                "BSQFeedback expects two inputs: "
                "[local, semantic]."
            )

        local, semantic = x

        local_q = self.local_reduce(local)
        semantic_q = self.semantic_reduce(semantic)

        semantic_q = self._align_semantic(
            semantic_q,
            local_q,
        )

        difference = torch.abs(
            local_q - semantic_q
        )

        correlation = (
            local_q * semantic_q
        )

        query_statistics = torch.cat(
            (
                local_q.mean(
                    dim=1,
                    keepdim=True,
                ),
                semantic_q.mean(
                    dim=1,
                    keepdim=True,
                ),
                difference.mean(
                    dim=1,
                    keepdim=True,
                ),
                correlation.mean(
                    dim=1,
                    keepdim=True,
                ),
            ),
            dim=1,
        )

        # 目标语义查询图Q
        query = self.query_gate(
            query_statistics
        )

        semantic_value = self.semantic_expand(
            semantic_q
        )

        output = (
            local
            + self.alpha.tanh()
            * query
            * semantic_value
        )

        if self.use_detail:
            # 高频残差：原特征减去局部低通结果
            high_frequency = (
                local
                - F.avg_pool2d(
                    local,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    count_include_pad=False,
                )
            )

            detail_q = self.detail_reduce(
                high_frequency
            )

            detail_strength = (
                detail_q
                .abs()
                .mean(
                    dim=1,
                    keepdim=True,
                )
            )

            semantic_agreement = (
                correlation
                .mean(
                    dim=1,
                    keepdim=True,
                )
            )

            reliability_input = torch.cat(
                (
                    detail_strength,
                    semantic_agreement,
                ),
                dim=1,
            )

            reliability = self.reliability_gate(
                reliability_input
            )

            detail_value = self.detail_expand(
                self.detail_refine(detail_q)
            )

            output = (
                output
                + self.beta.tanh()
                * query
                * reliability
                * detail_value
            )

        return output