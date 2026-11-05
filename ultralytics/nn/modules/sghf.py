from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv

__all__ = ("SGHFP2",)


class SGHFP2(nn.Module):
    """
    Semantic-Guided High-Frequency Selection for P2 fusion.

    Inputs:
        x[0]: P2 detail feature,          [B, C_detail, 160, 160]
        x[1]: upsampled P3 semantic feat, [B, C_sem,    160, 160]

    Output:
        concat([P3 semantic, refined P2]),
        with the same shape as the original Concat.
    """

    def __init__(
        self,
        c_detail: int,
        c_semantic: int,
        kernel_size: int = 3,
    ):
        super().__init__()

        # 将P3语义通道对齐到P2通道，仅用于生成门控
        self.semantic_proj = Conv(c_semantic, c_detail, k=1, s=1)

        # [P2低频, P3语义] -> 高频选择门
        self.gate_reduce = Conv(c_detail * 2, c_detail, k=1, s=1)

        # 最后一层不使用激活，之后直接接Sigmoid
        self.gate_dw = nn.Conv2d(
            c_detail,
            c_detail,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=c_detail,
            bias=True,
        )

        # 通道级残差强度，初始化为0
        # 初始状态下模块严格等价于原来的Concat
        self.alpha = nn.Parameter(
            torch.zeros(1, c_detail, 1, 1)
        )

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        detail, semantic = x

        # 正常YAML下二者都是160×160；保留尺寸保护
        if semantic.shape[-2:] != detail.shape[-2:]:
            semantic = F.interpolate(
                semantic,
                size=detail.shape[-2:],
                mode="nearest",
            )

        # 1. P2局部低频
        low = F.avg_pool2d(
            detail,
            kernel_size=3,
            stride=1,
            padding=1,
            count_include_pad=False,
        )

        # 2. P2高频残差
        high = detail - low

        # 3. P3语义投影
        semantic_guidance = self.semantic_proj(semantic)

        # 4. 生成逐位置、逐通道门控
        gate_input = torch.cat((low, semantic_guidance), dim=1)
        gate = torch.sigmoid(
            self.gate_dw(self.gate_reduce(gate_input))
        )

        # 5. 双向高频筛选
        # gate≈0：抑制高频
        # gate≈0.5：保持原特征
        # gate≈1：增强高频
        alpha = torch.tanh(self.alpha)
        refined_detail = (
            detail
            + alpha * (2.0 * gate - 1.0) * high
        )

        # 保持和原Concat一致的通道顺序：
        # [上采样P3语义, P2细节]
        return torch.cat((semantic, refined_detail), dim=1)