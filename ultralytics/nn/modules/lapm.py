from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ("LAPM",)


class LAPM(nn.Module):
    """
    Parameter-free Local Attention Pyramid Module.

    Implementation follows ESL-YOLO Equations (3)-(7):
        C_p = Sigmoid(PatchIn(F_p, p))
        F_{p-1} = alpha * F_p
                  + (1 - alpha) * C_p * F_p

    Args:
        p: Maximum pyramid level. p=2 means 2 -> 1 -> 0.
        alpha: Fixed residual weighting factor.
        eps: Numerical stability term for patch normalization.
    """

    def __init__(
        self,
        p: int = 2,
        alpha: float = 0.5,
        eps: float = 1e-5,
    ):
        super().__init__()

        self.p = int(p)
        self.alpha = float(alpha)
        self.eps = float(eps)

        if self.p < 0:
            raise ValueError(f"p must be >= 0, but got {p}")

        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(
                f"alpha must be in [0, 1], but got {alpha}"
            )

        if self.eps <= 0.0:
            raise ValueError(
                f"eps must be > 0, but got {eps}"
            )

    def _make_attention(
        self,
        x: torch.Tensor,
        p: int,
    ) -> torch.Tensor:
        """
        Divide every channel into 2^p × 2^p patches,
        independently normalize each patch, and apply Sigmoid.
        """
        b, c, h, w = x.shape

        # p=2 -> divide height and width into 4 parts:
        # a total of 4 × 4 = 16 patches per channel.
        grid = 1 << p

        if h % grid != 0 or w % grid != 0:
            raise ValueError(
                f"LAPM level p={p} requires H and W "
                f"divisible by {grid}, but received {(h, w)}"
            )

        patch_h = h // grid
        patch_w = w // grid

        # BCHW
        # -> B, C, grid_h, patch_h, grid_w, patch_w
        # -> B, C, grid_h, grid_w, patch_h, patch_w
        patches = (
            x.reshape(
                b,
                c,
                grid,
                patch_h,
                grid,
                patch_w,
            )
            .permute(0, 1, 2, 4, 3, 5)
        )

        # Calculate normalization statistics in FP32 when AMP is used.
        stats = (
            patches.float()
            if patches.dtype in (torch.float16, torch.bfloat16)
            else patches
        )

        # Independent instance normalization for every
        # sample, channel and patch.
        mean = stats.mean(
            dim=(-2, -1),
            keepdim=True,
        )

        centered = stats - mean

        # Equivalent to biased variance used by InstanceNorm.
        variance = centered.square().mean(
            dim=(-2, -1),
            keepdim=True,
        )

        normalized = centered * torch.rsqrt(
            variance + self.eps
        )

        attention = torch.sigmoid(normalized).to(
            dtype=x.dtype
        )

        # Recombine patches into the complete BCHW map.
        attention = (
            attention.permute(0, 1, 2, 4, 3, 5)
            .reshape(b, c, h, w)
        )

        return attention

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"LAPM expects BCHW input, "
                f"but got shape {tuple(x.shape)}"
            )

        out = x

        # ESL-YOLO: p=2 -> p=1 -> p=0
        for level in range(self.p, -1, -1):
            attention = self._make_attention(
                out,
                level,
            )

            out = (
                self.alpha * out
                + (1.0 - self.alpha)
                * attention
                * out
            )

        return out