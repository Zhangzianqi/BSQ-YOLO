"""Two-input BiFPN-style fusion adapted to the existing P2 interface.

Reference: EfficientDet (CVPR 2020), fast normalized feature fusion.
This is a PyTorch adaptation of one fusion node, not a complete BiFPN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("BiFPNFuseP2",)


class BiFPNFuseP2(nn.Module):
    def __init__(self, c_semantic, c_detail, c_out, eps=1e-4):
        super().__init__()
        if min(c_semantic, c_detail, c_out) <= 0 or eps <= 0:
            raise ValueError("Channels and eps must be positive.")
        self.eps = float(eps)

        def project(c_in):
            if c_in == c_out:
                return nn.Identity()
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, bias=False),
                nn.BatchNorm2d(c_out),
            )

        self.semantic_proj = project(c_semantic)
        self.detail_proj = project(c_detail)
        self.fusion_weights = nn.Parameter(torch.ones(2))
        self.act = nn.SiLU()
        self.dw = nn.Conv2d(c_out, c_out, 3, padding=1,
                            groups=c_out, bias=False)
        self.pw = nn.Conv2d(c_out, c_out, 1, bias=True)
        self.bn = nn.BatchNorm2d(c_out)

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError("BiFPNFuseP2 expects [upsampled_P3, B2_detail].")
        semantic, detail = x
        if semantic.shape[0] != detail.shape[0] or semantic.shape[-2:] != detail.shape[-2:]:
            raise ValueError("Both inputs must have matching batch and spatial sizes.")

        semantic = self.semantic_proj(semantic)
        detail = self.detail_proj(detail)
        weights = F.relu(self.fusion_weights.float())
        weights = (weights / (weights.sum() + self.eps)).to(semantic.dtype)
        fused = weights[0] * semantic + weights[1] * detail
        return self.bn(self.pw(self.dw(self.act(fused))))
