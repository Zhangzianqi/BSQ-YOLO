# Ultralytics AGPL-3.0 - https://ultralytics.com/license
"""CFPT adapted to a P2-P5 YOLO neck with a full-resolution P2 detail path.

CCA, CSA and cross-layer relative position encoding follow Du et al.,
Cross-Layer Feature Pyramid Transformer (TGRS 2025), https://github.com/duzw9311/CFPT.
Reference source: CFPT-main/mmdet/models/necks/cfpt.py (Apache-2.0).
See cfpt_LICENSE.txt for the upstream license.

Adaptation, NOT an exact reproduction: CCA -> CSA operates on P3/P4/P5 at
64 channels in the n model; P2 supplies space-to-depth detail to P3 and receives
spatially gated P3 context. Overlap reconstruction is averaged, Q/K/V windows
are partitioned separately, FFN expansion is 1, and residual updates are scaled.
Conv-BN projections/FFNs replace the reference's LayerNorm/Linear paths so
normalization can be folded into convolutions for batch-1 deployment.
Requires only PyTorch and the local Conv/Index modules (no MMDetection/timm).
WFSA and HRDP2Detect remain separate, unchanged modules in the YAML.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, Index

__all__ = ("CFPTP2Neck", "CFPTIndex")


class _CrossLayerPosition(nn.Module):
    """Upstream CFPT relative-coordinate interpolation, pre-indexed at init."""

    def __init__(self, windows, heads, spatial):
        super().__init__()
        coords = []
        for w in windows:
            axis = (torch.arange(w, dtype=torch.float32) - w // 2) * windows[0] / w
            if spatial:
                yy, xx = torch.meshgrid(axis, axis, indexing="ij")
                coords.append(torch.stack((yy.flatten(), xx.flatten())))
            else:
                coords.append(axis.unsqueeze(0))
        coords = torch.cat(coords, dim=1)
        relative = coords[:, :, None] - coords[:, None, :] + windows[0] - 1
        index = relative[0] * (2 * windows[0] - 1) + relative[1] if spatial else relative[0]
        size = (2 * windows[0] - 1) ** (2 if spatial else 1)
        self.table = nn.Parameter(torch.empty(size, heads))
        # Query/key level-pair bias is effective under softmax. A bias constant
        # across all keys in a query row (as in the reference) would cancel out.
        self.layer_bias = nn.Parameter(torch.zeros(len(windows) ** 2, heads))
        nn.init.trunc_normal_(self.table, std=0.02)
        self.register_buffer("index_floor", index.floor().long(), persistent=False)
        self.register_buffer("index_ceil", index.ceil().long(), persistent=False)
        self.register_buffer("fraction", (index - index.floor()).unsqueeze(-1), persistent=False)
        layers = torch.cat([torch.full((w ** (2 if spatial else 1),), i) for i, w in enumerate(windows)])
        self.register_buffer("layer_pairs", layers[:, None] * len(windows) + layers[None, :], persistent=False)

    def forward(self, dtype):
        bias = self.table[self.index_floor] * (1 - self.fraction) + self.table[self.index_ceil] * self.fraction
        bias = bias + self.layer_bias[self.layer_pairs]
        return bias.permute(2, 0, 1).unsqueeze(0).unsqueeze(2).to(dtype=dtype)


class _CosineAttention(nn.Module):
    """Explicit module boundary permits counting both attention matrix products."""

    def forward(self, q, k, v, bias):
        # eps=1e-6 remains representable in FP16, unlike normalize's 1e-12 default.
        q = F.normalize(q, dim=-1, eps=1e-6)
        k = F.normalize(k, dim=-1, eps=1e-6)
        scores = q @ k.transpose(-2, -1)
        return (scores + bias).softmax(dim=-1) @ v


class _CrossLayerAttention(nn.Module):
    """The three-level CFPT CCA or CSA, preserving native overlap grouping."""

    def __init__(self, channels, heads, reduction, mlp_ratio, residual_scale, spatial):
        super().__init__()
        residual_scale = float(residual_scale)
        self.spatial = spatial
        self.heads = heads
        self.hidden = channels // reduction
        if self.hidden < heads or self.hidden % heads:
            raise ValueError("CFPT hidden channels must be positive and divisible by num_heads.")
        self.head_dim = self.hidden // heads
        self.factors = (4, 2, 1)
        self.windows = (5, 3, 1) if spatial else (17, 5, 1)
        self.tokens = tuple(w * w if spatial else w for w in self.windows)
        self.position = _CrossLayerPosition(self.windows, heads, spatial)
        self.attention = _CosineAttention()
        self.cpe = nn.ModuleList(nn.Conv2d(channels, channels, 3, padding=1, groups=channels) for _ in range(3))
        self.qkv = nn.ModuleList(nn.Conv2d(channels, 3 * self.hidden, 1) for _ in range(3))
        self.proj = nn.ModuleList(Conv(self.hidden, channels, 1, act=False) for _ in range(3))
        self.ffn = nn.ModuleList(
            nn.Sequential(Conv(channels, int(channels * mlp_ratio), 1),
                          nn.Conv2d(int(channels * mlp_ratio), channels, 1))
            for _ in range(3)
        )
        # 4-D parameters also remain compatible with the project's MuSGD optimizer.
        self.attn_scale = nn.ParameterList(nn.Parameter(torch.full((1, channels, 1, 1), residual_scale)) for _ in range(3))
        self.ffn_scale = nn.ParameterList(nn.Parameter(torch.full((1, channels, 1, 1), residual_scale)) for _ in range(3))
        for i, (factor, window) in enumerate(zip(self.factors, self.windows)):
            if spatial:
                self.register_buffer(f"cover_{i}", torch.empty(0), persistent=False)
            else:
                length, stride = self.hidden * factor * factor, factor * factor
                pad = 0 if factor == 1 else 1
                ones = torch.ones(1, 1, length, 1)
                cover = F.fold(F.unfold(ones, (window, 1), padding=(pad, 0), stride=(stride, 1)),
                               (length, 1), (window, 1), padding=(pad, 0), stride=(stride, 1))
                self.register_buffer(f"cover_{i}", cover.clamp_min(1), persistent=False)

    @staticmethod
    def _spatial_coverage(height, width, window, stride, padding, reference):
        # Separable overlap counts, avoiding a large all-ones unfold/fold tensor.
        def count(length, step, pad):
            positions = torch.arange(length, device=reference.device)
            n = (length + 2 * pad - window) // step + 1
            lo = torch.div(positions + pad - window + step, step, rounding_mode="floor").clamp_min(0)
            hi = torch.div(positions + pad, step, rounding_mode="floor").clamp_max(n - 1)
            return (hi - lo + 1).clamp_min(1).to(dtype=reference.dtype)
        return count(height, stride, padding)[:, None] * count(width, stride, padding)[None, :]

    def forward(self, features):
        batch = features[0].shape[0]
        coarse_h, coarse_w = features[-1].shape[-2:]
        hw = coarse_h * coarse_w
        queries, keys, values, shortcuts = [], [], [], []
        for i, (x, factor, window) in enumerate(zip(features, self.factors, self.windows)):
            x = x + F.silu(self.cpe[i](x))
            shortcuts.append(x)
            qkv = self.qkv[i](x)
            if self.spatial:
                pad = 0 if factor == 1 else 1
                grouped = F.unfold(qkv, window, padding=pad, stride=factor)
                grouped = grouped.reshape(batch, 3, self.heads, self.head_dim, window * window, hw)
                grouped = grouped.permute(1, 0, 2, 5, 4, 3)
            else:
                # Split Q/K/V before channel unfolding: overlap cannot mix their boundaries.
                qkv = qkv.reshape(batch * 3, self.hidden, *x.shape[-2:])
                qkv = F.pixel_unshuffle(qkv, factor).flatten(2).transpose(1, 2).unsqueeze(-1)
                pad = 0 if factor == 1 else 1
                grouped = F.unfold(qkv, (window, 1), padding=(pad, 0), stride=(factor * factor, 1))
                grouped = grouped.reshape(batch, 3, hw, window, self.heads, self.head_dim)
                grouped = grouped.permute(1, 0, 4, 5, 3, 2)
            queries.append(grouped[0])
            keys.append(grouped[1])
            values.append(grouped[2])
        q, k, v = (torch.cat(parts, dim=-2) for parts in (queries, keys, values))
        out = self.attention(q, k, v, self.position(q.dtype))
        outputs = []
        for i, (part, factor, window, shortcut) in enumerate(zip(out.split(self.tokens, dim=-2), self.factors, self.windows, shortcuts)):
            height, width = shortcut.shape[-2:]
            pad = 0 if factor == 1 else 1
            if self.spatial:
                part = part.permute(0, 1, 4, 3, 2).reshape(batch, self.hidden * window * window, hw)
                part = F.fold(part, (height, width), window, padding=pad, stride=factor)
                # Coverage depends only on geometry, never on data or parameters.
                # Reuse it for fixed-resolution inference/training; recompute when
                # tracing/exporting so a traced graph does not depend on warmup.
                cover = getattr(self, f"cover_{i}")
                if (torch.jit.is_tracing() or torch.onnx.is_in_onnx_export()
                        or cover.shape != (height, width) or cover.device != part.device or cover.dtype != part.dtype):
                    # Keep cached constants usable by a later training forward,
                    # even if the cache was populated under inference_mode().
                    with torch.inference_mode(False):
                        cover = self._spatial_coverage(height, width, window, factor, pad, part)
                    if not torch.jit.is_tracing() and not torch.onnx.is_in_onnx_export():
                        setattr(self, f"cover_{i}", cover)
                part = part / cover
            else:
                part = part.permute(0, 4, 3, 1, 2).reshape(batch, hw * window, self.hidden)
                length = self.hidden * factor * factor
                part = F.fold(part, (length, 1), (window, 1), padding=(pad, 0), stride=(factor * factor, 1))
                part = part / getattr(self, f"cover_{i}")
                part = part.squeeze(-1).transpose(1, 2).reshape(batch, length, coarse_h, coarse_w)
                part = F.pixel_shuffle(part, factor)
            updated = shortcut + self.attn_scale[i] * self.proj[i](part)
            outputs.append(updated + self.ffn_scale[i] * self.ffn[i](updated))
        return outputs


class CFPTP2Neck(nn.Module):
    """[B2,B3,B4,B5] -> [P2,P3,P4,P5]; output channel widths follow YAML scaling.

    Args after parser injection: input widths, output widths, interaction width,
    attention heads, qkv reduction, FFN ratio, initial residual scale.
    P2 has a direct detail bypass; no pooling discards its original-resolution path.
    """

    def __init__(self, in_channels, out_channels, channels=64, heads=4, reduction=4, mlp_ratio=1.0, residual_scale=0.1):
        super().__init__()
        residual_scale = float(residual_scale)
        if len(in_channels) != 4 or len(out_channels) != 4:
            raise ValueError("CFPTP2Neck requires exactly four P2-P5 inputs and outputs.")
        if reduction < 1 or heads < 1 or channels % reduction or mlp_ratio <= 0 or residual_scale <= 0:
            raise ValueError("Invalid CFPT width/reduction/heads/FFN/residual settings.")
        self.out_channels = tuple(out_channels)
        self.interaction_channels = channels
        self.lateral = nn.ModuleList(Conv(c, channels, 1) for c in in_channels[1:])
        self.p2_down = nn.Sequential(nn.PixelUnshuffle(2), Conv(4 * in_channels[0], channels, 1))
        self.detail_scale = nn.Parameter(torch.full((1, channels, 1, 1), residual_scale))
        self.cca = _CrossLayerAttention(channels, heads, reduction, mlp_ratio, residual_scale, spatial=False)
        self.csa = _CrossLayerAttention(channels, heads, reduction, mlp_ratio, residual_scale, spatial=True)
        self.local = nn.ModuleList(Conv(ci, co, 1) if ci != co else nn.Identity() for ci, co in zip(in_channels, out_channels))
        self.out_proj = nn.ModuleList(Conv(channels, c, 1) for c in out_channels[1:])
        self.p2_context = Conv(channels, out_channels[0], 1)
        # A cheap spatial gate chooses where P3 context should update the P2 details.
        self.p2_gate = nn.Conv2d(out_channels[0] * 2, 1, 1)
        nn.init.zeros_(self.p2_gate.weight)
        nn.init.zeros_(self.p2_gate.bias)
        self.p2_scale = nn.Parameter(torch.full((1, out_channels[0], 1, 1), residual_scale))
        self.refine = nn.ModuleList(
            nn.Sequential(Conv(c, c, 3, g=c), Conv(c, c, 1)) for c in out_channels
        )
        self.refine_scale = nn.ParameterList(nn.Parameter(torch.full((1, c, 1, 1), residual_scale)) for c in out_channels)

    def forward(self, features):
        fine = features[0]
        multi = [proj(x) for proj, x in zip(self.lateral, features[1:])]
        multi[0] = multi[0] + self.detail_scale * self.p2_down(fine)
        multi = self.csa(self.cca(multi))
        context = F.interpolate(self.p2_context(multi[0]), size=fine.shape[-2:], mode="nearest")
        detail = self.local[0](fine)
        gate = self.p2_gate(torch.cat((detail, context), dim=1)).sigmoid()
        p2 = detail + self.p2_scale * gate * context
        outputs = [p2] + [local(x) + proj(y) for local, proj, x, y in zip(self.local[1:], self.out_proj, features[1:], multi)]
        return tuple(x + scale * refine(x) for x, scale, refine in zip(outputs, self.refine_scale, self.refine))


class CFPTIndex(Index):
    """Select one neck output; parser infers its scaled width from CFPTP2Neck."""
