# ultralytics/nn/modules/acfi.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv

class LayerFeatureAggregation(nn.Module):
    """LFA：全局最大池化与平均池化联合提取特征摘要向量"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        c_hidden = max(channels // reduction, 8)
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, c_hidden, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(c_hidden, channels, bias=False)
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        max_pool = F.adaptive_max_pool2d(x, (1, 1)).view(b, c)
        avg_pool = F.adaptive_avg_pool2d(x, (1, 1)).view(b, c)
        return self.act(self.shared_mlp(max_pool) + self.shared_mlp(avg_pool))

class ACFI(nn.Module):
    """ACFI 跨层信息交互模块：加在 Backbone ↔ Neck 界面"""
    def __init__(self, c1_list: list, c2: int, target_idx: int = 0, num_heads: int = 4):
        super().__init__()
        self.target_idx = target_idx
        num_inputs = len(c1_list)

        self.conv_align = nn.ModuleList([
            Conv(c_in, c2, 1, 1) if c_in != c2 else nn.Identity()
            for c_in in c1_list
        ])
        self.lfas = nn.ModuleList([LayerFeatureAggregation(c2) for _ in range(num_inputs)])

        num_heads = min(num_heads, c2)
        while c2 % num_heads != 0:
            num_heads -= 1
            
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=c2, nhead=num_heads, dim_feedforward=c2 * 2,
            dropout=0.0, activation="gelu", batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.conv_fuse = Conv(c2 * num_inputs, c2, 1, 1)

    def forward(self, x: list) -> torch.Tensor:
        target_h, target_w = x[self.target_idx].shape[2:]

        aligned_x = []
        for i, feat in enumerate(x):
            feat_aligned = self.conv_align[i](feat)
            if feat_aligned.shape[2:] != (target_h, target_w):
                feat_aligned = F.interpolate(feat_aligned, size=(target_h, target_w), mode="bilinear", align_corners=False)
            aligned_x.append(feat_aligned)

        lfa_vectors = [self.lfas[i](aligned_x[i]) for i in range(len(x))]
        vector_seq = torch.stack(lfa_vectors, dim=1)
        attn_seq = self.transformer_encoder(vector_seq)

        enhanced_feats = []
        for i in range(len(x)):
            weight = torch.sigmoid(attn_seq[:, i, :]).unsqueeze(-1).unsqueeze(-1)
            feat_enhanced = aligned_x[i] * weight + aligned_x[i]
            enhanced_feats.append(feat_enhanced)

        return self.conv_fuse(torch.cat(enhanced_feats, dim=1))

class ASFF(nn.Module):
    """ASFF 空间自适应融合模块（轻量瘦身版）"""
    def __init__(self, c1_list: list, c2: int, target_idx: int = 0):
        super().__init__()
        self.target_idx = target_idx
        num_inputs = len(c1_list)

        self.conv_align = nn.ModuleList([
            Conv(c_in, c2, 1, 1) if c_in != c2 else nn.Identity()
            for c_in in c1_list
        ])
        self.weight_convs = nn.ModuleList([
            nn.Conv2d(c2, 1, kernel_size=1) for _ in range(num_inputs)
        ])
        # 改动点：改用 1x1 卷积提取输出，在 160x160 分辨率下可节省大量计算量
        self.conv_out = Conv(c2, c2, 1, 1)

    def forward(self, x: list) -> torch.Tensor:
        target_h, target_w = x[self.target_idx].shape[2:]

        aligned_x = []
        for i, feat in enumerate(x):
            feat_aligned = self.conv_align[i](feat)
            if feat_aligned.shape[2:] != (target_h, target_w):
                feat_aligned = F.interpolate(feat_aligned, size=(target_h, target_w), mode="bilinear", align_corners=False)
            aligned_x.append(feat_aligned)

        weight_maps = [self.weight_convs[i](aligned_x[i]) for i in range(len(x))]
        weights = F.softmax(torch.cat(weight_maps, dim=1), dim=1)

        fused = sum(weights[:, i : i + 1, :, :] * aligned_x[i] for i in range(len(x)))
        return self.conv_out(fused)