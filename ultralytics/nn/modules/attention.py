# Ultralytics 🚀 AGPL-3.0 License - Light-weight Attention Modules for Small Object Detection
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["EMA", "SimAM", "CoordAtt"]


class EMA(nn.Module):
    """
    Efficient Multi-Scale Attention (EMA) Module.
    Paper: https://arxiv.org/abs/2305.13563
    保留高分辨率特征（不经过像素下采样），跨 Spatial 与 Channel 维度建立长程相关性，
    极其适合水面/小目标检测中的空域信息保留。
    参数量极低（只包含 1x1 降维/升维 Conv 权重）。
    """

    def __init__(self, channels: int, factor: int = 8):
        super().__init__()
        self.groups = factor
        assert channels % self.groups == 0, f"channels {channels} must be divisible by factor {factor}"
        self.softmax = nn.Softmax(dim=-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        group_ch = channels // self.groups
        self.gn = nn.GroupNorm(group_ch, group_ch)
        self.conv1x1 = nn.Conv2d(group_ch, group_ch, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(group_ch, group_ch, kernel_size=3, stride=1, padding=1, groups=group_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        group_ch = c // self.groups
        x_g = x.view(b * self.groups, group_ch, h, w)

        x_h = self.pool_h(x_g)
        x_w = self.pool_w(x_g).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)

        x1 = self.gn(x_g * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(x_g)

        x11 = self.softmax(self.agp(x1).view(b * self.groups, group_ch, -1).permute(0, 2, 1))
        x12 = x2.view(b * self.groups, group_ch, -1)
        out1 = torch.matmul(x11, x12).view(b * self.groups, 1, h, w)

        x21 = self.softmax(self.agp(x2).view(b * self.groups, group_ch, -1).permute(0, 2, 1))
        x22 = x1.view(b * self.groups, group_ch, -1)
        out2 = torch.matmul(x21, x22).view(b * self.groups, 1, h, w)

        weights = (out1 + out2).sigmoid()
        out = (x_g * weights).view(b, c, h, w)
        return out


class SimAM(nn.Module):
    """
    SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks (ICML 2021).
    Paper: http://proceedings.mlr.press/v139/yang21o/yang21o.pdf
    无参数（0 Parameter）注意力机制！基于三维神经元能量函数推导封闭解，计算 3D 显著性权重。
    对小目标高分辨率特征图非常安全，完全不占用模型权重量。
    """

    def __init__(self, channels: int = None, e_lambda: float = 1e-4):
        super().__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        n = h * w - 1
        d = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        v = d.sum(dim=[2, 3], keepdim=True) / n
        e_inv = d / (4 * (v + self.e_lambda)) + 0.5
        return x * self.activaton(e_inv)


class CoordAtt(nn.Module):
    """
    Coordinate Attention for Efficient Mobile Network Design (CVPR 2021).
    Paper: https://arxiv.org/abs/2103.02907
    沿着 X 轴和 Y 轴分别进行自适应池化，精准保留小目标的坐标和空间位置信息。
    参数量极低（由 1x1 卷积和 reduction 率控制）。
    """

    def __init__(self, channels: int, reduction: int = 32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()

        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        b, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        return identity * a_h * a_w


class CSGA(nn.Module):
    """
    Cross-Scale Semantic Gated Attention (CSGA) 跨尺度语义门控注意力.
    专为水面小目标设计的不落俗套注意力机制：
    普通注意力只在单一特征层内部算空间/通道关系，容易把水面波浪反光误认成目标。
    CSGA 利用深层特征 (P4/P5, 富含目标语义) 生成空间掩码，去“门控过滤”浅层高分辨率 P2 (富含几何细节与高频波浪杂波)。
    通过深层语义指导浅层去噪，且自带 Alpha 残差控制，绝对保证训练稳定与涨点安全性！
    """

    def __init__(self, c_shallow: int, c_deep: int, reduction: int = 4):
        super().__init__()
        c_hidden = max(8, c_shallow // reduction)
        # 深层特征语义提取器
        self.deep_gate = nn.Sequential(
            nn.Conv2d(c_deep, c_hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_hidden, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        # 浅层特征局部上下文提纯
        self.shallow_conv = nn.Sequential(
            nn.Conv2d(c_shallow, c_shallow, kernel_size=3, padding=1, groups=c_shallow, bias=False),
            nn.BatchNorm2d(c_shallow),
            nn.SiLU(inplace=True)
        )
        # 残差平衡标量 (可学习Alpha，初始为0.5，确保初始状态稳健不崩)
        self.gamma = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: list) -> torch.Tensor:
        """
        x[0]: x_shallow (如 P2, shape: [B, C1, H1, W1])
        x[1]: x_deep (如 P4/P5, shape: [B, C2, H2, W2])
        """
        x_shallow, x_deep = x[0], x[1]
        
        # 1. 提炼深层语义门控权重
        gate = self.deep_gate(x_deep)
        if gate.shape[2:] != x_shallow.shape[2:]:
            gate = F.interpolate(gate, size=x_shallow.shape[2:], mode="bilinear", align_corners=False)
            
        # 2. 浅层局部细节抽取
        feat_shallow = self.shallow_conv(x_shallow)
        
        # 3. 深层语义门控过滤浅层杂波 + 残差连接保护机制
        gated_feat = feat_shallow * gate
        return x_shallow + self.gamma * gated_feat

