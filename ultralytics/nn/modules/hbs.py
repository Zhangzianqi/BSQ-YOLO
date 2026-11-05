# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
HBS: Hierarchical Background Smoothing (SET, CVPR 2025) 的 YOLO11 移植。

论文《SET: Spectral Enhancement for Tiny Object Detection》通过频域遮挡归因发现：
FPN 特征上的高频响应中，背景高频（水面反光/涟漪等）是噪声源，压制背景高频可
稳定提升小目标检测。HBS 是其核心模块：只对背景区域做"通道压缩-扩张"的自适应
平滑（公式 3），前景保持原样；卷积核大小随 FPN 层 stride 递增（公式 4）。

本文件提供两个变体：
- HBS      : 原文版本。forward 需要 GT 框生成的二值 mask（Phase A 用，训练期辅助分支）。
- HBSLite  : 无 mask 弱化版（Phase B 用）。用 1x1 conv + sigmoid 学一个软前景 mask，
             按 M·P + (1-M)·φ(P) 门控融合，可端到端训练、训推同用。

与论文的差异（Phase B 写论文时要主动交代）：
1. mask 来源：原文用 GT 框硬 mask（监督），HBSLite 用特征自学习软 mask（无监督）；
2. 使用方式：原文仅训练期辅助分支、推理零开销，HBSLite 内联进 neck、训推同用；
3. φ 轻量化：原文 φ 是两枚 K×K 全通道卷积（移植到 YOLO11s 时 P5 层 1024 通道
   导致参数量 +30M、计算量 +28G），HBSLite 改为深度可分离版——
   1×1 压缩 (C→C/4) → ReLU → depthwise K×K → ReLU → 1×1 扩张 (C/4→C) + 残差。
   通道压缩瓶颈与空间平滑解耦，参数/计算量降约 97%，语义不变（轻量适配消融）。
4. 保留项：通道压缩率 r=4（公式 3）、核大小随 stride 递增 K=3/5/5/7（公式 4）。
"""

import math

import torch
import torch.nn as nn


class _AdaptiveSmoothing(nn.Module):
    """公式 (3) 的自适应平滑算子 φ（原文版）：压缩-激活-扩张 + 残差。

    φ(x) = σ(w^e ⊗ σ(w^r ⊗ x)) + x
    w^r: C → C/r 的通道压缩卷积（表达能力瓶颈，迫使网络丢掉不重要的高频细节）
    w^e: C/r → C 的通道扩张卷积
    + x: 残差连接，保证背景低频结构不丢，只学"平滑修正量"（ResNet 思路）
    """

    def __init__(self, channels: int, reduction: int = 4, kernel: int = 3):
        super().__init__()
        c_hidden = max(channels // reduction, 1)
        pad = (kernel - 1) // 2
        self.reduce = nn.Conv2d(channels, c_hidden, kernel, padding=pad, bias=False)
        self.expand = nn.Conv2d(c_hidden, channels, kernel, padding=pad, bias=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, h, w]，输出与输入同 shape。"""
        return self.expand(self.act(self.reduce(x))) + x


class _AdaptiveSmoothingLite(nn.Module):
    """公式 (3) 的深度可分离轻量版（Phase B 用）。

    1×1 压缩 (C→C/4) → ReLU → depthwise K×K → ReLU → 1×1 扩张 (C/4→C) → +x

    保留原文两个核心语义：
    - 通道压缩瓶颈（C→C/4 再扩张）→ 迫使网络丢弃次要高频（JPEG 类比）；
    - 残差 +x → 背景低频结构不丢，只学"平滑修正量"。
    仅把 K×K 空间卷积换成逐通道 depthwise（groups=C/4），参数量与计算量降 ~97%，
    避免 P5 层（1024 通道 × 7×7 核）在 YOLO11s 上参数量爆炸。
    """

    def __init__(self, channels: int, reduction: int = 4, kernel: int = 3):
        super().__init__()
        c_hidden = max(channels // reduction, 1)
        pad = (kernel - 1) // 2
        self.reduce = nn.Conv2d(channels, c_hidden, 1, bias=False)          # 通道压缩
        self.smooth = nn.Conv2d(c_hidden, c_hidden, kernel, padding=pad,
                                groups=c_hidden, bias=False)                # depthwise 空间平滑
        self.expand = nn.Conv2d(c_hidden, channels, 1, bias=False)          # 通道扩张
        self.act = nn.ReLU(inplace=True)
        # ★ 关键修复：expand 零初始化 → 残差项起步为 0 → φ(P) ≈ P
        #   P^E = 0.5P + 0.5·P = P，HBSLite 起步退化为恒等，与预训练模型行为一致
        nn.init.zeros_(self.expand.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, h, w]，输出与输入同 shape。"""
        out = self.act(self.reduce(x))     # C → C/4
        out = self.act(self.smooth(out))   # C/4 → C/4
        return self.expand(out) + x        # C/4 → C；残差加原始输入 x（不能被覆盖）


class HBS(nn.Module):
    """原文 HBS（Phase A 用）：前/背景解耦，只平滑背景。

    P^E = P ⊙ M + φ(P ⊙ ¬M)   （公式 2）
    依赖 GT 框 mask，仅训练期调用；推理时由辅助分支整体裁掉。
    """

    def __init__(self, channels: int, stride: int, reduction: int = 4):
        super().__init__()
        k = self._kernel_size_from_stride(stride)
        self.phi = _AdaptiveSmoothing(channels, reduction, k)

    @staticmethod
    def _kernel_size_from_stride(stride: int) -> int:
        """公式 (4): K_i = ceil(log2(S_i)/2) * 2 + 1，保证奇数、padding 对称。"""
        return math.ceil(math.log2(stride) / 2) * 2 + 1

    def forward(self, P: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """P: [B, C, h, w] FPN 特征；M: [B, 1, h, w] 二值 mask（1=前景）。"""
        fg = P * M                    # P ⊙ M：只留前景
        bg = P * (1.0 - M)            # P ⊙ ¬M：只留背景
        return fg + self.phi(bg)      # P^E


class HBSLite(nn.Module):
    """Phase B 无 mask 版：1×1 conv 学软 mask，按门控融合平滑结果。

    M_soft = σ(w^m ⊗ P)                      （学出来的软前景 mask，0~1）
    P^E = M_soft ⊙ P + (1 - M_soft) ⊙ φ(P)   （门控融合，同公式 2 结构）

    φ 用深度可分离轻量版 _AdaptiveSmoothingLite（见上）；核大小 K_i（公式 4）
    与通道压缩率 r=4（公式 3）和原文一致。
    注：mask_conv 与 φ.expand 均零初始化 → 起步 P^E ≈ P（恒等），
    mask 从"全图一视同仁"(≈0.5) 起步，靠梯度把前景位置学成 1。
    """

    def __init__(self, channels: int, stride: int, reduction: int = 4):
        super().__init__()
        k = self._kernel_size_from_stride(stride)
        self.phi = _AdaptiveSmoothingLite(channels, reduction, k)
        self.mask_conv = nn.Conv2d(channels, 1, 1, bias=False)
        # 从"全图等权"起步：mask 初始 ≈ 0.5，φ 支路初始 ≈ P（残差+零初始化）
        nn.init.zeros_(self.mask_conv.weight)

    @staticmethod
    def _kernel_size_from_stride(stride: int) -> int:
        """公式 (4): K_i = ceil(log2(S_i)/2) * 2 + 1。"""
        return math.ceil(math.log2(stride) / 2) * 2 + 1

    def forward(self, P: torch.Tensor) -> torch.Tensor:
        """P: [B, C, h, w] FPN 特征，输出同 shape。训练/推理同一路径。"""
        M = torch.sigmoid(self.mask_conv(P))      # 软前景 mask [B, 1, h, w]
        return M * P + (1.0 - M) * self.phi(P)    # 门控融合