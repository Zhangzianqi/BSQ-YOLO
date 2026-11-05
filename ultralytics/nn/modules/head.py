# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Model head modules."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_

from ultralytics.utils import NOT_MACOS14
from ultralytics.utils.tal import dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import TORCH_1_11, fuse_conv_and_bn, smart_inference_mode

from .block import DFL, SAVPE, BNContrastiveHead, ContrastiveHead, Proto, Proto26, RealNVP, Residual, SwiGLUFFN, RepVGGDW
from .conv import Conv, DWConv
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from .utils import bias_init_with_prob, linear_init

__all__ = "OBB", "Classify", "Detect", "Pose", "RTDETRDecoder", "Segment", "YOLOEDetect", "YOLOESegment", "v10Detect", "FGHTDSIDetect", "HCFDDetect", "SBAHDetect", "SGSDetect", "HRDP2Detect", "SRDHRDDetect", "DSTSP2Detect", "SBAHHRDDetect"


class Detect(nn.Module):
    """YOLO Detect head for object detection models.

    This class implements the detection head used in YOLO models for predicting bounding boxes and class probabilities.
    It supports both training and inference modes, with optional end-to-end detection capabilities.

    Attributes:
        dynamic (bool): Force grid reconstruction.
        export (bool): Export mode flag.
        format (str): Export format.
        end2end (bool): End-to-end detection mode.
        max_det (int): Maximum detections per image.
        shape (tuple): Input shape.
        anchors (torch.Tensor): Anchor points.
        strides (torch.Tensor): Feature map strides.
        legacy (bool): Backward compatibility for v3/v5/v8/v9/v11 models.
        xyxy (bool): Output format, xyxy or xywh.
        nc (int): Number of classes.
        nl (int): Number of detection layers.
        reg_max (int): DFL channels.
        no (int): Number of outputs per anchor.
        stride (torch.Tensor): Strides computed during build.
        cv2 (nn.ModuleList): Convolution layers for box regression.
        cv3 (nn.ModuleList): Convolution layers for classification.
        dfl (nn.Module): Distribution Focal Loss layer.
        one2one_cv2 (nn.ModuleList): One-to-one convolution layers for box regression.
        one2one_cv3 (nn.ModuleList): One-to-one convolution layers for classification.

    Methods:
        forward: Perform forward pass and return predictions.
        bias_init: Initialize detection head biases.
        decode_bboxes: Decode bounding boxes from predictions.
        postprocess: Post-process model predictions.

    Examples:
        Create a detection head for 80 classes
        >>> detect = Detect(nc=80, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = detect(x)
    """

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    format = None  # export format
    max_det = 300  # max_det
    agnostic_nms = False
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    legacy = False  # backward compatibility for v3/v5/v8/v9 models
    xyxy = False  # xyxy or xywh output

    def __init__(self, nc: int = 80, reg_max=16, end2end=False, ch: tuple = ()):
        """Initialize the YOLO detection layer with specified number of classes and channels.

        Args:
            nc (int): Number of classes.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = reg_max  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch
            )
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for v3/v5/v8/v9/v11 backward compatibility."""
        return dict(box_head=self.cv2, cls_head=self.cv3)

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3)

    @property
    def end2end(self):
        """Checks if the model has one2one for v3/v5/v8/v9/v11 backward compatibility."""
        return getattr(self, "_end2end", True) and hasattr(self, "one2one")

    @end2end.setter
    def end2end(self, value):
        """Override the end-to-end detection mode."""
        self._end2end = value

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module = None, cls_head: torch.nn.Module = None
    ) -> dict[str, torch.Tensor]:
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if box_head is None or cls_head is None:  # for fused inference
            return dict()
        bs = x[0].shape[0]  # batch size
        boxes = torch.cat([box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        scores = torch.cat([cls_head[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
        return dict(boxes=boxes, scores=scores, feats=x)

    def forward(
        self, x: list[torch.Tensor]
    ) -> dict[str, torch.Tensor] | torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        preds = self.forward_head(x, **self.one2many)
        if self.end2end:
            x_detach = [xi.detach() for xi in x]
            one2one = self.forward_head(x_detach, **self.one2one)
            preds = {"one2many": preds, "one2one": one2one}
        if self.training:
            return preds
        y = self._inference(preds["one2one"] if self.end2end else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps.

        Args:
            x (dict[str, torch.Tensor]): Dictionary of predictions from detection layers.

        Returns:
            (torch.Tensor): Concatenated tensor of decoded bounding boxes and class probabilities.
        """
        # Inference path
        dbox = self._get_decode_boxes(x)
        return torch.cat((dbox, x["scores"].sigmoid()), 1)

    def _get_decode_boxes(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Get decoded boxes based on anchors and strides."""
        shape = x["feats"][0].shape  # BCHW
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (a.transpose(0, 1) for a in make_anchors(x["feats"], self.stride, 0.5))
            self.shape = shape

        dbox = self.decode_bboxes(self.dfl(x["boxes"]), self.anchors.unsqueeze(0)) * self.strides
        return dbox

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        for i, (a, b) in enumerate(zip(self.one2many["box_head"], self.one2many["cls_head"])):  # from
            a[-1].bias.data[:] = 2.0  # box
            b[-1].bias.data[: self.nc] = math.log(
                5 / self.nc / (640 / self.stride[i]) ** 2
            )  # cls (.01 objects, 80 classes, 640 img)
        if self.end2end:
            for i, (a, b) in enumerate(zip(self.one2one["box_head"], self.one2one["cls_head"])):  # from
                a[-1].bias.data[:] = 2.0  # box
                b[-1].bias.data[: self.nc] = math.log(
                    5 / self.nc / (640 / self.stride[i]) ** 2
                )  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor, xywh: bool = True) -> torch.Tensor:
        """Decode bounding boxes from predictions."""
        return dist2bbox(
            bboxes,
            anchors,
            xywh=xywh and not self.end2end and not self.xyxy,
            dim=1,
        )

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Post-processes YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x1, y1, x2, y2, class_probs].

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x1, y1, x2, y2, max_class_prob, class_index].
        """
        boxes, scores = preds.split([4, self.nc], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        return torch.cat([boxes, scores, conf], dim=-1)

    def get_topk_index(self, scores: torch.Tensor, max_det: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get top-k indices from scores.

        Args:
            scores (torch.Tensor): Scores tensor with shape (batch_size, num_anchors, num_classes).
            max_det (int): Maximum detections per image.

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor): Top scores, class indices, and filtered indices.
        """
        batch_size, anchors, nc = scores.shape  # i.e. shape(16,8400,84)
        # Use max_det directly during export for TensorRT compatibility (requires k to be constant),
        # otherwise use min(max_det, anchors) for safety with small inputs during Python inference
        k = max_det if self.export else min(max_det, anchors)
        if self.agnostic_nms:
            scores, labels = scores.max(dim=-1, keepdim=True)
            scores, indices = scores.topk(k, dim=1)
            labels = labels.gather(1, indices)
            return scores, labels, indices
        ori_index = scores.max(dim=-1)[0].topk(k)[1].unsqueeze(-1)
        scores = scores.gather(dim=1, index=ori_index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(k)
        idx = ori_index[torch.arange(batch_size)[..., None], index // nc]  # original index
        return scores[..., None], (index % nc)[..., None].float(), idx

    def fuse(self) -> None:
        """Remove the one2many head for inference optimization."""
        self.cv2 = self.cv3 = None

class _FGHTDSIInjection(nn.Module):
    """
    Parameter-efficient Fine-Grained HTDSI injection.

    Args:
        c_detail: P2/P3检测特征通道数。
        c_semantic: P3-TD/P4-TD语义特征通道数。
        reduction: 通道压缩比例，默认8。
        use_fine_detail: 是否启用3×3、5×5细粒度几何分支。
    """

    def __init__(
        self,
        c_detail: int,
        c_semantic: int,
        reduction: int = 8,
        use_fine_detail: bool = True,
    ):
        super().__init__()

        # 压缩交互通道。
        # s模型中：
        # P2: 128 -> 16
        # P3: 256 -> 32
        c_mid = max(c_detail // reduction, 16)
        c_gate = max(c_mid // 2, 8)

        self.use_fine_detail = use_fine_detail

        # -------------------------------------------------
        # 1. 高级语义压缩
        # 先在低分辨率特征上压缩通道，然后再上采样，
        # 比先扩展到c_detail通道再上采样更省计算量。
        # -------------------------------------------------
        self.semantic_reduce = Conv(
            c_semantic,
            c_mid,
            k=1,
            s=1,
        )

        # 上采样后进行轻量空间校正
        self.semantic_refine = DWConv(
            c_mid,
            c_mid,
            k=3,
            s=1,
            act=False,
        )

        # 将压缩语义恢复到目标特征通道
        self.semantic_expand = Conv(
            c_mid,
            c_detail,
            k=1,
            s=1,
            act=False,
        )

        # -------------------------------------------------
        # 2. P2/P3细节特征压缩
        # -------------------------------------------------
        self.detail_reduce = Conv(
            c_detail,
            c_mid,
            k=1,
            s=1,
        )

        # -------------------------------------------------
        # 3. 跨尺度一致性空间门控
        #
        # 输入只有4张统计图：
        # detail、semantic、difference、correlation
        #
        # 避免在高分辨率P2上直接拼接4组完整通道。
        # -------------------------------------------------
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.Sigmoid(),
        )

        # -------------------------------------------------
        # 4. 轻量通道门控
        # -------------------------------------------------
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                c_mid,
                c_gate,
                kernel_size=1,
                bias=True,
            ),
            nn.SiLU(),
            nn.Conv2d(
                c_gate,
                c_detail,
                kernel_size=1,
                bias=True,
            ),
            nn.Sigmoid(),
        )

        # 高级语义残差系数
        # alpha=0时严格退化为原始Detect
        self.alpha = nn.Parameter(
            torch.zeros(1, 1, 1, 1)
        )

        # -------------------------------------------------
        # 5. 细粒度几何分支
        #
        # 3×3：瓶盖、瓶口、罐口等局部结构
        # 5×5：整体轮廓、长宽比、侧壁形态
        # -------------------------------------------------
        if self.use_fine_detail:
            self.detail_conv3 = DWConv(
                c_mid,
                c_mid,
                k=3,
                s=1,
                act=False,
            )

            self.detail_conv5 = DWConv(
                c_mid,
                c_mid,
                k=5,
                s=1,
                act=False,
            )

            # 将压缩后的几何细节恢复至原通道数
            self.detail_expand = Conv(
                c_mid,
                c_detail,
                k=1,
                s=1,
                act=False,
            )

            # 细粒度残差系数
            self.beta = nn.Parameter(
                torch.zeros(1, 1, 1, 1)
            )

        # 初始门控值约为0.5，避免Sigmoid一开始饱和
        nn.init.zeros_(self.spatial_gate[0].weight)
        nn.init.zeros_(self.spatial_gate[0].bias)

        nn.init.zeros_(self.channel_gate[3].weight)
        nn.init.zeros_(self.channel_gate[3].bias)

    def forward(
        self,
        detail: torch.Tensor,
        semantic: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            detail:
                P2或P3检测特征。

            semantic:
                P3-TD或P4-TD高级语义。

        Returns:
            增强后的分类特征。
        """

        # -------------------------------------------------
        # 1. 高级语义对齐
        # -------------------------------------------------

        # 在低分辨率上压缩通道
        semantic_q = self.semantic_reduce(semantic)

        # 使用size而不是固定scale_factor，
        # 兼容矩形输入和奇数尺寸
        semantic_q = F.interpolate(
            semantic_q,
            size=detail.shape[-2:],
            mode="nearest",
        )

        semantic_q = self.semantic_refine(semantic_q)

        # 用于最终语义残差
        semantic_value = self.semantic_expand(semantic_q)

        # -------------------------------------------------
        # 2. 细节特征压缩
        # -------------------------------------------------
        detail_q = self.detail_reduce(detail)

        # -------------------------------------------------
        # 3. 跨尺度一致性线索
        # -------------------------------------------------
        difference = torch.abs(detail_q - semantic_q)
        correlation = detail_q * semantic_q

        spatial_statistics = torch.cat(
            (
                detail_q.mean(dim=1, keepdim=True),
                semantic_q.mean(dim=1, keepdim=True),
                difference.mean(dim=1, keepdim=True),
                correlation.mean(dim=1, keepdim=True),
            ),
            dim=1,
        )

        spatial_weight = self.spatial_gate(
            spatial_statistics
        )

        channel_weight = self.channel_gate(
            correlation
        )

        # B×C×H×W
        support_weight = (
            spatial_weight * channel_weight
        )

        # -------------------------------------------------
        # 4. 高级语义残差
        #
        # 主要解决：
        # 垃圾与反光、波纹、泡沫之间的错误识别
        # -------------------------------------------------
        output = (
            detail
            + self.alpha.tanh()
            * support_weight
            * semantic_value
        )

        # -------------------------------------------------
        # 5. 细粒度几何残差
        #
        # 主要解决：
        # 瓶子与易拉罐之间的细粒度混淆
        # -------------------------------------------------
        if self.use_fine_detail:
            detail_3x3 = self.detail_conv3(detail_q)
            detail_5x5 = self.detail_conv5(detail_q)

            # 同时保留局部结构、整体轮廓及尺度差异
            fine_detail = (
                detail_3x3
                + detail_5x5
                + torch.abs(detail_3x3 - detail_5x5)
            )

            fine_detail = self.detail_expand(
                fine_detail
            )

            # 细节增强也必须经过语义门控，
            # 避免无差别放大水面高频纹理
            output = (
                output
                + self.beta.tanh()
                * support_weight
                * fine_detail
            )

        return output

class _SBAHSemanticChannelGate(nn.Module):
    """Use P3 global semantics to recalibrate only the P2 classification feature."""

    def __init__(
        self,
        c_detail: int,
        c_semantic: int,
        reduction: int = 4,
    ):
        super().__init__()

        c_mid = max(c_detail // reduction, 8)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(c_semantic, c_mid, 1, bias=True)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(c_mid, c_detail, 1, bias=True)

        # Identity initialization:
        # tanh(0)=0, so output initially equals the original P2 feature.
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(
        self,
        detail: torch.Tensor,
        semantic: torch.Tensor,
    ) -> torch.Tensor:
        gate = self.pool(semantic)
        gate = self.fc1(gate)
        gate = self.act(gate)
        gate = torch.tanh(self.fc2(gate))

        # Gate range is approximately [0, 2].
        return detail * (1.0 + gate)

class _HRDDRepDW5(RepVGGDW):
    """
    Train-time multi-branch depthwise detail block.

    Training:
        DW5x5 + DW3x3 + DW1x5 + DW5x1
        + identity BN
        + fixed central high-pass branch

    Deployment:
        all branches are fused into one DW5x5 convolution.
    """

    def __init__(self, channels: int):
        # Do not call RepVGGDW.__init__(), because its native structure is 7x7+3x3.
        # Inheriting RepVGGDW allows the existing BaseModel.fuse() to recognize us.
        nn.Module.__init__(self)

        self.channels = int(channels)

        # No activation inside branches: all branches must stay linearly fuseable.
        self.dw5 = Conv(
            channels, channels,
            k=5, s=1, p=2,
            g=channels,
            act=False,
        )
        self.dw3 = Conv(
            channels, channels,
            k=3, s=1, p=1,
            g=channels,
            act=False,
        )
        self.dw_h = Conv(
            channels, channels,
            k=(1, 5), s=1, p=(0, 2),
            g=channels,
            act=False,
        )
        self.dw_v = Conv(
            channels, channels,
            k=(5, 1), s=1, p=(2, 0),
            g=channels,
            act=False,
        )

        # Identity branch with trainable BN.
        self.id_bn = nn.BatchNorm2d(channels)

        # Per-channel strength of x - AvgPool3x3(x).
        # It provides a lightweight central-difference/high-frequency prior.
        self.edge_scale = nn.Parameter(
            torch.full((channels,), 0.1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "reparam"):
            return self.forward_fuse(x)

        edge = x - F.avg_pool2d(
            x,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        return (
            self.dw5(x)
            + self.dw3(x)
            + self.dw_h(x)
            + self.dw_v(x)
            + self.id_bn(x)
            + self.edge_scale.view(1, -1, 1, 1) * edge
        )

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """Deployment forward: only one depthwise 5x5 remains."""
        return self.reparam(x)

    @staticmethod
    def _pad_to_5x5(kernel: torch.Tensor) -> torch.Tensor:
        """Center-pad 3x3, 1x5 or 5x1 kernels to 5x5."""
        height, width = kernel.shape[-2:]
        pad_h = 5 - height
        pad_w = 5 - width

        return F.pad(
            kernel,
            (
                pad_w // 2,
                pad_w - pad_w // 2,
                pad_h // 2,
                pad_h - pad_h // 2,
            ),
        )

    @torch.no_grad()
    def fuse(self):
        """Fuse every training branch into a single DW5x5 convolution."""
        if hasattr(self, "reparam"):
            return

        kernels = []
        biases = []

        # Fuse Conv+BN for every learned depthwise branch.
        for branch in (
            self.dw5,
            self.dw3,
            self.dw_h,
            self.dw_v,
        ):
            fused = fuse_conv_and_bn(
                branch.conv,
                branch.bn,
            )
            kernels.append(
                self._pad_to_5x5(fused.weight)
            )
            biases.append(fused.bias)

        device = kernels[0].device
        dtype = kernels[0].dtype

        # --------------------------------------------------
        # Identity BN -> depthwise identity 5x5 kernel
        # --------------------------------------------------
        identity_kernel = torch.zeros(
            (self.channels, 1, 5, 5),
            device=device,
            dtype=dtype,
        )
        identity_kernel[:, 0, 2, 2] = 1.0

        identity_scale = (
            self.id_bn.weight
            / torch.sqrt(
                self.id_bn.running_var
                + self.id_bn.eps
            )
        ).to(dtype=dtype)

        identity_kernel *= identity_scale.view(
            -1, 1, 1, 1
        )

        identity_bias = (
            self.id_bn.bias
            - self.id_bn.running_mean
            * identity_scale
        ).to(dtype=dtype)

        # --------------------------------------------------
        # x - AvgPool3x3(x) -> fixed 3x3 high-pass kernel
        # embedded in a 5x5 kernel
        # --------------------------------------------------
        edge_kernel = torch.zeros_like(
            identity_kernel
        )
        edge_kernel[:, :, 1:4, 1:4] = -1.0 / 9.0
        edge_kernel[:, :, 2, 2] += 1.0

        edge_kernel *= self.edge_scale.to(
            dtype=dtype
        ).view(-1, 1, 1, 1)

        final_kernel = (
            torch.stack(kernels, dim=0).sum(dim=0)
            + identity_kernel
            + edge_kernel
        )
        final_bias = (
            torch.stack(biases, dim=0).sum(dim=0)
            + identity_bias
        )

        self.reparam = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=self.channels,
            bias=True,
        ).to(device=device, dtype=dtype)

        self.reparam.weight.copy_(final_kernel)
        self.reparam.bias.copy_(final_bias)
        self.reparam.requires_grad_(False)

        # Delete training-only branches.
        for name in (
            "dw5",
            "dw3",
            "dw_h",
            "dw_v",
            "id_bn",
            "edge_scale",
        ):
            delattr(self, name)


class _HRDResidualDWPW(nn.Module):
    """DRep-DW5 + pointwise convolution + channel-wise residual scale."""

    def __init__(
        self,
        channels: int,
        residual_scale: float = 0.1,
    ):
        super().__init__()

        self.spatial = _HRDDRepDW5(channels)

        # The pointwise layer restores channel mixing lost by DW convolution.
        self.pw = Conv(
            channels,
            channels,
            k=1,
            s=1,
            act=False,
        )

        # Small residual initialization prevents the new branch from
        # strongly disturbing the pretrained/original feature distribution.
        self.gamma = nn.Parameter(
            torch.full(
                (1, channels, 1, 1),
                float(residual_scale),
            )
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.pw(
            self.spatial(x)
        )
        return self.act(
            x + self.gamma * residual
        )


class HRDP2Detect(Detect):
    """
    Half-retained Detail-Reparameterized P2 detection head.

    Only:
        1. replaces the second convolution of P2 box regression;
        2. applies a P3 semantic channel gate to P2 classification input.

    P3/P4/P5 and the original P2 classification tower remain unchanged.
    """

    def __init__(
        self,
        nc: int = 80,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple = (),
        gate_reduction: int = 4,
        use_semantic_gate: bool = True,
        residual_scale: float = 0.1,
        p2_reg_channels: int | None = None,
):
        if end2end:
            raise NotImplementedError(
                "HRDP2Detect currently supports "
                "end2end=False only."
            )

        if len(ch) != 4:
            raise ValueError(
                "HRDP2Detect expects exactly four inputs: "
                "P2, P3, P4 and P5; "
                f"received {len(ch)}."
            )

        if gate_reduction <= 0:
            raise ValueError(
                "gate_reduction must be positive, "
                f"got {gate_reduction}."
            )

        prediction_channels = tuple(ch)

        # Retain original DFL, decode, bias initialization and loss interface.
        super().__init__(
            nc=nc,
            reg_max=reg_max,
            end2end=False,
            ch=prediction_channels,
        )
        self.end2end = False

        # --------------------------------------------------
        # P2 regression tower
        #
        # p2_reg_channels=None:
        #     preserve the original HRD-P2 implementation.
        #
        # p2_reg_channels=48:
        #     DW3 64->64
        #     PW1 64->48
        #     HRDResidualDWPW(48)
        #     prediction Conv 48->4*reg_max
        # --------------------------------------------------
        original_p2_box = self.cv2[0]

        if len(original_p2_box) != 3:
            raise RuntimeError(
                "Unexpected Detect.cv2[0] structure. "
                "HRDP2Detect expects [Conv, Conv, Conv2d]."
            )

        box_in_channels = (
            original_p2_box[0].conv.in_channels
        )

        box_channels = (
            original_p2_box[0].conv.out_channels
        )

        box_output_channels = (
            original_p2_box[-1].out_channels
        )

        if p2_reg_channels is None:
            # Original HRD-P2 behavior.
            self.p2_reg_channels = box_channels

            self.cv2[0] = nn.Sequential(
                original_p2_box[0],
                _HRDResidualDWPW(
                    channels=box_channels,
                    residual_scale=residual_scale,
                ),
                original_p2_box[-1],
            )

        else:
            p2_reg_channels = int(p2_reg_channels)

            if p2_reg_channels <= 0:
                raise ValueError(
                    "p2_reg_channels must be positive or None, "
                    f"got {p2_reg_channels}."
                )

            self.p2_reg_channels = p2_reg_channels

            self.cv2[0] = nn.Sequential(
                # Factorize the original full 3x3 convolution.
                nn.Sequential(
                    DWConv(
                        box_in_channels,
                        box_in_channels,
                        k=3,
                        s=1,
                    ),
                    Conv(
                        box_in_channels,
                        p2_reg_channels,
                        k=1,
                        s=1,
                    ),
                ),

                # Retain the original HRD localization refinement.
                _HRDResidualDWPW(
                    channels=p2_reg_channels,
                    residual_scale=residual_scale,
                ),

                # Preserve the 4*reg_max DFL output.
                nn.Conv2d(
                    p2_reg_channels,
                    box_output_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                ),
            )

        # --------------------------------------------------
        # P3 global semantics -> P2 classification only
        # --------------------------------------------------
        self.use_semantic_gate = bool(
            use_semantic_gate
        )

        self.p2_semantic_gate = (
            _SBAHSemanticChannelGate(
                c_detail=prediction_channels[0],
                c_semantic=prediction_channels[1],
                reduction=gate_reduction,
            )
            if self.use_semantic_gate
            else None
        )

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: torch.nn.Module = None,
        cls_head: torch.nn.Module = None,
    ) -> dict[str, torch.Tensor]:
        if box_head is None or cls_head is None:
            return {}

        if len(x) != self.nl:
            raise ValueError(
                f"HRDP2Detect expects {self.nl} feature maps, "
                f"got {len(x)}."
            )

        batch_size = x[0].shape[0]

        # Regression always uses original raw P2/P3/P4/P5 features.
        boxes = torch.cat(
            [
                box_head[level](x[level]).view(
                    batch_size,
                    4 * self.reg_max,
                    -1,
                )
                for level in range(self.nl)
            ],
            dim=-1,
        )

        # The P3 semantic gate changes only the input of P2 classification.
        p2_cls_feature = (
            self.p2_semantic_gate(x[0], x[1])
            if self.use_semantic_gate
            else x[0]
        )

        cls_logits = [
            cls_head[0](p2_cls_feature).view(
                batch_size,
                self.nc,
                -1,
            )
        ]

        # Original P3/P4/P5 classification towers and inputs.
        cls_logits.extend(
            cls_head[level](x[level]).view(
                batch_size,
                self.nc,
                -1,
            )
            for level in range(1, self.nl)
        )

        scores = torch.cat(
            cls_logits,
            dim=-1,
        )

        # Important: only the four raw prediction features go to DFL decoding.
        return {
            "boxes": boxes,
            "scores": scores,
            "feats": x,
        }

class _SRDSharedDRepDW5(RepVGGDW):
    """Shared reparameterizable DW5 spatial kernel."""

    def __init__(self, channels: int):
        # Keep inheritance from RepVGGDW so BaseModel.fuse() can find it.
        nn.Module.__init__(self)

        self.channels = int(channels)

        # Shared convolution weights; normalization is applied separately
        # for P3/P4/P5 in the outer block.
        self.dw5 = nn.Conv2d(
            channels,
            channels,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=channels,
            bias=False,
        )
        self.dw3 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.dw_h = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, 5),
            stride=1,
            padding=(0, 2),
            groups=channels,
            bias=False,
        )
        self.dw_v = nn.Conv2d(
            channels,
            channels,
            kernel_size=(5, 1),
            stride=1,
            padding=(2, 0),
            groups=channels,
            bias=False,
        )

        self.identity_scale = nn.Parameter(
            torch.ones(channels)
        )
        self.edge_scale = nn.Parameter(
            torch.full((channels,), 0.1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "reparam"):
            return self.forward_fuse(x)

        edge = x - F.avg_pool2d(
            x,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        return (
            self.dw5(x)
            + self.dw3(x)
            + self.dw_h(x)
            + self.dw_v(x)
            + self.identity_scale.view(1, -1, 1, 1) * x
            + self.edge_scale.view(1, -1, 1, 1) * edge
        )

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        return self.reparam(x)

    @staticmethod
    def _pad_to_5x5(kernel: torch.Tensor) -> torch.Tensor:
        height, width = kernel.shape[-2:]

        pad_h = 5 - height
        pad_w = 5 - width

        return F.pad(
            kernel,
            (
                pad_w // 2,
                pad_w - pad_w // 2,
                pad_h // 2,
                pad_h - pad_h // 2,
            ),
        )

    @torch.no_grad()
    def fuse(self):
        """Fuse all spatial branches into one deployment DW5 convolution."""

        if hasattr(self, "reparam"):
            return

        final_kernel = (
            self.dw5.weight
            + self._pad_to_5x5(self.dw3.weight)
            + self._pad_to_5x5(self.dw_h.weight)
            + self._pad_to_5x5(self.dw_v.weight)
        )

        identity_kernel = torch.zeros_like(final_kernel)
        identity_kernel[:, 0, 2, 2] = self.identity_scale

        edge_kernel = torch.zeros_like(final_kernel)
        edge_kernel[:, :, 1:4, 1:4] = -1.0 / 9.0
        edge_kernel[:, :, 2, 2] += 1.0
        edge_kernel *= self.edge_scale.view(-1, 1, 1, 1)

        final_kernel = (
            final_kernel
            + identity_kernel
            + edge_kernel
        )

        self.reparam = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=self.channels,
            bias=True,
        ).to(
            device=final_kernel.device,
            dtype=final_kernel.dtype,
        )

        self.reparam.weight.copy_(final_kernel)
        self.reparam.bias.zero_()
        self.reparam.requires_grad_(False)

        for name in (
            "dw5",
            "dw3",
            "dw_h",
            "dw_v",
            "identity_scale",
            "edge_scale",
        ):
            delattr(self, name)


class _SRDScaleSharedBox(nn.Module):
    """Shared spatial/PW weights with scale-independent GroupNorm."""

    def __init__(
        self,
        channels: int,
        levels: int = 3,
        gn_groups: int = 8,
        residual_scale: float = 0.1,
    ):
        super().__init__()

        groups = min(gn_groups, channels)
        while channels % groups != 0:
            groups -= 1

        # Shared P3/P4/P5 convolution weights.
        self.spatial = _SRDSharedDRepDW5(channels)
        self.pw = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        # Independent normalization for P3/P4/P5.
        self.spatial_norm = nn.ModuleList(
            nn.GroupNorm(groups, channels)
            for _ in range(levels)
        )
        self.pw_norm = nn.ModuleList(
            nn.GroupNorm(groups, channels)
            for _ in range(levels)
        )

        # Independent channel-wise residual strength for each scale.
        self.gamma = nn.Parameter(
            torch.full(
                (levels, channels),
                float(residual_scale),
            )
        )

        self.act = nn.SiLU(inplace=True)

    def forward(
        self,
        x: torch.Tensor,
        level: int,
    ) -> torch.Tensor:
        residual = self.spatial(x)
        residual = self.spatial_norm[level](residual)
        residual = self.act(residual)

        residual = self.pw(residual)
        residual = self.pw_norm[level](residual)

        gamma = self.gamma[level].view(
            1,
            -1,
            1,
            1,
        )

        return self.act(
            x + gamma * residual
        )


class SRDHRDDetect(HRDP2Detect):
    """
    HRD-P2 plus scale-normalized shared P3/P4/P5 regression.

    P2 regression and classification remain identical to HRDP2Detect.
    Only the second regression stage of P3/P4/P5 is replaced.
    """

    def __init__(
        self,
        nc: int = 80,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple = (),
        gate_reduction: int = 4,
        use_semantic_gate: bool = True,
        residual_scale: float = 0.1,
        gn_groups: int = 8,
    ):
        super().__init__(
            nc=nc,
            reg_max=reg_max,
            end2end=end2end,
            ch=ch,
            gate_reduction=gate_reduction,
            use_semantic_gate=use_semantic_gate,
            residual_scale=residual_scale,
        )

        box_channels = []

        for level in range(1, 4):
            tower = self.cv2[level]

            if len(tower) != 3:
                raise RuntimeError(
                    "SRDHRDDetect expects P3-P5 box towers "
                    "to be [Conv, Conv, Conv2d]."
                )

            box_channels.append(
                tower[0].conv.out_channels
            )

        if len(set(box_channels)) != 1:
            raise ValueError(
                "P3/P4/P5 box hidden channels must be identical, "
                f"got {box_channels}."
            )

        self.shared_box = _SRDScaleSharedBox(
            channels=box_channels[0],
            levels=3,
            gn_groups=gn_groups,
            residual_scale=residual_scale,
        )

        # One scalar for P3/P4/P5 regression distributions.
        self.box_scale = nn.Parameter(
            torch.ones(3)
        )

        # Keep each scale's first adapter and final DFL predictor.
        # Remove the three original independent second 3x3 convolutions.
        for level in range(1, 4):
            tower = self.cv2[level]

            self.cv2[level] = nn.Sequential(
                tower[0],
                nn.Identity(),
                tower[-1],
            )

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: torch.nn.Module = None,
        cls_head: torch.nn.Module = None,
    ) -> dict[str, torch.Tensor]:
        if box_head is None or cls_head is None:
            return {}

        if len(x) != self.nl:
            raise ValueError(
                f"SRDHRDDetect expects {self.nl} feature maps, "
                f"got {len(x)}."
            )

        batch_size = x[0].shape[0]

        # P2 stays exactly as in HRDP2Detect.
        box_logits = [
            box_head[0](x[0])
        ]

        # P3/P4/P5:
        # independent first adapter -> shared block with independent GN
        # -> independent DFL predictor.
        for level in range(1, self.nl):
            feature = box_head[level][0](x[level])

            feature = self.shared_box(
                feature,
                level=level - 1,
            )

            prediction = box_head[level][-1](feature)
            prediction = (
                self.box_scale[level - 1]
                * prediction
            )

            box_logits.append(prediction)

        boxes = torch.cat(
            [
                prediction.view(
                    batch_size,
                    4 * self.reg_max,
                    -1,
                )
                for prediction in box_logits
            ],
            dim=-1,
        )

        # Keep the original P3 semantic gate for P2 classification.
        p2_cls_feature = (
            self.p2_semantic_gate(x[0], x[1])
            if self.use_semantic_gate
            else x[0]
        )

        cls_logits = [
            cls_head[0](p2_cls_feature)
        ]

        cls_logits.extend(
            cls_head[level](x[level])
            for level in range(1, self.nl)
        )

        scores = torch.cat(
            [
                prediction.view(
                    batch_size,
                    self.nc,
                    -1,
                )
                for prediction in cls_logits
            ],
            dim=-1,
        )

        # Preserve the current Ultralytics loss/decode interface.
        return {
            "boxes": boxes,
            "scores": scores,
            "feats": x,
        }

class _SBAHBlurMixer(nn.Module):
    """
    Partial-channel 3x3/5x5 blur-aware mixer for P2 classification.

    Only one quarter of the channels uses 3x3 DWConv and another
    quarter uses 5x5 DWConv, avoiding full-channel large convolutions.
    """

    def __init__(self, c: int):
        super().__init__()

        if c % 4 != 0:
            raise ValueError(
                f"SBAH P2 channels must be divisible by 4, got {c}."
            )

        c_half = c // 2
        c_quarter = c // 4

        self.reduce = Conv(c, c_half, k=1, s=1)
        self.dw3 = DWConv(c_quarter, c_quarter, k=3, s=1)
        self.dw5 = DWConv(c_quarter, c_quarter, k=5, s=1)

        # Small residual contribution at initialization.
        self.mix_scale = nn.Parameter(torch.tensor(0.1))

    @staticmethod
    def channel_shuffle(
        x: torch.Tensor,
        groups: int = 2,
    ) -> torch.Tensor:
        b, c, h, w = x.shape

        if c % groups != 0:
            raise ValueError(
                f"Channels {c} must be divisible by groups {groups}."
            )

        x = x.view(b, groups, c // groups, h, w)
        x = x.transpose(1, 2).contiguous()
        return x.view(b, c, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reduced = self.reduce(x)
        branch3, branch5 = reduced.chunk(2, dim=1)

        mixed = torch.cat(
            (
                reduced,
                self.dw3(branch3),
                self.dw5(branch5),
            ),
            dim=1,
        )
        mixed = self.channel_shuffle(mixed, groups=2)

        return x + torch.tanh(self.mix_scale) * mixed


class _SBAHScaleSharedDS(nn.Module):
    """
    Depthwise-separable residual block.

    Convolution weights are shared between feature levels, while every
    feature level retains independent GroupNorm parameters.
    """

    def __init__(
        self,
        c: int,
        levels: int = 1,
        gn_groups: int = 8,
    ):
        super().__init__()

        groups = max(1, min(gn_groups, c))
        while c % groups != 0:
            groups -= 1

        self.dw = nn.Conv2d(
            c,
            c,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=c,
            bias=False,
        )
        self.pw = nn.Conv2d(
            c,
            c,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        # Independent normalization for each pyramid level.
        self.dw_norm = nn.ModuleList(
            [nn.GroupNorm(groups, c) for _ in range(levels)]
        )
        self.pw_norm = nn.ModuleList(
            [nn.GroupNorm(groups, c) for _ in range(levels)]
        )

        self.act = nn.SiLU(inplace=True)

    def forward(
        self,
        x: torch.Tensor,
        level: int = 0,
    ) -> torch.Tensor:
        y = self.dw(x)
        y = self.dw_norm[level](y)
        y = self.act(y)

        y = self.pw(y)
        y = self.pw_norm[level](y)

        return self.act(x + y)

class DSTSP2Detect(Detect):
    """
    Detail-Shared then Task-Split P2 detection head.

    P2:
        shared DRep-DW5 detail stem
        -> lightweight box/cls task adapters

    P3/P4/P5:
        level-specific 1x1 adapters
        -> shared stem
        -> shared but decoupled box/cls towers
        -> level-specific predictors
    """

    def __init__(
        self,
        nc: int = 80,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple = (),
        shared_ch: int = 48,
        gate_reduction: int = 4,
        gn_groups: int = 8,
        residual_scale: float = 0.1,
    ):
        if end2end:
            raise NotImplementedError(
                "DSTSP2Detect currently supports end2end=False only."
            )

        if len(ch) != 4:
            raise ValueError(
                "DSTSP2Detect expects exactly four inputs: "
                "P2, P3, P4 and P5; "
                f"received {len(ch)}."
            )

        if shared_ch <= 0:
            raise ValueError(
                f"shared_ch must be positive, got {shared_ch}."
            )

        if gate_reduction <= 0:
            raise ValueError(
                "gate_reduction must be positive, "
                f"got {gate_reduction}."
            )

        prediction_channels = tuple(ch)

        # 保留当前Detect的DFL、解码、损失和推理接口
        super().__init__(
            nc=nc,
            reg_max=reg_max,
            end2end=False,
            ch=prediction_channels,
        )
        self.end2end = False

        p2_channels = prediction_channels[0]

        # --------------------------------------------------
        # 1. P2分类和回归共享一次DRep-DW5细节提取
        # --------------------------------------------------
        self.p2_detail_stem = _HRDResidualDWPW(
            channels=p2_channels,
            residual_scale=residual_scale,
        )

        # --------------------------------------------------
        # 2. P2回归轻量任务适配器
        # --------------------------------------------------
        self.p2_box_adapter = _SBAHScaleSharedDS(
            p2_channels,
            levels=1,
            gn_groups=gn_groups,
        )

        # --------------------------------------------------
        # 3. P3语义只校准P2分类
        # --------------------------------------------------
        self.p2_semantic_gate = _SBAHSemanticChannelGate(
            c_detail=p2_channels,
            c_semantic=prediction_channels[1],
            reduction=gate_reduction,
        )

        self.p2_cls_adapter = _SBAHScaleSharedDS(
            p2_channels,
            levels=1,
            gn_groups=gn_groups,
        )

        # --------------------------------------------------
        # 4. P3/P4/P5先对齐到48通道
        # 当前n模型：64/128/256 -> 48/48/48
        # --------------------------------------------------
        self.adapters = nn.ModuleList(
            [
                Conv(c, shared_ch, k=1, s=1)
                for c in prediction_channels[1:]
            ]
        )

        # 卷积权重共享，各尺度GN参数独立
        self.shared_stem = _SBAHScaleSharedDS(
            shared_ch,
            levels=3,
            gn_groups=gn_groups,
        )

        self.shared_box = _SBAHScaleSharedDS(
            shared_ch,
            levels=3,
            gn_groups=gn_groups,
        )

        self.shared_cls = _SBAHScaleSharedDS(
            shared_ch,
            levels=3,
            gn_groups=gn_groups,
        )

        # --------------------------------------------------
        # 5. 四尺度独立预测器
        # --------------------------------------------------
        self.cv2 = nn.ModuleList(
            [nn.Conv2d(p2_channels, 4 * reg_max, 1)]
            + [
                nn.Conv2d(shared_ch, 4 * reg_max, 1)
                for _ in range(3)
            ]
        )

        self.cv3 = nn.ModuleList(
            [nn.Conv2d(p2_channels, nc, 1)]
            + [
                nn.Conv2d(shared_ch, nc, 1)
                for _ in range(3)
            ]
        )

        # P2/P3/P4/P5独立回归缩放
        self.reg_scale = nn.Parameter(
            torch.ones(4)
        )

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: nn.Module = None,
        cls_head: nn.Module = None,
    ) -> dict[str, torch.Tensor]:
        if box_head is None or cls_head is None:
            return {}

        if len(x) != 4:
            raise ValueError(
                f"DSTSP2Detect expects 4 feature maps, got {len(x)}."
            )

        prediction_features = list(x)
        batch_size = prediction_features[0].shape[0]

        # --------------------------------------------------
        # P2：先共享细节主干
        # --------------------------------------------------
        p2_detail = self.p2_detail_stem(
            prediction_features[0]
        )

        # P2回归任务适配
        p2_box_feature = self.p2_box_adapter(
            p2_detail,
            level=0,
        )

        # P2分类：共享细节 + P3语义校准
        p2_cls_feature = self.p2_semantic_gate(
            p2_detail,
            prediction_features[1],
        )
        p2_cls_feature = self.p2_cls_adapter(
            p2_cls_feature,
            level=0,
        )

        box_logits = [
            box_head[0](p2_box_feature)
            * self.reg_scale[0]
        ]
        cls_logits = [
            cls_head[0](p2_cls_feature)
        ]

        # --------------------------------------------------
        # P3/P4/P5共享检测塔
        # --------------------------------------------------
        for level, feature in enumerate(
            prediction_features[1:]
        ):
            feature = self.adapters[level](feature)
            feature = self.shared_stem(
                feature,
                level,
            )

            box_feature = self.shared_box(
                feature,
                level,
            )
            cls_feature = self.shared_cls(
                feature,
                level,
            )

            box_logits.append(
                box_head[level + 1](box_feature)
                * self.reg_scale[level + 1]
            )
            cls_logits.append(
                cls_head[level + 1](cls_feature)
            )

        boxes = torch.cat(
            [
                output.view(
                    batch_size,
                    4 * self.reg_max,
                    -1,
                )
                for output in box_logits
            ],
            dim=-1,
        )

        scores = torch.cat(
            [
                output.view(
                    batch_size,
                    self.nc,
                    -1,
                )
                for output in cls_logits
            ],
            dim=-1,
        )

        # feats必须返回原始四尺度特征，
        # Detect会使用它们生成anchor和stride
        return {
            "boxes": boxes,
            "scores": scores,
            "feats": prediction_features,
        }

    def bias_init(self):
        """Initialize four direct prediction layers."""

        with torch.no_grad():
            for i, (box_pred, cls_pred) in enumerate(
                zip(self.cv2, self.cv3)
            ):
                box_pred.bias.fill_(2.0)

                cls_bias = math.log(
                    5
                    / self.nc
                    / (640 / self.stride[i]) ** 2
                )
                cls_pred.bias[: self.nc].fill_(cls_bias)

class SBAHDetect(Detect):
    """
    P2-expert Blur-Aware Asymmetric Shared Head.

    Input order:
        0: P2 prediction feature
        1: P3 prediction feature
        2: P4 prediction feature
        3: P5 prediction feature

    P2:
        - independent lightweight regression tower;
        - P3 semantic-gated blur-aware classification tower.

    P3/P4/P5:
        - level-specific 1x1 adapters;
        - shared stem;
        - shared regression/classification towers;
        - level-specific prediction convolutions.
    """

    def __init__(
        self,
        nc: int = 80,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple = (),
        shared_ch: int = 48,
        gate_reduction: int = 4,
        gn_groups: int = 8,
    ):
        if end2end:
            raise NotImplementedError(
                "SBAHDetect currently supports end2end=False only."
            )

        if len(ch) != 4:
            raise ValueError(
                "SBAHDetect expects exactly four inputs: "
                "P2, P3, P4 and P5; "
                f"received {len(ch)}."
            )

        if shared_ch <= 0:
            raise ValueError(
                f"shared_ch must be positive, got {shared_ch}."
            )

        prediction_channels = tuple(ch)

        # Retain Detect decoding, DFL and inference interfaces.
        super().__init__(
            nc=nc,
            reg_max=reg_max,
            end2end=False,
            ch=prediction_channels,
        )
        self.end2end = False

        # --------------------------------------------------
        # 1. P2 semantic gate and blur-aware classification
        # --------------------------------------------------
        self.p2_semantic_gate = _SBAHSemanticChannelGate(
            c_detail=prediction_channels[0],
            c_semantic=prediction_channels[1],
            reduction=gate_reduction,
        )

        self.p2_blur_mixer = _SBAHBlurMixer(
            prediction_channels[0]
        )

        self.p2_cls_tower = _SBAHScaleSharedDS(
            prediction_channels[0],
            levels=1,
            gn_groups=gn_groups,
        )

        # --------------------------------------------------
        # 2. Independent P2 localization expert
        # --------------------------------------------------
        self.p2_box_tower = nn.ModuleList(
            [
                _SBAHScaleSharedDS(
                    prediction_channels[0],
                    levels=1,
                    gn_groups=gn_groups,
                ),
                _SBAHScaleSharedDS(
                    prediction_channels[0],
                    levels=1,
                    gn_groups=gn_groups,
                ),
            ]
        )

        # --------------------------------------------------
        # 3. P3/P4/P5 channel adapters
        # Current n-model: 64/128/256 -> 48/48/48
        # --------------------------------------------------
        self.adapters = nn.ModuleList(
            [
                Conv(c, shared_ch, k=1, s=1)
                for c in prediction_channels[1:]
            ]
        )

        # Shared convolution weights, independent scale normalization.
        self.shared_stem = _SBAHScaleSharedDS(
            shared_ch,
            levels=3,
            gn_groups=gn_groups,
        )
        self.shared_box = _SBAHScaleSharedDS(
            shared_ch,
            levels=3,
            gn_groups=gn_groups,
        )
        self.shared_cls = _SBAHScaleSharedDS(
            shared_ch,
            levels=3,
            gn_groups=gn_groups,
        )

        # --------------------------------------------------
        # 4. Replace the original expensive Detect towers
        # --------------------------------------------------
        self.cv2 = nn.ModuleList(
            [nn.Conv2d(prediction_channels[0], 4 * reg_max, 1)]
            + [
                nn.Conv2d(shared_ch, 4 * reg_max, 1)
                for _ in range(3)
            ]
        )

        self.cv3 = nn.ModuleList(
            [nn.Conv2d(prediction_channels[0], nc, 1)]
            + [
                nn.Conv2d(shared_ch, nc, 1)
                for _ in range(3)
            ]
        )

        # Independent localization scale for P2/P3/P4/P5.
        self.reg_scale = nn.Parameter(torch.ones(4))

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: nn.Module = None,
        cls_head: nn.Module = None,
    ) -> dict[str, torch.Tensor]:
        if box_head is None or cls_head is None:
            return {}

        if len(x) != 4:
            raise ValueError(
                f"SBAHDetect forward expects 4 feature maps, got {len(x)}."
            )

        prediction_features = list(x)
        bs = prediction_features[0].shape[0]

        # --------------------------------------------------
        # P2 regression: retain raw geometry
        # --------------------------------------------------
        p2_box_feature = prediction_features[0]

        for block in self.p2_box_tower:
            p2_box_feature = block(
                p2_box_feature,
                level=0,
            )

        # --------------------------------------------------
        # P2 classification: P3 semantics + blur-aware detail
        # --------------------------------------------------
        p2_cls_feature = self.p2_semantic_gate(
            prediction_features[0],
            prediction_features[1],
        )
        p2_cls_feature = self.p2_blur_mixer(p2_cls_feature)
        p2_cls_feature = self.p2_cls_tower(
            p2_cls_feature,
            level=0,
        )

        box_logits = [
            box_head[0](p2_box_feature) * self.reg_scale[0]
        ]
        cls_logits = [
            cls_head[0](p2_cls_feature)
        ]

        # --------------------------------------------------
        # P3/P4/P5 shared head
        # --------------------------------------------------
        for level, feature in enumerate(
            prediction_features[1:]
        ):
            feature = self.adapters[level](feature)
            feature = self.shared_stem(feature, level)

            box_feature = self.shared_box(feature, level)
            cls_feature = self.shared_cls(feature, level)

            box_logits.append(
                box_head[level + 1](box_feature)
                * self.reg_scale[level + 1]
            )
            cls_logits.append(
                cls_head[level + 1](cls_feature)
            )

        boxes = torch.cat(
            [
                output.view(
                    bs,
                    4 * self.reg_max,
                    -1,
                )
                for output in box_logits
            ],
            dim=-1,
        )

        scores = torch.cat(
            [
                output.view(
                    bs,
                    self.nc,
                    -1,
                )
                for output in cls_logits
            ],
            dim=-1,
        )

        # Only the four prediction maps may be returned as feats.
        return {
            "boxes": boxes,
            "scores": scores,
            "feats": prediction_features,
        }

    def bias_init(self):
        """Initialize direct Conv2d prediction biases."""

        with torch.no_grad():
            for i, (box_pred, cls_pred) in enumerate(
                zip(self.cv2, self.cv3)
            ):
                box_pred.bias.fill_(2.0)

                cls_bias = math.log(
                    5
                    / self.nc
                    / (640 / self.stride[i]) ** 2
                )
                cls_pred.bias[: self.nc].fill_(cls_bias)

class _SBAHP2HRDBoxTower(nn.Module):
    """
    Independent P2 localization tower:

    DRep-DW5 + PW residual detail block
    followed by one DWPW residual adapter.
    """

    def __init__(
        self,
        channels: int,
        gn_groups: int = 8,
        residual_scale: float = 0.1,
    ):
        super().__init__()

        self.detail_block = _HRDResidualDWPW(
            channels,
            residual_scale=residual_scale,
        )

        self.adapter = _SBAHScaleSharedDS(
            channels,
            levels=1,
            gn_groups=gn_groups,
        )

    def forward(
        self,
        x: torch.Tensor,
        level: int = 0,
    ) -> torch.Tensor:
        if level != 0:
            raise ValueError(
                "_SBAHP2HRDBoxTower is only used for P2 level 0."
            )

        x = self.detail_block(x)
        return self.adapter(x, level=0)


class SBAHHRDDetect(SBAHDetect):
    """
    SBAH-HRD64 detection head.

    P2:
        - independent HRD regression expert;
        - semantic-gated blur-aware classification expert.

    P3/P4/P5:
        - align to shared_ch=64;
        - shared convolution weights;
        - independent GroupNorm and regression scale.
    """

    def __init__(
        self,
        nc: int = 80,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple = (),
        shared_ch: int = 64,
        gate_reduction: int = 4,
        gn_groups: int = 8,
        residual_scale: float = 0.1,
    ):
        super().__init__(
            nc=nc,
            reg_max=reg_max,
            end2end=end2end,
            ch=ch,
            shared_ch=shared_ch,
            gate_reduction=gate_reduction,
            gn_groups=gn_groups,
        )

        # 真正替换父类的两层普通P2回归塔，
        # 不是只更换检测头名称。
        self.p2_box_tower = nn.ModuleList(
            [
                _SBAHP2HRDBoxTower(
                    channels=int(ch[0]),
                    gn_groups=gn_groups,
                    residual_scale=residual_scale,
                )
            ]
        )

def _sgs_valid_gn_groups(
    channels: int,
    requested_groups: int,
) -> int:
    """Find the largest valid GroupNorm group count."""

    groups = max(
        1,
        min(int(requested_groups), int(channels)),
    )

    while channels % groups != 0:
        groups -= 1

    return groups


class _SGSLevelAdapter(nn.Module):
    """
    Level-specific channel adapter.

    P2/P3 already have 64 channels in the current n model, so their
    projection is Identity. P4/P5 use a 1x1 projection.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        gn_groups: int = 8,
    ):
        super().__init__()

        self.proj = (
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
            _sgs_valid_gn_groups(c2, gn_groups),
            c2,
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.proj(x)))


class _SGSSharedDWPW(nn.Module):
    """
    Cross-scale shared depthwise-pointwise residual block.

    DW/PW convolution weights are shared by P2/P3/P4/P5.
    Every scale has independent GroupNorm parameters.
    """

    def __init__(
        self,
        channels: int,
        levels: int = 4,
        gn_groups: int = 8,
    ):
        super().__init__()

        groups = _sgs_valid_gn_groups(
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

        self.pw = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        # Same convolution, independent normalization per scale.
        self.dw_norm = nn.ModuleList(
            [
                nn.GroupNorm(groups, channels)
                for _ in range(levels)
            ]
        )
        self.pw_norm = nn.ModuleList(
            [
                nn.GroupNorm(groups, channels)
                for _ in range(levels)
            ]
        )

        self.act = nn.SiLU(inplace=True)

    def forward(
        self,
        x: torch.Tensor,
        level: int,
    ) -> torch.Tensor:
        identity = x

        x = self.dw(x)
        x = self.dw_norm[level](x)
        x = self.act(x)

        x = self.pw(x)
        x = self.pw_norm[level](x)

        return self.act(identity + x)


class SGSDetect(Detect):
    """
    Four-scale Shared-GN Separable decoupled detection head.

    Input order:
        0: P2
        1: P3
        2: P4
        3: P5

    The classification and regression towers are completely decoupled.
    Convolution weights are shared across feature levels, while GN,
    predictors and regression scales remain level-specific.
    """

    def __init__(
        self,
        nc: int = 80,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple = (),
        shared_ch: int = 64,
        num_convs: int = 2,
        gn_groups: int = 8,
    ):
        if end2end:
            raise NotImplementedError(
                "SGSDetect currently supports end2end=False only."
            )

        if len(ch) != 4:
            raise ValueError(
                "SGSDetect expects exactly four inputs: "
                "P2, P3, P4 and P5; "
                f"received {len(ch)}."
            )

        if shared_ch <= 0:
            raise ValueError(
                f"shared_ch must be positive, got {shared_ch}."
            )

        if num_convs <= 0:
            raise ValueError(
                f"num_convs must be positive, got {num_convs}."
            )

        prediction_channels = tuple(ch)

        # Retain the original DFL, decoding and inference interfaces.
        super().__init__(
            nc=nc,
            reg_max=reg_max,
            end2end=False,
            ch=prediction_channels,
        )
        self.end2end = False

        # --------------------------------------------------
        # 1. Level-specific adapters
        # Current n model: [64, 64, 128, 256] -> all 64
        # --------------------------------------------------
        self.adapters = nn.ModuleList(
            [
                _SGSLevelAdapter(
                    c1=c,
                    c2=shared_ch,
                    gn_groups=gn_groups,
                )
                for c in prediction_channels
            ]
        )

        # --------------------------------------------------
        # 2. Shared but task-decoupled towers
        #
        # ModuleList represents tower depth, not feature levels.
        # The same block is called by all four levels.
        # --------------------------------------------------
        self.box_tower = nn.ModuleList(
            [
                _SGSSharedDWPW(
                    channels=shared_ch,
                    levels=self.nl,
                    gn_groups=gn_groups,
                )
                for _ in range(num_convs)
            ]
        )

        self.cls_tower = nn.ModuleList(
            [
                _SGSSharedDWPW(
                    channels=shared_ch,
                    levels=self.nl,
                    gn_groups=gn_groups,
                )
                for _ in range(num_convs)
            ]
        )

        # --------------------------------------------------
        # 3. Level-specific predictors
        #
        # These overwrite the expensive cv2/cv3 towers constructed
        # temporarily by Detect.__init__().
        # --------------------------------------------------
        self.cv2 = nn.ModuleList(
            [
                nn.Conv2d(
                    shared_ch,
                    4 * reg_max,
                    kernel_size=1,
                )
                for _ in range(self.nl)
            ]
        )

        self.cv3 = nn.ModuleList(
            [
                nn.Conv2d(
                    shared_ch,
                    nc,
                    kernel_size=1,
                )
                for _ in range(self.nl)
            ]
        )

        # Independent scale for P2/P3/P4/P5 regression logits.
        self.reg_scale = nn.Parameter(
            torch.ones(self.nl)
        )

    @staticmethod
    def _run_tower(
        x: torch.Tensor,
        tower: nn.ModuleList,
        level: int,
    ) -> torch.Tensor:
        # Same tower modules are reused for every feature level.
        for block in tower:
            x = block(x, level)

        return x

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: nn.Module = None,
        cls_head: nn.Module = None,
    ) -> dict[str, torch.Tensor]:
        if box_head is None or cls_head is None:
            return {}

        if len(x) != self.nl:
            raise ValueError(
                f"SGSDetect expects {self.nl} feature maps, "
                f"got {len(x)}."
            )

        prediction_features = list(x)
        bs = prediction_features[0].shape[0]

        box_logits = []
        cls_logits = []

        for level, feature in enumerate(
            prediction_features
        ):
            adapted = self.adapters[level](feature)

            # Classification and regression do not share task features.
            box_feature = self._run_tower(
                adapted,
                self.box_tower,
                level,
            )
            cls_feature = self._run_tower(
                adapted,
                self.cls_tower,
                level,
            )

            box_output = box_head[level](box_feature)
            box_output = (
                box_output * self.reg_scale[level]
            )

            cls_output = cls_head[level](cls_feature)

            box_logits.append(box_output)
            cls_logits.append(cls_output)

        boxes = torch.cat(
            [
                output.view(
                    bs,
                    4 * self.reg_max,
                    -1,
                )
                for output in box_logits
            ],
            dim=-1,
        )

        scores = torch.cat(
            [
                output.view(
                    bs,
                    self.nc,
                    -1,
                )
                for output in cls_logits
            ],
            dim=-1,
        )

        # Must contain only P2/P3/P4/P5.
        return {
            "boxes": boxes,
            "scores": scores,
            "feats": prediction_features,
        }

    def bias_init(self):
        """
        Initialize direct Conv2d prediction biases.

        The original Detect implementation uses a[-1], which cannot
        be used after cv2/cv3 are replaced by direct Conv2d layers.
        """

        with torch.no_grad():
            for i, (box_pred, cls_pred) in enumerate(
                zip(self.cv2, self.cv3)
            ):
                box_pred.bias.fill_(2.0)

                cls_bias = math.log(
                    5
                    / self.nc
                    / (640 / self.stride[i]) ** 2
                )
                cls_pred.bias[: self.nc].fill_(
                    cls_bias
                )

class FGHTDSIDetect(Detect):
    """
    Four-scale Detect with parameter-efficient FG-HTDSI.

    YAML input order:
        0: layer19, final P2 prediction feature
        1: layer22, final P3 prediction feature
        2: layer25, final P4 prediction feature
        3: layer28, final P5 prediction feature
        4: layer16, P3-TD semantic source for P2
        5: layer13, P4-TD semantic source for P3
    """

    def __init__(
        self,
        nc: int = 80,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple = (),
    ):
        # 当前版本对应标准YOLO11检测模式
        if end2end:
            raise NotImplementedError(
                "FGHTDSIDetect currently supports "
                "end2end=False only."
            )

        if len(ch) != 6:
            raise ValueError(
                "FGHTDSIDetect expects exactly 6 inputs: "
                "P2, P3, P4, P5, P3-TD, P4-TD; "
                f"received {len(ch)}."
            )

        # 前四路才是真正参与检测预测的特征
        prediction_channels = tuple(ch[:4])

        # 原Detect只能根据四个预测尺度构建
        super().__init__(
            nc=nc,
            reg_max=reg_max,
            end2end=end2end,
            ch=prediction_channels,
        )

        # layer16 P3-TD -> layer19 P2
        self.p2_fghtdsi = _FGHTDSIInjection(
            c_detail=prediction_channels[0],
            c_semantic=ch[4],
            reduction=8,
            use_fine_detail=True,
        )

        # layer13 P4-TD -> layer22 P3
        self.p3_fghtdsi = _FGHTDSIInjection(
            c_detail=prediction_channels[1],
            c_semantic=ch[5],
            reduction=8,
            use_fine_detail=True,
        )

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: torch.nn.Module = None,
        cls_head: torch.nn.Module = None,
    ) -> dict[str, torch.Tensor]:
        """
        Original features are used for box regression.
        FG-HTDSI features are used for classification.
        """

        if box_head is None or cls_head is None:
            return {}

        # self.nl由super根据前四路通道创建，因此self.nl=4
        prediction_features = list(x[: self.nl])

        # 六输入中的额外语义源
        semantic_for_p2 = x[4]  # layer16
        semantic_for_p3 = x[5]  # layer13

        # 分类和回归使用不同的输入列表
        classification_features = prediction_features.copy()
        box_features = prediction_features.copy()

        # P2分类：layer19 + layer16
        classification_features[0] = self.p2_fghtdsi(
            prediction_features[0],
            semantic_for_p2,
        )

        # P3分类：layer22 + layer13
        classification_features[1] = self.p3_fghtdsi(
            prediction_features[1],
            semantic_for_p3,
        )

        # P4、P5分类保持原始特征
        bs = prediction_features[0].shape[0]

        # 回归分支仍然使用原始P2/P3/P4/P5
        boxes = torch.cat(
            [
                box_head[i](box_features[i]).view(
                    bs,
                    4 * self.reg_max,
                    -1,
                )
                for i in range(self.nl)
            ],
            dim=-1,
        )

        # 分类分支中的P2、P3使用FG-HTDSI增强特征
        scores = torch.cat(
            [
                cls_head[i](classification_features[i]).view(
                    bs,
                    self.nc,
                    -1,
                )
                for i in range(self.nl)
            ],
            dim=-1,
        )

        # 这里只能返回四个预测特征
        # anchor和stride不能读取额外两个语义源
        return {
            "boxes": boxes,
            "scores": scores,
            "feats": prediction_features,
        }

class Segment(Detect):
    """YOLO Segment head for segmentation models.

    This class extends the Detect head to include mask prediction capabilities for instance segmentation tasks.

    Attributes:
        nm (int): Number of masks.
        npr (int): Number of protos.
        proto (Proto): Prototype generation module.
        cv4 (nn.ModuleList): Convolution layers for mask coefficients.

    Methods:
        forward: Return model outputs and mask coefficients.

    Examples:
        Create a segmentation head
        >>> segment = Segment(nc=80, nm=32, npr=256, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = segment(x)
    """

    def __init__(self, nc: int = 80, nm: int = 32, npr: int = 256, reg_max=16, end2end=False, ch: tuple = ()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers.

        Args:
            nc (int): Number of classes.
            nm (int): Number of masks.
            npr (int): Number of protos.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, reg_max, end2end, ch)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)
        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for backward compatibility."""
        return dict(box_head=self.cv2, cls_head=self.cv3, mask_head=self.cv4)

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, mask_head=self.one2one_cv4)

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        outputs = super().forward(x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self.proto(x[0])  # mask protos
        if isinstance(preds, dict):  # training and validating during training
            if self.end2end:
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = proto.detach()
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode predicted bounding boxes and class probabilities, concatenated with mask coefficients."""
        preds = super()._inference(x)
        return torch.cat([preds, x["mask_coefficient"]], dim=1)

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module, cls_head: torch.nn.Module, mask_head: torch.nn.Module
    ) -> dict[str, torch.Tensor]:
        """Concatenates and returns predicted bounding boxes, class probabilities, and mask coefficients."""
        preds = super().forward_head(x, box_head, cls_head)
        if mask_head is not None:
            bs = x[0].shape[0]  # batch size
            preds["mask_coefficient"] = torch.cat([mask_head[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
        return preds

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Post-process YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc + nm) with last dimension
                format [x1, y1, x2, y2, class_probs, mask_coefficient].

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6 + nm) and last
                dimension format [x1, y1, x2, y2, max_class_prob, class_index, mask_coefficient].
        """
        boxes, scores, mask_coefficient = preds.split([4, self.nc, self.nm], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        mask_coefficient = mask_coefficient.gather(dim=1, index=idx.repeat(1, 1, self.nm))
        return torch.cat([boxes, scores, conf, mask_coefficient], dim=-1)

    def fuse(self) -> None:
        """Remove the one2many head for inference optimization."""
        self.cv2 = self.cv3 = self.cv4 = None


class Segment26(Segment):
    """YOLO26 Segment head for segmentation models.

    This class extends the Segment head with Proto26 for mask prediction in instance segmentation tasks.

    Attributes:
        nm (int): Number of masks.
        npr (int): Number of protos.
        proto (Proto26): Prototype generation module.
        cv4 (nn.ModuleList): Convolution layers for mask coefficients.

    Methods:
        forward: Return model outputs and mask coefficients.

    Examples:
        Create a segmentation head
        >>> segment = Segment26(nc=80, nm=32, npr=256, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = segment(x)
    """

    def __init__(self, nc: int = 80, nm: int = 32, npr: int = 256, reg_max=16, end2end=False, ch: tuple = ()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers.

        Args:
            nc (int): Number of classes.
            nm (int): Number of masks.
            npr (int): Number of protos.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, nm, npr, reg_max, end2end, ch)
        self.proto = Proto26(ch, self.npr, self.nm, nc)  # protos

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        outputs = Detect.forward(self, x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self.proto(x)  # mask protos
        if isinstance(preds, dict):  # training and validating during training
            if self.end2end:
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = (
                    tuple(p.detach() for p in proto) if isinstance(proto, tuple) else proto.detach()
                )
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)

    def fuse(self) -> None:
        """Remove the one2many head and extra part of proto module for inference optimization."""
        super().fuse()
        if hasattr(self.proto, "fuse"):
            self.proto.fuse()


class OBB(Detect):
    """YOLO OBB detection head for detection with rotation models.

    This class extends the Detect head to include oriented bounding box prediction with rotation angles.

    Attributes:
        ne (int): Number of extra parameters.
        cv4 (nn.ModuleList): Convolution layers for angle prediction.
        angle (torch.Tensor): Predicted rotation angles.

    Methods:
        forward: Concatenate and return predicted bounding boxes and class probabilities.
        decode_bboxes: Decode rotated bounding boxes.

    Examples:
        Create an OBB detection head
        >>> obb = OBB(nc=80, ne=1, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = obb(x)
    """

    def __init__(self, nc: int = 80, ne: int = 1, reg_max=16, end2end=False, ch: tuple = ()):
        """Initialize OBB with number of classes `nc` and layer channels `ch`.

        Args:
            nc (int): Number of classes.
            ne (int): Number of extra parameters.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, reg_max, end2end, ch)
        self.ne = ne  # number of extra parameters

        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)
        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for backward compatibility."""
        return dict(box_head=self.cv2, cls_head=self.cv3, angle_head=self.cv4)

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, angle_head=self.one2one_cv4)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode predicted bounding boxes and class probabilities, concatenated with rotation angles."""
        # For decode_bboxes convenience
        self.angle = x["angle"]
        preds = super()._inference(x)
        return torch.cat([preds, x["angle"]], dim=1)

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module, cls_head: torch.nn.Module, angle_head: torch.nn.Module
    ) -> dict[str, torch.Tensor]:
        """Concatenates and returns predicted bounding boxes, class probabilities, and angles."""
        preds = super().forward_head(x, box_head, cls_head)
        if angle_head is not None:
            bs = x[0].shape[0]  # batch size
            angle = torch.cat(
                [angle_head[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2
            )  # OBB theta logits
            angle = (angle.sigmoid() - 0.25) * math.pi  # [-pi/4, 3pi/4]
            preds["angle"] = angle
        return preds

    def decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """Decode rotated bounding boxes."""
        return dist2rbox(bboxes, self.angle, anchors, dim=1)

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Post-process YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc + ne) with last dimension
                format [x, y, w, h, class_probs, angle].

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 7) and last
                dimension format [x, y, w, h, max_class_prob, class_index, angle].
        """
        boxes, scores, angle = preds.split([4, self.nc, self.ne], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        angle = angle.gather(dim=1, index=idx.repeat(1, 1, self.ne))
        return torch.cat([boxes, scores, conf, angle], dim=-1)

    def fuse(self) -> None:
        """Remove the one2many head for inference optimization."""
        self.cv2 = self.cv3 = self.cv4 = None


class OBB26(OBB):
    """YOLO26 OBB detection head for detection with rotation models. This class extends the OBB head with modified angle
    processing that outputs raw angle predictions without sigmoid transformation, compared to the original
    OBB class.

    Attributes:
        ne (int): Number of extra parameters.
        cv4 (nn.ModuleList): Convolution layers for angle prediction.
        angle (torch.Tensor): Predicted rotation angles.

    Methods:
        forward_head: Concatenate and return predicted bounding boxes, class probabilities, and raw angles.

    Examples:
        Create an OBB26 detection head
        >>> obb26 = OBB26(nc=80, ne=1, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = obb26(x)
    """

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module, cls_head: torch.nn.Module, angle_head: torch.nn.Module
    ) -> dict[str, torch.Tensor]:
        """Concatenates and returns predicted bounding boxes, class probabilities, and raw angles."""
        preds = Detect.forward_head(self, x, box_head, cls_head)
        if angle_head is not None:
            bs = x[0].shape[0]  # batch size
            angle = torch.cat(
                [angle_head[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2
            )  # OBB theta logits (raw output without sigmoid transformation)
            preds["angle"] = angle
        return preds


class Pose(Detect):
    """YOLO Pose head for keypoints models.

    This class extends the Detect head to include keypoint prediction capabilities for pose estimation tasks.

    Attributes:
        kpt_shape (tuple): Number of keypoints and dimensions (2 for x,y or 3 for x,y,visible).
        nk (int): Total number of keypoint values.
        cv4 (nn.ModuleList): Convolution layers for keypoint prediction.

    Methods:
        forward: Perform forward pass through YOLO model and return predictions.
        kpts_decode: Decode keypoints from predictions.

    Examples:
        Create a pose detection head
        >>> pose = Pose(nc=80, kpt_shape=(17, 3), ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = pose(x)
    """

    def __init__(self, nc: int = 80, kpt_shape: tuple = (17, 3), reg_max=16, end2end=False, ch: tuple = ()):
        """Initialize YOLO network with default parameters and Convolutional Layers.

        Args:
            nc (int): Number of classes.
            kpt_shape (tuple): Number of keypoints, number of dims (2 for x,y or 3 for x,y,visible).
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, reg_max, end2end, ch)
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)
        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for backward compatibility."""
        return dict(box_head=self.cv2, cls_head=self.cv3, pose_head=self.cv4)

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, pose_head=self.one2one_cv4)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode predicted bounding boxes and class probabilities, concatenated with keypoints."""
        preds = super()._inference(x)
        return torch.cat([preds, self.kpts_decode(x["kpts"])], dim=1)

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module, cls_head: torch.nn.Module, pose_head: torch.nn.Module
    ) -> dict[str, torch.Tensor]:
        """Concatenates and returns predicted bounding boxes, class probabilities, and keypoints."""
        preds = super().forward_head(x, box_head, cls_head)
        if pose_head is not None:
            bs = x[0].shape[0]  # batch size
            preds["kpts"] = torch.cat([pose_head[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], 2)
        return preds

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Post-process YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc + nk) with last dimension
                format [x1, y1, x2, y2, class_probs, keypoints].

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6 + self.nk) and
                last dimension format [x1, y1, x2, y2, max_class_prob, class_index, keypoints].
        """
        boxes, scores, kpts = preds.split([4, self.nc, self.nk], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        kpts = kpts.gather(dim=1, index=idx.repeat(1, 1, self.nk))
        return torch.cat([boxes, scores, conf, kpts], dim=-1)

    def fuse(self) -> None:
        """Remove the one2many head for inference optimization."""
        self.cv2 = self.cv3 = self.cv4 = None

    def kpts_decode(self, kpts: torch.Tensor) -> torch.Tensor:
        """Decode keypoints from predictions."""
        ndim = self.kpt_shape[1]
        bs = kpts.shape[0]
        if self.export:
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                if NOT_MACOS14:
                    y[:, 2::ndim].sigmoid_()
                else:  # Apple macOS14 MPS bug https://github.com/ultralytics/ultralytics/pull/21878
                    y[:, 2::ndim] = y[:, 2::ndim].sigmoid()
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y


class Pose26(Pose):
    """YOLO26 Pose head for keypoints models.

    This class extends the Pose head with normalizing flow for keypoint prediction in pose estimation tasks.

    Attributes:
        kpt_shape (tuple): Number of keypoints and dimensions (2 for x,y or 3 for x,y,visible).
        nk (int): Total number of keypoint values.
        cv4 (nn.ModuleList): Convolution layers for keypoint prediction.

    Methods:
        forward: Perform forward pass through YOLO model and return predictions.
        kpts_decode: Decode keypoints from predictions.

    Examples:
        Create a pose detection head
        >>> pose = Pose26(nc=80, kpt_shape=(17, 3), ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = pose(x)
    """

    def __init__(self, nc: int = 80, kpt_shape: tuple = (17, 3), reg_max=16, end2end=False, ch: tuple = ()):
        """Initialize YOLO network with default parameters and Convolutional Layers.

        Args:
            nc (int): Number of classes.
            kpt_shape (tuple): Number of keypoints, number of dims (2 for x,y or 3 for x,y,visible).
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, kpt_shape, reg_max, end2end, ch)
        self.flow_model = RealNVP()

        c4 = max(ch[0] // 4, kpt_shape[0] * (kpt_shape[1] + 2))
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3)) for x in ch)

        self.cv4_kpts = nn.ModuleList(nn.Conv2d(c4, self.nk, 1) for _ in ch)
        self.nk_sigma = kpt_shape[0] * 2  # sigma_x, sigma_y for each keypoint
        self.cv4_sigma = nn.ModuleList(nn.Conv2d(c4, self.nk_sigma, 1) for _ in ch)

        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)
            self.one2one_cv4_kpts = copy.deepcopy(self.cv4_kpts)
            self.one2one_cv4_sigma = copy.deepcopy(self.cv4_sigma)

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for backward compatibility."""
        return dict(
            box_head=self.cv2,
            cls_head=self.cv3,
            pose_head=self.cv4,
            kpts_head=self.cv4_kpts,
            kpts_sigma_head=self.cv4_sigma,
        )

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(
            box_head=self.one2one_cv2,
            cls_head=self.one2one_cv3,
            pose_head=self.one2one_cv4,
            kpts_head=self.one2one_cv4_kpts,
            kpts_sigma_head=self.one2one_cv4_sigma,
        )

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: torch.nn.Module,
        cls_head: torch.nn.Module,
        pose_head: torch.nn.Module,
        kpts_head: torch.nn.Module,
        kpts_sigma_head: torch.nn.Module,
    ) -> dict[str, torch.Tensor]:
        """Concatenates and returns predicted bounding boxes, class probabilities, and keypoints."""
        preds = Detect.forward_head(self, x, box_head, cls_head)
        if pose_head is not None:
            bs = x[0].shape[0]  # batch size
            features = [pose_head[i](x[i]) for i in range(self.nl)]
            preds["kpts"] = torch.cat([kpts_head[i](features[i]).view(bs, self.nk, -1) for i in range(self.nl)], 2)
            if self.training:
                preds["kpts_sigma"] = torch.cat(
                    [kpts_sigma_head[i](features[i]).view(bs, self.nk_sigma, -1) for i in range(self.nl)], 2
                )
        return preds

    def fuse(self) -> None:
        """Remove the one2many head for inference optimization."""
        super().fuse()
        self.cv4_kpts = self.cv4_sigma = self.flow_model = self.one2one_cv4_sigma = None

    def kpts_decode(self, kpts: torch.Tensor) -> torch.Tensor:
        """Decode keypoints from predictions."""
        ndim = self.kpt_shape[1]
        bs = kpts.shape[0]
        if self.export:
            y = kpts.view(bs, *self.kpt_shape, -1)
            # NCNN fix
            a = (y[:, :, :2] + self.anchors) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                if NOT_MACOS14:
                    y[:, 2::ndim].sigmoid_()
                else:  # Apple macOS14 MPS bug https://github.com/ultralytics/ultralytics/pull/21878
                    y[:, 2::ndim] = y[:, 2::ndim].sigmoid()
            y[:, 0::ndim] = (y[:, 0::ndim] + self.anchors[0]) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] + self.anchors[1]) * self.strides
            return y


class Classify(nn.Module):
    """YOLO classification head, i.e. x(b,c1,20,20) to x(b,c2).

    This class implements a classification head that transforms feature maps into class predictions.

    Attributes:
        export (bool): Export mode flag.
        conv (Conv): Convolutional layer for feature transformation.
        pool (nn.AdaptiveAvgPool2d): Global average pooling layer.
        drop (nn.Dropout): Dropout layer for regularization.
        linear (nn.Linear): Linear layer for final classification.

    Methods:
        forward: Perform forward pass on input feature maps.

    Examples:
        Create a classification head
        >>> classify = Classify(c1=1024, c2=1000)
        >>> x = torch.randn(1, 1024, 20, 20)
        >>> output = classify(x)
    """

    export = False  # export mode

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int | None = None, g: int = 1):
        """Initialize YOLO classification head to transform input tensor from (b,c1,20,20) to (b,c2) shape.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output classes.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
        """
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x: list[torch.Tensor] | torch.Tensor) -> torch.Tensor | tuple:
        """Perform forward pass on input feature maps."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        if self.training:
            return x
        y = x.softmax(1)  # get final output
        return y if self.export else (y, x)


class WorldDetect(Detect):
    """Head for integrating YOLO detection models with semantic understanding from text embeddings.

    This class extends the standard Detect head to incorporate text embeddings for enhanced semantic understanding in
    object detection tasks.

    Attributes:
        cv3 (nn.ModuleList): Convolution layers for embedding features.
        cv4 (nn.ModuleList): Contrastive head layers for text-vision alignment.

    Methods:
        forward: Concatenate and return predicted bounding boxes and class probabilities.
        bias_init: Initialize detection head biases.

    Examples:
        Create a WorldDetect head
        >>> world_detect = WorldDetect(nc=80, embed=512, with_bn=False, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> text = torch.randn(1, 80, 512)
        >>> outputs = world_detect(x, text)
    """

    def __init__(
        self,
        nc: int = 80,
        embed: int = 512,
        with_bn: bool = False,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple = (),
    ):
        """Initialize YOLO detection layer with nc classes and layer channels ch.

        Args:
            nc (int): Number of classes.
            embed (int): Embedding dimension.
            with_bn (bool): Whether to use batch normalization in contrastive head.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, reg_max=reg_max, end2end=end2end, ch=ch)
        c3 = max(ch[0], min(self.nc, 100))
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch)
        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)

    def forward(self, x: list[torch.Tensor], text: torch.Tensor) -> dict[str, torch.Tensor] | tuple:
        """Concatenate and return predicted bounding boxes and class probabilities."""
        feats = [xi.clone() for xi in x]  # save original features for anchor generation
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv4[i](self.cv3[i](x[i]), text)), 1)
        self.no = self.nc + self.reg_max * 4  # self.nc could be changed when inference with different texts
        bs = x[0].shape[0]
        x_cat = torch.cat([xi.view(bs, self.no, -1) for xi in x], 2)
        boxes, scores = x_cat.split((self.reg_max * 4, self.nc), 1)
        preds = dict(boxes=boxes, scores=scores, feats=feats)
        if self.training:
            return preds
        y = self._inference(preds)
        return y if self.export else (y, preds)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            # b[-1].bias.data[:] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


class LRPCHead(nn.Module):
    """Lightweight Region Proposal and Classification Head for efficient object detection.

    This head combines region proposal filtering with classification to enable efficient detection with dynamic
    vocabulary support.

    Attributes:
        vocab (nn.Module): Vocabulary/classification layer.
        pf (nn.Module): Proposal filter module.
        loc (nn.Module): Localization module.
        enabled (bool): Whether the head is enabled.

    Methods:
        conv2linear: Convert a 1x1 convolutional layer to a linear layer.
        forward: Process classification and localization features to generate detection proposals.

    Examples:
        Create an LRPC head
        >>> vocab = nn.Conv2d(256, 80, 1)
        >>> pf = nn.Conv2d(256, 1, 1)
        >>> loc = nn.Conv2d(256, 4, 1)
        >>> head = LRPCHead(vocab, pf, loc, enabled=True)
    """

    def __init__(self, vocab: nn.Module, pf: nn.Module, loc: nn.Module, enabled: bool = True):
        """Initialize LRPCHead with vocabulary, proposal filter, and localization components.

        Args:
            vocab (nn.Module): Vocabulary/classification module.
            pf (nn.Module): Proposal filter module.
            loc (nn.Module): Localization module.
            enabled (bool): Whether to enable the head functionality.
        """
        super().__init__()
        self.vocab = self.conv2linear(vocab) if enabled else vocab
        self.pf = pf
        self.loc = loc
        self.enabled = enabled

    @staticmethod
    def conv2linear(conv: nn.Conv2d) -> nn.Linear:
        """Convert a 1x1 convolutional layer to a linear layer."""
        assert isinstance(conv, nn.Conv2d) and conv.kernel_size == (1, 1)
        linear = nn.Linear(conv.in_channels, conv.out_channels)
        linear.weight.data = conv.weight.view(conv.out_channels, -1).data
        linear.bias.data = conv.bias.data
        return linear

    def forward(self, cls_feat: torch.Tensor, loc_feat: torch.Tensor, conf: float) -> tuple[tuple, torch.Tensor]:
        """Process classification and localization features to generate detection proposals."""
        if self.enabled:
            pf_score = self.pf(cls_feat)[0, 0].flatten(0)
            mask = pf_score.sigmoid() > conf
            cls_feat = cls_feat.flatten(2).transpose(-1, -2)
            cls_feat = self.vocab(cls_feat[:, mask] if conf else cls_feat * mask.unsqueeze(-1).int())
            return self.loc(loc_feat), cls_feat.transpose(-1, -2), mask
        else:
            cls_feat = self.vocab(cls_feat)
            loc_feat = self.loc(loc_feat)
            return (
                loc_feat,
                cls_feat.flatten(2),
                torch.ones(cls_feat.shape[2] * cls_feat.shape[3], device=cls_feat.device, dtype=torch.bool),
            )


class YOLOEDetect(Detect):
    """Head for integrating YOLO detection models with semantic understanding from text embeddings.

    This class extends the standard Detect head to support text-guided detection with enhanced semantic understanding
    through text embeddings and visual prompt embeddings.

    Attributes:
        is_fused (bool): Whether the model is fused for inference.
        cv3 (nn.ModuleList): Convolution layers for embedding features.
        cv4 (nn.ModuleList): Contrastive head layers for text-vision alignment.
        reprta (Residual): Residual block for text prompt embeddings.
        savpe (SAVPE): Spatial-aware visual prompt embeddings module.
        embed (int): Embedding dimension.

    Methods:
        fuse: Fuse text features with model weights for efficient inference.
        get_tpe: Get text prompt embeddings with normalization.
        get_vpe: Get visual prompt embeddings with spatial awareness.
        forward_lrpc: Process features with fused text embeddings for prompt-free model.
        forward: Process features with class prompt embeddings to generate detections.
        bias_init: Initialize biases for detection heads.

    Examples:
        Create a YOLOEDetect head
        >>> yoloe_detect = YOLOEDetect(nc=80, embed=512, with_bn=True, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> cls_pe = torch.randn(1, 80, 512)
        >>> outputs = yoloe_detect(x, cls_pe)
    """

    is_fused = False

    def __init__(
        self, nc: int = 80, embed: int = 512, with_bn: bool = False, reg_max=16, end2end=False, ch: tuple = ()
    ):
        """Initialize YOLO detection layer with nc classes and layer channels ch.

        Args:
            nc (int): Number of classes.
            embed (int): Embedding dimension.
            with_bn (bool): Whether to use batch normalization in contrastive head.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, reg_max, end2end, ch)
        c3 = max(ch[0], min(self.nc, 100))
        assert c3 <= embed
        assert with_bn
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, embed, 1),
                )
                for x in ch
            )
        )
        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)
        if end2end:
            self.one2one_cv3 = copy.deepcopy(self.cv3)  # overwrite with new cv3
            self.one2one_cv4 = copy.deepcopy(self.cv4)

        self.reprta = Residual(SwiGLUFFN(embed, embed))
        self.savpe = SAVPE(ch, c3, embed)
        self.embed = embed

    @smart_inference_mode()
    def fuse(self, txt_feats: torch.Tensor = None):
        """Fuse text features with model weights for efficient inference."""
        if txt_feats is None:  # means eliminate one2many branch
            self.cv2 = self.cv3 = self.cv4 = None
            return
        if self.is_fused:
            return

        assert not self.training
        txt_feats = txt_feats.to(torch.float32).squeeze(0)
        self._fuse_tp(txt_feats, self.cv3, self.cv4)
        if self.end2end:
            self._fuse_tp(txt_feats, self.one2one_cv3, self.one2one_cv4)
        del self.reprta
        self.reprta = nn.Identity()
        self.is_fused = True

    def _fuse_tp(self, txt_feats: torch.Tensor, cls_head: torch.nn.Module, bn_head: torch.nn.Module) -> None:
        """Fuse text prompt embeddings with model weights for efficient inference."""
        for cls_h, bn_h in zip(cls_head, bn_head):
            assert isinstance(cls_h, nn.Sequential)
            assert isinstance(bn_h, BNContrastiveHead)
            conv = cls_h[-1]
            assert isinstance(conv, nn.Conv2d)
            logit_scale = bn_h.logit_scale
            bias = bn_h.bias
            norm = bn_h.norm

            t = txt_feats * logit_scale.exp()
            conv: nn.Conv2d = fuse_conv_and_bn(conv, norm)

            w = conv.weight.data.squeeze(-1).squeeze(-1)
            b = conv.bias.data

            w = t @ w
            b1 = (t @ b.reshape(-1).unsqueeze(-1)).squeeze(-1)
            b2 = torch.ones_like(b1) * bias

            conv = (
                nn.Conv2d(
                    conv.in_channels,
                    w.shape[0],
                    kernel_size=1,
                )
                .requires_grad_(False)
                .to(conv.weight.device)
            )

            conv.weight.data.copy_(w.unsqueeze(-1).unsqueeze(-1))
            conv.bias.data.copy_(b1 + b2)
            cls_h[-1] = conv

            bn_h.fuse()

    def get_tpe(self, tpe: torch.Tensor | None) -> torch.Tensor | None:
        """Get text prompt embeddings with normalization."""
        return None if tpe is None else F.normalize(self.reprta(tpe), dim=-1, p=2)

    def get_vpe(self, x: list[torch.Tensor], vpe: torch.Tensor) -> torch.Tensor:
        """Get visual prompt embeddings with spatial awareness."""
        if vpe.shape[1] == 0:  # no visual prompt embeddings
            return torch.zeros(x[0].shape[0], 0, self.embed, device=x[0].device)
        if vpe.ndim == 4:  # (B, N, H, W)
            vpe = self.savpe(x, vpe)
        assert vpe.ndim == 3  # (B, N, D)
        return vpe

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        """Process features with class prompt embeddings to generate detections."""
        if hasattr(self, "lrpc"):  # for prompt-free inference
            return self.forward_lrpc(x[:3])
        return super().forward(x)

    def forward_lrpc(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        """Process features with fused text embeddings to generate detections for prompt-free model."""
        boxes, scores, index = [], [], []
        bs = x[0].shape[0]
        cv2 = self.cv2 if not self.end2end else self.one2one_cv2
        cv3 = self.cv3 if not self.end2end else self.one2one_cv3
        for i in range(self.nl):
            cls_feat = cv3[i](x[i])
            loc_feat = cv2[i](x[i])
            assert isinstance(self.lrpc[i], LRPCHead)
            box, score, idx = self.lrpc[i](
                cls_feat,
                loc_feat,
                0 if self.export and not self.dynamic else getattr(self, "conf", 0.001),
            )
            boxes.append(box.view(bs, self.reg_max * 4, -1))
            scores.append(score)
            index.append(idx)
        preds = dict(boxes=torch.cat(boxes, 2), scores=torch.cat(scores, 2), feats=x, index=torch.cat(index))
        y = self._inference(preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def _get_decode_boxes(self, x):
        """Decode predicted bounding boxes for inference."""
        dbox = super()._get_decode_boxes(x)
        if hasattr(self, "lrpc"):
            dbox = dbox if self.export and not self.dynamic else dbox[..., x["index"]]
        return dbox

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for v3/v5/v8/v9/v11 backward compatibility."""
        return dict(box_head=self.cv2, cls_head=self.cv3, contrastive_head=self.cv4)

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, contrastive_head=self.one2one_cv4)

    def forward_head(self, x, box_head, cls_head, contrastive_head):
        """Concatenates and returns predicted bounding boxes, class probabilities, and contrastive scores."""
        assert len(x) == 4, f"Expected 4 features including 3 feature maps and 1 text embeddings, but got {len(x)}."
        if box_head is None or cls_head is None:  # for fused inference
            return dict()
        bs = x[0].shape[0]  # batch size
        boxes = torch.cat([box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        self.nc = x[-1].shape[1]
        scores = torch.cat(
            [contrastive_head[i](cls_head[i](x[i]), x[-1]).reshape(bs, self.nc, -1) for i in range(self.nl)], dim=-1
        )
        self.no = self.nc + self.reg_max * 4  # self.nc could be changed when inference with different texts
        return dict(boxes=boxes, scores=scores, feats=x[:3])

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        for i, (a, b, c) in enumerate(
            zip(self.one2many["box_head"], self.one2many["cls_head"], self.one2many["contrastive_head"])
        ):
            a[-1].bias.data[:] = 2.0  # box
            b[-1].bias.data[:] = 0.0
            c.bias.data[:] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)
        if self.end2end:
            for i, (a, b, c) in enumerate(
                zip(self.one2one["box_head"], self.one2one["cls_head"], self.one2one["contrastive_head"])
            ):
                a[-1].bias.data[:] = 2.0  # box
                b[-1].bias.data[:] = 0.0
                c.bias.data[:] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)


class YOLOESegment(YOLOEDetect):
    """YOLO segmentation head with text embedding capabilities.

    This class extends YOLOEDetect to include mask prediction capabilities for instance segmentation tasks with
    text-guided semantic understanding.

    Attributes:
        nm (int): Number of masks.
        npr (int): Number of protos.
        proto (Proto): Prototype generation module.
        cv5 (nn.ModuleList): Convolution layers for mask coefficients.

    Methods:
        forward: Return model outputs and mask coefficients.

    Examples:
        Create a YOLOESegment head
        >>> yoloe_segment = YOLOESegment(nc=80, nm=32, npr=256, embed=512, with_bn=True, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> text = torch.randn(1, 80, 512)
        >>> outputs = yoloe_segment(x, text)
    """

    def __init__(
        self,
        nc: int = 80,
        nm: int = 32,
        npr: int = 256,
        embed: int = 512,
        with_bn: bool = False,
        reg_max=16,
        end2end=False,
        ch: tuple = (),
    ):
        """Initialize YOLOESegment with class count, mask parameters, and embedding dimensions.

        Args:
            nc (int): Number of classes.
            nm (int): Number of masks.
            npr (int): Number of protos.
            embed (int): Embedding dimension.
            with_bn (bool): Whether to use batch normalization in contrastive head.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, embed, with_bn, reg_max, end2end, ch)
        self.nm = nm
        self.npr = npr
        self.proto = Proto(ch[0], self.npr, self.nm)

        c5 = max(ch[0] // 4, self.nm)
        self.cv5 = nn.ModuleList(nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nm, 1)) for x in ch)
        if end2end:
            self.one2one_cv5 = copy.deepcopy(self.cv5)

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for v3/v5/v8/v9/v11 backward compatibility."""
        return dict(box_head=self.cv2, cls_head=self.cv3, mask_head=self.cv5, contrastive_head=self.cv4)

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(
            box_head=self.one2one_cv2,
            cls_head=self.one2one_cv3,
            mask_head=self.one2one_cv5,
            contrastive_head=self.one2one_cv4,
        )

    def forward_lrpc(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        """Process features with fused text embeddings to generate detections for prompt-free model."""
        boxes, scores, index = [], [], []
        bs = x[0].shape[0]
        cv2 = self.cv2 if not self.end2end else self.one2one_cv2
        cv3 = self.cv3 if not self.end2end else self.one2one_cv3
        cv5 = self.cv5 if not self.end2end else self.one2one_cv5
        for i in range(self.nl):
            cls_feat = cv3[i](x[i])
            loc_feat = cv2[i](x[i])
            assert isinstance(self.lrpc[i], LRPCHead)
            box, score, idx = self.lrpc[i](
                cls_feat,
                loc_feat,
                0 if self.export and not self.dynamic else getattr(self, "conf", 0.001),
            )
            boxes.append(box.view(bs, self.reg_max * 4, -1))
            scores.append(score)
            index.append(idx)
        mc = torch.cat([cv5[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
        index = torch.cat(index)
        preds = dict(
            boxes=torch.cat(boxes, 2),
            scores=torch.cat(scores, 2),
            feats=x,
            index=index,
            mask_coefficient=mc * index.int() if self.export and not self.dynamic else mc[..., index],
        )
        y = self._inference(preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        outputs = super().forward(x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self.proto(x[0])  # mask protos
        if isinstance(preds, dict):  # training and validating during training
            if self.end2end:
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = proto.detach()
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode predicted bounding boxes and class probabilities, concatenated with mask coefficients."""
        preds = super()._inference(x)
        return torch.cat([preds, x["mask_coefficient"]], dim=1)

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: torch.nn.Module,
        cls_head: torch.nn.Module,
        mask_head: torch.nn.Module,
        contrastive_head: torch.nn.Module,
    ) -> dict[str, torch.Tensor]:
        """Concatenates and returns predicted bounding boxes, class probabilities, and mask coefficients."""
        preds = super().forward_head(x, box_head, cls_head, contrastive_head)
        if mask_head is not None:
            bs = x[0].shape[0]  # batch size
            preds["mask_coefficient"] = torch.cat([mask_head[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
        return preds

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Post-process YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc + nm) with last dimension
                format [x1, y1, x2, y2, class_probs, mask_coefficient].

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6 + nm) and last
                dimension format [x1, y1, x2, y2, max_class_prob, class_index, mask_coefficient].
        """
        boxes, scores, mask_coefficient = preds.split([4, self.nc, self.nm], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        mask_coefficient = mask_coefficient.gather(dim=1, index=idx.repeat(1, 1, self.nm))
        return torch.cat([boxes, scores, conf, mask_coefficient], dim=-1)

    def fuse(self, txt_feats: torch.Tensor = None):
        """Fuse text features with model weights for efficient inference."""
        super().fuse(txt_feats)
        if txt_feats is None:  # means eliminate one2many branch
            self.cv5 = None
            if hasattr(self.proto, "fuse"):
                self.proto.fuse()
            return


class YOLOESegment26(YOLOESegment):
    """YOLOE-style segmentation head module using Proto26 for mask generation.

    This class extends the YOLOESegment functionality to include segmentation capabilities by integrating a Proto26
    generation module and convolutional layers to predict mask coefficients.

    Args:
        nc (int): Number of classes. Defaults to 80.
        nm (int): Number of masks. Defaults to 32.
        npr (int): Number of prototype channels. Defaults to 256.
        embed (int): Embedding dimensionality. Defaults to 512.
        with_bn (bool): Whether to use Batch Normalization. Defaults to False.
        reg_max (int): Maximum number of DFL channels. Defaults to 16.
        end2end (bool): Whether to use end-to-end detection mode. Defaults to False.
        ch (tuple[int, ...]): Input channels for each scale.

    Attributes:
        nm (int): Number of segmentation masks.
        npr (int): Number of prototype channels.
        proto (Proto26): Prototype generation module for segmentation.
        cv5 (nn.ModuleList): Convolutional layers for generating mask coefficients from features.
        one2one_cv5 (nn.ModuleList, optional): Deep copy of cv5 for end-to-end detection branches.
    """

    def __init__(
        self,
        nc: int = 80,
        nm: int = 32,
        npr: int = 256,
        embed: int = 512,
        with_bn: bool = False,
        reg_max=16,
        end2end=False,
        ch: tuple = (),
    ):
        """Initialize YOLOESegment26 with class count, mask parameters, and embedding dimensions."""
        YOLOEDetect.__init__(self, nc, embed, with_bn, reg_max, end2end, ch)
        self.nm = nm
        self.npr = npr
        self.proto = Proto26(ch, self.npr, self.nm, nc)  # protos

        c5 = max(ch[0] // 4, self.nm)
        self.cv5 = nn.ModuleList(nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nm, 1)) for x in ch)
        if end2end:
            self.one2one_cv5 = copy.deepcopy(self.cv5)

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        outputs = YOLOEDetect.forward(self, x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self.proto([xi.detach() for xi in x], return_semseg=False)  # mask protos

        if isinstance(preds, dict):  # training and validating during training
            if self.end2end and not hasattr(self, "lrpc"):  # not prompt-free
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = proto.detach()
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)


class RTDETRDecoder(nn.Module):
    """Real-Time Deformable Transformer Decoder (RTDETRDecoder) module for object detection.

    This decoder module utilizes Transformer architecture along with deformable convolutions to predict bounding boxes
    and class labels for objects in an image. It integrates features from multiple layers and runs through a series of
    Transformer decoder layers to output the final predictions.

    Attributes:
        export (bool): Export mode flag.
        hidden_dim (int): Dimension of hidden layers.
        nhead (int): Number of heads in multi-head attention.
        nl (int): Number of feature levels.
        nc (int): Number of classes.
        num_queries (int): Number of query points.
        num_decoder_layers (int): Number of decoder layers.
        input_proj (nn.ModuleList): Input projection layers for backbone features.
        decoder (DeformableTransformerDecoder): Transformer decoder module.
        denoising_class_embed (nn.Embedding): Class embeddings for denoising.
        num_denoising (int): Number of denoising queries.
        label_noise_ratio (float): Label noise ratio for training.
        box_noise_scale (float): Box noise scale for training.
        learnt_init_query (bool): Whether to learn initial query embeddings.
        tgt_embed (nn.Embedding): Target embeddings for queries.
        query_pos_head (MLP): Query position head.
        enc_output (nn.Sequential): Encoder output layers.
        enc_score_head (nn.Linear): Encoder score prediction head.
        enc_bbox_head (MLP): Encoder bbox prediction head.
        dec_score_head (nn.ModuleList): Decoder score prediction heads.
        dec_bbox_head (nn.ModuleList): Decoder bbox prediction heads.

    Methods:
        forward: Run forward pass and return bounding box and classification scores.

    Examples:
        Create an RTDETRDecoder
        >>> decoder = RTDETRDecoder(nc=80, ch=(512, 1024, 2048), hd=256, nq=300)
        >>> x = [torch.randn(1, 512, 64, 64), torch.randn(1, 1024, 32, 32), torch.randn(1, 2048, 16, 16)]
        >>> outputs = decoder(x)
    """

    export = False  # export mode
    shapes = []
    anchors = torch.empty(0)
    valid_mask = torch.empty(0)
    dynamic = False

    def __init__(
        self,
        nc: int = 80,
        ch: tuple = (512, 1024, 2048),
        hd: int = 256,  # hidden dim
        nq: int = 300,  # num queries
        ndp: int = 4,  # num decoder points
        nh: int = 8,  # num head
        ndl: int = 6,  # num decoder layers
        d_ffn: int = 1024,  # dim of feedforward
        dropout: float = 0.0,
        act: nn.Module = nn.ReLU(),
        eval_idx: int = -1,
        # Training args
        nd: int = 100,  # num denoising
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        learnt_init_query: bool = False,
    ):
        """Initialize the RTDETRDecoder module with the given parameters.

        Args:
            nc (int): Number of classes.
            ch (tuple): Channels in the backbone feature maps.
            hd (int): Dimension of hidden layers.
            nq (int): Number of query points.
            ndp (int): Number of decoder points.
            nh (int): Number of heads in multi-head attention.
            ndl (int): Number of decoder layers.
            d_ffn (int): Dimension of the feed-forward networks.
            dropout (float): Dropout rate.
            act (nn.Module): Activation function.
            eval_idx (int): Evaluation index.
            nd (int): Number of denoising.
            label_noise_ratio (float): Label noise ratio.
            box_noise_scale (float): Box noise scale.
            learnt_init_query (bool): Whether to learn initial query embeddings.
        """
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # num level
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        # Backbone feature projection
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)
        # NOTE: simplified version but it's not consistent with .pt weights.
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # Denoising part
        self.denoising_class_embed = nn.Embedding(nc, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # Decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        # Encoder head
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        # Decoder head
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        self._reset_parameters()

    def forward(self, x: list[torch.Tensor], batch: dict | None = None) -> tuple | torch.Tensor:
        """Run the forward pass of the module, returning bounding box and classification scores for the input.

        Args:
            x (list[torch.Tensor]): List of feature maps from the backbone.
            batch (dict, optional): Batch information for training.

        Returns:
            outputs (tuple | torch.Tensor): During training, returns a tuple of bounding boxes, scores, and other
                metadata. During inference, returns a tensor of shape (bs, 300, 4+nc) containing bounding boxes and
                class scores.
        """
        from ultralytics.models.utils.ops import get_cdn_group

        # Input projection and embedding
        feats, shapes = self._get_encoder_input(x)

        # Prepare denoising training
        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch,
            self.nc,
            self.num_queries,
            self.denoising_class_embed.weight,
            self.num_denoising,
            self.label_noise_ratio,
            self.box_noise_scale,
            self.training,
        )

        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        # Decoder
        dec_bboxes, dec_scores = self.decoder(
            embed,
            refer_bbox,
            feats,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
        )
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return x
        # (bs, 300, 4+nc)
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, x)

    @staticmethod
    def _generate_anchors(
        shapes: list[list[int]],
        grid_size: float = 0.05,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        eps: float = 1e-2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate anchor bounding boxes for given shapes with specific grid size and validate them.

        Args:
            shapes (list): List of feature map shapes.
            grid_size (float, optional): Base size of grid cells.
            dtype (torch.dtype, optional): Data type for tensors.
            device (str, optional): Device to create tensors on.
            eps (float, optional): Small value for numerical stability.

        Returns:
            anchors (torch.Tensor): Generated anchor boxes.
            valid_mask (torch.Tensor): Valid mask for anchors.
        """
        anchors = []
        for i, (h, w) in enumerate(shapes):
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_11 else torch.meshgrid(sy, sx)
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)

            valid_WH = torch.tensor([w, h], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0**i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float("inf"))
        return anchors, valid_mask

    def _get_encoder_input(self, x: list[torch.Tensor]) -> tuple[torch.Tensor, list[list[int]]]:
        """Process and return encoder inputs by getting projection features from input and concatenating them.

        Args:
            x (list[torch.Tensor]): List of feature maps from the backbone.

        Returns:
            feats (torch.Tensor): Processed features.
            shapes (list): List of feature map shapes.
        """
        # Get projection features
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        # Get encoder inputs
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c]
            feats.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2]
            shapes.append([h, w])

        # [b, h*w, c]
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(
        self,
        feats: torch.Tensor,
        shapes: list[list[int]],
        dn_embed: torch.Tensor | None = None,
        dn_bbox: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate and prepare the input required for the decoder from the provided features and shapes.

        Args:
            feats (torch.Tensor): Processed features from encoder.
            shapes (list): List of feature map shapes.
            dn_embed (torch.Tensor, optional): Denoising embeddings.
            dn_bbox (torch.Tensor, optional): Denoising bounding boxes.

        Returns:
            embeddings (torch.Tensor): Query embeddings for decoder.
            refer_bbox (torch.Tensor): Reference bounding boxes.
            enc_bboxes (torch.Tensor): Encoded bounding boxes.
            enc_scores (torch.Tensor): Encoded scores.
        """
        bs = feats.shape[0]
        if self.dynamic or self.shapes != shapes:
            self.anchors, self.valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
            self.shapes = shapes

        # Prepare input for decoder
        features = self.enc_output(self.valid_mask * feats)  # bs, h*w, 256
        enc_outputs_scores = self.enc_score_head(features)  # (bs, h*w, nc)

        # Query selection
        # (bs*num_queries,)
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        # (bs*num_queries,)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        # (bs, num_queries, 256)
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        # (bs, num_queries, 4)
        top_k_anchors = self.anchors[:, topk_ind].view(bs, self.num_queries, -1)

        # Dynamic anchors + static content
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors

        enc_bboxes = refer_bbox.sigmoid()
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    def _reset_parameters(self):
        """Initialize or reset the parameters of the model's various components with predefined weights and biases."""
        # Class and bbox head init
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        # NOTE: the weight initialization in `linear_init` would cause NaN when training with custom datasets.
        # linear_init(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.0)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            # linear_init(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.0)
            constant_(reg_.layers[-1].bias, 0.0)

        linear_init(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)


class v10Detect(Detect):
    """v10 Detection head from https://arxiv.org/pdf/2405.14458.

    This class implements the YOLOv10 detection head with dual-assignment training and consistent dual predictions for
    improved efficiency and performance.

    Attributes:
        end2end (bool): End-to-end detection mode.
        max_det (int): Maximum number of detections.
        cv3 (nn.ModuleList): Light classification head layers.
        one2one_cv3 (nn.ModuleList): One-to-one classification head layers.

    Methods:
        __init__: Initialize the v10Detect object with specified number of classes and input channels.
        forward: Perform forward pass of the v10Detect module.
        bias_init: Initialize biases of the Detect module.
        fuse: Remove the one2many head for inference optimization.

    Examples:
        Create a v10Detect head
        >>> v10_detect = v10Detect(nc=80, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = v10_detect(x)
    """

    end2end = True

    def __init__(self, nc: int = 80, ch: tuple = ()):
        """Initialize the v10Detect object with the specified number of classes and input channels.

        Args:
            nc (int): Number of classes.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, end2end=True, ch=ch)
        c3 = max(ch[0], min(self.nc, 100))  # channels
        # Light cls head
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1)),
                nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.one2one_cv3 = copy.deepcopy(self.cv3)

    def fuse(self):
        """Remove the one2many head for inference optimization."""
        self.cv2 = self.cv3 = None

class _HCFDFusion(nn.Module):
    """Fuse final P2 detail with backbone P3/P4 information at P3 scale."""

    def __init__(
        self,
        c_p2: int,
        c_backbone_p3: int,
        c_backbone_p4: int,
        c_fuse: int = 32,
    ):
        super().__init__()

        # P2执行Space-to-Depth后通道变成4*c_p2
        self.p2_proj = Conv(c_p2 * 4, c_fuse, k=1, s=1)
        self.p3_proj = Conv(c_backbone_p3, c_fuse, k=1, s=1)
        self.p4_proj = Conv(c_backbone_p4, c_fuse, k=1, s=1)

        # 为P2/P3/P4生成三个逐位置尺度权重
        self.scale_gate = nn.Sequential(
            nn.Conv2d(
                c_fuse * 3,
                c_fuse * 3,
                kernel_size=3,
                padding=1,
                groups=c_fuse * 3,
                bias=False,
            ),
            nn.BatchNorm2d(c_fuse * 3),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_fuse * 3, 3, kernel_size=1, bias=True),
        )

    @staticmethod
    def space_to_depth(x: torch.Tensor) -> torch.Tensor:
        """B,C,H,W -> B,4C,H/2,W/2."""
        return torch.cat(
            (
                x[..., ::2, ::2],
                x[..., 1::2, ::2],
                x[..., ::2, 1::2],
                x[..., 1::2, 1::2],
            ),
            dim=1,
        )

    def forward(
        self,
        p2: torch.Tensor,
        backbone_p3: torch.Tensor,
        backbone_p4: torch.Tensor,
    ) -> torch.Tensor:

        # P2: 160×160 -> 80×80，保留2×2像素信息
        e2 = self.p2_proj(self.space_to_depth(p2))

        # P3保持80×80
        e3 = self.p3_proj(backbone_p3)

        # P4: 40×40 -> 80×80
        e4 = self.p4_proj(backbone_p4)
        e4 = F.interpolate(
            e4,
            size=e3.shape[-2:],
            mode="nearest",
        )

        fusion_input = torch.cat((e2, e3, e4), dim=1)

        # B×3×80×80，在尺度维度做Softmax
        scale_weight = torch.softmax(
            self.scale_gate(fusion_input),
            dim=1,
        )

        fused = (
            scale_weight[:, 0:1] * e2
            + scale_weight[:, 1:2] * e3
            + scale_weight[:, 2:3] * e4
        )

        return fused

class HCFDDetect(Detect):
    """
    P2-based four-scale Detect with hierarchical cross-scale
    fine-grained classification enhancement.

    YAML input order:
        0: final P2, layer19
        1: final P3, layer22
        2: final P4, layer25
        3: final P5, layer28
        4: backbone P3, layer4
        5: backbone P4, layer6
    """

    def __init__(
        self,
        nc: int = 80,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple = (),
        fusion_ch: int = 32,
        embed_dim: int = 16,
        num_prototypes: int = 2,
        proto_weight: float = 0.05,
        proto_margin: float = 0.15,
    ):
        if end2end:
            raise NotImplementedError(
                "HCFDDetect currently supports end2end=False only."
            )

        if len(ch) != 6:
            raise ValueError(
                "HCFDDetect expects 6 inputs: "
                "P2, P3, P4, P5, backbone-P3, backbone-P4."
            )

        # 前四个输入才是真正的检测尺度
        prediction_channels = tuple(ch[:4])

        # Detect只能使用四个检测尺度构建
        super().__init__(
            nc=nc,
            reg_max=reg_max,
            end2end=end2end,
            ch=prediction_channels,
        )

        self.hcfd_fusion = _HCFDFusion(
            c_p2=prediction_channels[0],
            c_backbone_p3=ch[4],
            c_backbone_p4=ch[5],
            c_fuse=fusion_ch,
        )

        # 融合特征分别映射到P2/P3通道
        self.fusion_to_p2 = Conv(
            fusion_ch,
            prediction_channels[0],
            k=1,
            s=1,
            act=False,
        )
        self.fusion_to_p3 = Conv(
            fusion_ch,
            prediction_channels[1],
            k=1,
            s=1,
            act=False,
        )

        # 初始化为0，初始状态严格等价于普通P2 Detect
        self.gamma_p2 = nn.Parameter(torch.zeros(1))
        self.gamma_p3 = nn.Parameter(torch.zeros(1))

        # 训练期细粒度嵌入分支
        self.hcfd_embed = nn.ModuleList(
            (
                nn.Conv2d(
                    prediction_channels[0],
                    embed_dim,
                    kernel_size=1,
                ),
                nn.Conv2d(
                    prediction_channels[1],
                    embed_dim,
                    kernel_size=1,
                ),
            )
        )

        # 每类K个外观原型，例如完整/遮挡、竖直/侧躺
        self.hcfd_prototypes = nn.Parameter(
            torch.empty(nc, num_prototypes, embed_dim)
        )
        nn.init.normal_(self.hcfd_prototypes, std=0.02)

        self.hcfd_proto_weight = proto_weight
        self.hcfd_proto_margin = proto_margin
        self.hcfd_proto_tau = 0.1

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: torch.nn.Module = None,
        cls_head: torch.nn.Module = None,
    ) -> dict[str, torch.Tensor]:

        if box_head is None or cls_head is None:
            return {}

        # 只取前四层进行预测和Anchor生成
        prediction_features = list(x[: self.nl])

        backbone_p3 = x[4]
        backbone_p4 = x[5]

        # 多尺度融合：B×32×80×80
        fused = self.hcfd_fusion(
            prediction_features[0],
            backbone_p3,
            backbone_p4,
        )

        box_features = prediction_features
        classification_features = prediction_features.copy()

        # P2分类增强
        p2_guidance = self.fusion_to_p2(fused)
        p2_guidance = F.interpolate(
            p2_guidance,
            size=prediction_features[0].shape[-2:],
            mode="nearest",
        )

        classification_features[0] = (
            prediction_features[0]
            + torch.tanh(self.gamma_p2) * p2_guidance
        )

        # P3分类增强
        p3_guidance = self.fusion_to_p3(fused)

        classification_features[1] = (
            prediction_features[1]
            + torch.tanh(self.gamma_p3) * p3_guidance
        )

        # P4/P5分类保持不变
        bs = prediction_features[0].shape[0]

        boxes = torch.cat(
            [
                box_head[i](box_features[i]).view(
                    bs,
                    4 * self.reg_max,
                    -1,
                )
                for i in range(self.nl)
            ],
            dim=-1,
        )

        scores = torch.cat(
            [
                cls_head[i](classification_features[i]).view(
                    bs,
                    self.nc,
                    -1,
                )
                for i in range(self.nl)
            ],
            dim=-1,
        )

        output = {
            "boxes": boxes,
            "scores": scores,

            # 这里只能返回P2/P3/P4/P5四个检测特征
            "feats": prediction_features,
        }

        # 原型分支只在训练时计算
        if self.training:
            output["hcfd_embeddings"] = [
                F.normalize(
                    self.hcfd_embed[i](classification_features[i]),
                    dim=1,
                    eps=1e-6,
                )
                for i in range(2)
            ]

            output["hcfd_prototypes"] = self.hcfd_prototypes
            output["hcfd_proto_weight"] = self.hcfd_proto_weight
            output["hcfd_proto_margin"] = self.hcfd_proto_margin
            output["hcfd_proto_tau"] = self.hcfd_proto_tau

        return output
