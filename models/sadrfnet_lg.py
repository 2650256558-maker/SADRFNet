"""
SADRFNet-LG: Scene-Adaptive Dilated Reparameterized Local--Global Fusion Network for
remote-sensing single-label scene classification.

Recommended local--global design
--------------------------------
1) A complete ImageNet-compatible ResNet-50 backbone is retained. Each ResNet
   residual stage is the LOCAL representation path and focuses on stable
   texture, edge, object-part, and semantic features.
2) For every selected stage, a separate compressed Dilated Reparameterization
   Block (DRB) path builds a GLOBAL representation with a large-kernel
   depthwise branch and several dilated depthwise branches.
3) Local ResNet and global DRB features are not combined as
   ``vanilla + small_alpha * DRB``. A scene-conditioned two-way softmax router
   assigns channel-wise responsibility to the two streams, and a complementary
   difference projection recovers information unique to either stream.
4) ``DRB_STAGE_RATIOS`` controls the depth of the global DRB path while leaving
   all original ResNet bottlenecks intact. With ratio 0.33 in ResNet stage 3,
   the local path keeps all six bottlenecks and the global path uses two DRB
   context units.
5) Multi-stage progressive fusion integrates the coordinated stage features.
   The original final ResNet feature remains the semantic anchor in the final
   scene-adaptive residual fusion module.
6) The multi-branch depthwise structure inside each DRB can still be fused into
   one equivalent large-kernel depthwise convolution for deployment.

Backward compatibility
----------------------
- ``StageDRBConfig``, ``SADRFNet``, and ``sadrfnet50`` keep their public APIs.
- Existing ``DRB_MODE = "adapter"`` now selects the recommended local--global
  coordination design.
- ``DRB_MODE = "replace"`` remains available for direct-replacement ablations.
- Old trained checkpoints are not structurally compatible and must not be
  loaded with ``strict=True`` into this redesigned model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Basic layers
# -----------------------------------------------------------------------------


def _make_divisible(v: int, divisor: int = 8) -> int:
    """Round channel number to be divisible by `divisor`."""
    return int((v + divisor - 1) // divisor * divisor)


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=stride,
        padding=0,
        bias=False,
    )


def conv3x3(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        dilation=dilation,
        bias=False,
    )


class ConvBNAct(nn.Sequential):
    """Conv2d + BatchNorm2d + optional activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        act_layer: Optional[nn.Module] = nn.GELU,
    ) -> None:
        if padding is None:
            padding = kernel_size // 2
        layers: List[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if act_layer is not None:
            layers.append(act_layer())
        super().__init__(*layers)


class SqueezeExcitation(nn.Module):
    """Channel attention for CNN features."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(_make_divisible(channels // reduction), 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        scale = self.fc(self.avg_pool(x))
        return x * scale


class ConvFFN(nn.Module):
    """
    CNN-style feed-forward network.

    It follows a Transformer FFN idea but keeps spatial structure:
    1x1 expansion -> depthwise 3x3 local mixing -> 1x1 projection.
    A residual connection is used to stabilize training on small datasets.
    """

    def __init__(
        self,
        channels: int,
        expansion: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden = _make_divisible(int(channels * expansion))
        self.norm = nn.BatchNorm2d(channels)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.act1 = nn.GELU()
        self.dwconv = nn.Conv2d(
            hidden,
            hidden,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(hidden)
        self.act2 = nn.GELU()
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.act2(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return identity + x


class SEConvFFN(nn.Module):
    """SE + ConvFFN module used to replace final ReLU in modified blocks."""

    def __init__(
        self,
        channels: int,
        se_reduction: int = 16,
        ffn_expansion: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.se = SqueezeExcitation(channels, reduction=se_reduction)
        self.ffn = ConvFFN(channels, expansion=ffn_expansion, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.ffn(self.se(x))


# -----------------------------------------------------------------------------
# Dilated reparameterization block
# -----------------------------------------------------------------------------


class _DWConvBN(nn.Sequential):
    """Depthwise Conv2d + BN branch used inside DRB."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
    ) -> None:
        effective_kernel = dilation * (kernel_size - 1) + 1
        padding = effective_kernel // 2
        super().__init__(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
        )
        self.kernel_size = kernel_size
        self.dilation = dilation


class DilatedReparamSpatialBlock(nn.Module):
    """
    Dilated Reparameterization Block for spatial feature extraction.

    Training-time structure:
        large-kernel depthwise branch
        + several dilated 3x3 depthwise branches
        -> 1x1 pointwise projection

    Deployment-time structure after `switch_to_deploy()`:
        a single equivalent large-kernel depthwise conv
        -> 1x1 pointwise projection

    This block is designed as a drop-in replacement for the 3x3 spatial
    convolution inside a ResNet bottleneck block. It preserves the input/output
    channel number.
    """

    def __init__(
        self,
        channels: int,
        stride: int = 1,
        large_kernel_size: int = 7,
        dilations: Sequence[int] = (1, 2, 3),
        deploy: bool = False,
    ) -> None:
        super().__init__()
        if large_kernel_size % 2 == 0:
            raise ValueError("large_kernel_size must be odd.")
        self.channels = channels
        self.stride = stride
        self.large_kernel_size = large_kernel_size
        self.deploy = deploy

        if deploy:
            self.deploy_dw = nn.Conv2d(
                channels,
                channels,
                kernel_size=large_kernel_size,
                stride=stride,
                padding=large_kernel_size // 2,
                groups=channels,
                bias=True,
            )
        else:
            self.large_branch = _DWConvBN(
                channels,
                kernel_size=large_kernel_size,
                stride=stride,
                dilation=1,
            )
            branches = []
            for d in dilations:
                effective_kernel = d * (3 - 1) + 1
                if effective_kernel > large_kernel_size:
                    raise ValueError(
                        f"dilation={d} produces effective kernel "
                        f"{effective_kernel}, larger than large_kernel_size={large_kernel_size}."
                    )
                branches.append(_DWConvBN(channels, kernel_size=3, stride=stride, dilation=d))
            self.dilated_branches = nn.ModuleList(branches)

        self.pw = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.deploy:
            out = self.deploy_dw(x)
        else:
            out = self.large_branch(x)
            for branch in self.dilated_branches:
                out = out + branch(x)
        out = self.pw(out)
        return out

    @staticmethod
    def _fuse_conv_bn(branch: _DWConvBN) -> Tuple[Tensor, Tensor]:
        conv = branch[0]
        bn = branch[1]
        weight = conv.weight
        if conv.bias is None:
            bias = torch.zeros(weight.size(0), device=weight.device, dtype=weight.dtype)
        else:
            bias = conv.bias

        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = torch.sqrt(running_var + eps)
        scale = (gamma / std).reshape(-1, 1, 1, 1)
        fused_weight = weight * scale
        fused_bias = beta + (bias - running_mean) * gamma / std
        return fused_weight, fused_bias

    @staticmethod
    def _dilate_kernel(kernel: Tensor, dilation: int) -> Tensor:
        if dilation == 1:
            return kernel
        c, one, k, _ = kernel.shape
        effective = dilation * (k - 1) + 1
        dilated = kernel.new_zeros((c, one, effective, effective))
        dilated[:, :, ::dilation, ::dilation] = kernel
        return dilated

    @staticmethod
    def _pad_to_large_kernel(kernel: Tensor, target_kernel_size: int) -> Tensor:
        current = kernel.size(-1)
        if current == target_kernel_size:
            return kernel
        if current > target_kernel_size:
            raise ValueError("Current kernel is larger than target kernel.")
        pad_total = target_kernel_size - current
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return F.pad(kernel, [pad_left, pad_right, pad_left, pad_right])

    @torch.no_grad()
    def get_equivalent_kernel_bias(self) -> Tuple[Tensor, Tensor]:
        if self.deploy:
            return self.deploy_dw.weight, self.deploy_dw.bias

        kernel, bias = self._fuse_conv_bn(self.large_branch)
        kernel = self._pad_to_large_kernel(kernel, self.large_kernel_size)

        for branch in self.dilated_branches:
            branch_kernel, branch_bias = self._fuse_conv_bn(branch)
            branch_kernel = self._dilate_kernel(branch_kernel, branch.dilation)
            branch_kernel = self._pad_to_large_kernel(branch_kernel, self.large_kernel_size)
            kernel = kernel + branch_kernel
            bias = bias + branch_bias
        return kernel, bias

    @torch.no_grad()
    def switch_to_deploy(self) -> None:
        """Fuse depthwise branches into a single large-kernel depthwise conv."""
        if self.deploy:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.deploy_dw = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=self.large_kernel_size,
            stride=self.stride,
            padding=self.large_kernel_size // 2,
            groups=self.channels,
            bias=True,
        ).to(kernel.device)
        self.deploy_dw.weight.data.copy_(kernel)
        self.deploy_dw.bias.data.copy_(bias)

        del self.large_branch
        del self.dilated_branches
        self.deploy = True


# -----------------------------------------------------------------------------
# ResNet bottleneck blocks
# -----------------------------------------------------------------------------


class VanillaBottleneck(nn.Module):
    """Standard ResNet bottleneck used in Stage 1 and unmodified blocks."""

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = conv3x3(width, width, stride=stride, groups=groups, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class DRBBottleneck(nn.Module):
    """
    ResNet bottleneck with Dilated Reparam Spatial Block.

    Difference from vanilla bottleneck:
    - The 3x3 spatial conv is replaced by `DilatedReparamSpatialBlock`.
    - The final ReLU after residual addition is replaced by SE + ConvFFN.
    """

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        large_kernel_size: int = 7,
        dilations: Sequence[int] = (1, 2, 3),
        ffn_expansion: float = 2.0,
        ffn_dropout: float = 0.0,
        deploy: bool = False,
    ) -> None:
        super().__init__()
        if groups != 1:
            raise ValueError("DRBBottleneck currently expects groups=1 for a ResNet-50 style model.")
        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.act1 = nn.GELU()

        self.spatial = DilatedReparamSpatialBlock(
            channels=width,
            stride=stride,
            large_kernel_size=large_kernel_size,
            dilations=dilations,
            deploy=deploy,
        )
        self.act2 = nn.GELU()

        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.downsample = downsample
        self.post = SEConvFFN(
            planes * self.expansion,
            se_reduction=16,
            ffn_expansion=ffn_expansion,
            dropout=ffn_dropout,
        )
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.spatial(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.post(out)
        return out


class DRBAdapterBottleneck(nn.Module):
    """
    ResNet bottleneck with a weakly gated DRB adapter branch.

    Instead of replacing the ImageNet-pretrained 3x3 spatial convolution, this
    block preserves the original conv2/bn2 path and adds a DRB branch in
    parallel:

        out = BN(Conv3x3(out)) + sigmoid(adapter_logit) * DRB(out)

    Therefore, the standard conv1/bn1, conv2/bn2, conv3/bn3 and downsample
    names remain compatible with torchvision ResNet-50 checkpoints. The DRB
    branch starts with a small gate and can become stronger on datasets where
    larger receptive fields or dilated spatial context are useful.
    """

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        large_kernel_size: int = 7,
        dilations: Sequence[int] = (1, 2, 3),
        adapter_init: float = -4.0,
        use_se_ffn: bool = True,
        ffn_expansion: float = 1.0,
        ffn_dropout: float = 0.0,
        deploy: bool = False,
    ) -> None:
        super().__init__()
        if groups != 1:
            raise ValueError("DRBAdapterBottleneck currently expects groups=1 for a ResNet-50 style model.")
        width = int(planes * (base_width / 64.0)) * groups

        # Keep standard ResNet names so ImageNet weights can be loaded.
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.relu = nn.ReLU(inplace=True)

        # Pretrained vanilla spatial path: this is the learned ImageNet 3x3
        # feature extractor and should not be removed during fine-tuning.
        self.conv2 = conv3x3(width, width, stride=stride, groups=groups)
        self.bn2 = nn.BatchNorm2d(width)

        # New DRB auxiliary branch. It is gated so that the block initially
        # behaves close to the pretrained vanilla bottleneck.
        self.drb = DilatedReparamSpatialBlock(
            channels=width,
            stride=stride,
            large_kernel_size=large_kernel_size,
            dilations=dilations,
            deploy=deploy,
        )
        self.adapter_logit = nn.Parameter(torch.tensor(float(adapter_init)))

        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

        if use_se_ffn:
            self.post = SEConvFFN(
                planes * self.expansion,
                se_reduction=16,
                ffn_expansion=ffn_expansion,
                dropout=ffn_dropout,
            )
        else:
            self.post = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        vanilla = self.bn2(self.conv2(out))
        drb_out = self.drb(out)
        alpha = torch.sigmoid(self.adapter_logit)
        out = vanilla + alpha * drb_out
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.post(out)
        return out

    def get_adapter_strength(self) -> Tensor:
        """Return the current scalar DRB adapter strength alpha in [0, 1]."""
        return torch.sigmoid(self.adapter_logit.detach())


# -----------------------------------------------------------------------------
# Progressive fusion head
# -----------------------------------------------------------------------------


class StageResidualRefine(nn.Module):
    """
    Residual refinement for each stage feature before progressive fusion.

    This implements the "Residual" operation in the user's diagram. It first
    projects all stage features to the same channel dimension and then applies
    lightweight depthwise residual refinement.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = ConvBNAct(in_channels, out_channels, kernel_size=1)
        self.refine = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=out_channels,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        return self.act(x + self.refine(x))


class DownsampleForFusion(nn.Module):
    """Downsample a fusion feature by 2x before fusing with the next stage."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.down = ConvBNAct(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: Tensor, target_size: Tuple[int, int]) -> Tensor:
        x = self.down(x)
        if x.shape[-2:] != target_size:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return x


class PairFusionBlock(nn.Module):
    """
    Adaptive fusion of two stage features with the same spatial size/channels.

    The fusion gate is computed from global context and normalized by softmax.
    This is more suitable for scene classification than direct summation because
    different scenes may rely on different semantic levels.
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 32)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, 2, kernel_size=1, bias=True),
        )
        self.concat_proj = ConvBNAct(channels * 2, channels, kernel_size=1)
        self.refine = StageResidualRefine(channels, channels)

    def forward(self, x_low: Tensor, x_high: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x_low: downsampled lower-stage feature.
            x_high: current higher-stage feature.
        Returns:
            fused feature and 2-way adaptive fusion weights.
        """
        if x_low.shape[-2:] != x_high.shape[-2:]:
            x_low = F.interpolate(x_low, size=x_high.shape[-2:], mode="bilinear", align_corners=False)
        x_cat = torch.cat([x_low, x_high], dim=1)
        weights = torch.softmax(self.gate(x_cat), dim=1)
        weighted = weights[:, 0:1] * x_low + weights[:, 1:2] * x_high
        fused = weighted + self.concat_proj(x_cat)
        fused = self.refine(fused)
        return fused, weights.flatten(1)


class ProgressiveStageFusion(nn.Module):
    """Progressively fuse x1, x2, x3, x4 into f4."""

    def __init__(self, in_channels_list: Sequence[int], fusion_channels: int = 256) -> None:
        super().__init__()
        if len(in_channels_list) != 4:
            raise ValueError("in_channels_list must contain four stage channel numbers.")
        self.residuals = nn.ModuleList(
            [StageResidualRefine(c, fusion_channels) for c in in_channels_list]
        )
        self.down12 = DownsampleForFusion(fusion_channels)
        self.down23 = DownsampleForFusion(fusion_channels)
        self.down34 = DownsampleForFusion(fusion_channels)
        self.fuse2 = PairFusionBlock(fusion_channels)
        self.fuse3 = PairFusionBlock(fusion_channels)
        self.fuse4 = PairFusionBlock(fusion_channels)

    def forward(self, features: Sequence[Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        x1, x2, x3, x4 = features
        r1 = self.residuals[0](x1)
        r2 = self.residuals[1](x2)
        r3 = self.residuals[2](x3)
        r4 = self.residuals[3](x4)

        d1 = self.down12(r1, target_size=r2.shape[-2:])
        f2, w2 = self.fuse2(d1, r2)

        d2 = self.down23(f2, target_size=r3.shape[-2:])
        f3, w3 = self.fuse3(d2, r3)

        d3 = self.down34(f3, target_size=r4.shape[-2:])
        f4, w4 = self.fuse4(d3, r4)

        gates = {
            "stage2_pair_weights": w2,
            "stage3_pair_weights": w3,
            "stage4_pair_weights": w4,
        }
        return f4, gates


class SceneAdaptiveResidualFusion(nn.Module):
    """
    Scene-adaptive residual fusion for remote-sensing scene classification.

    Version 2 is deliberately more conservative than the earlier implementation:
    the original stage-4 feature is kept as a 2048-d semantic anchor by default
    instead of being compressed through a randomly initialized 4096->512 dual-
    pooling projection. The progressively fused feature only acts as a weakly
    gated residual supplement at the beginning of fine-tuning.

    This is more suitable for small scene-classification datasets because it
    preserves the ImageNet-pretrained stage-4 representation and prevents
    randomly initialized multi-stage fusion features from dominating early
    training.
    """

    def __init__(
        self,
        original_channels: int,
        fusion_channels: int,
        embed_dim: int,
        dropout: float = 0.2,
        init_residual_gate: float = -4.0,
    ) -> None:
        super().__init__()
        self.original_channels = original_channels
        self.fusion_channels = fusion_channels
        self.embed_dim = embed_dim

        if original_channels == embed_dim:
            self.original_proj = nn.Identity()
        else:
            self.original_proj = nn.Linear(original_channels, embed_dim, bias=False)
        self.original_norm = nn.LayerNorm(embed_dim)

        # f4 is still described by GAP+GMP because it carries object/detail cues
        # from the progressive multi-stage branch.
        self.fused_proj = nn.Sequential(
            nn.Linear(fusion_channels * 2, embed_dim, bias=False),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        gate_hidden = max(embed_dim // 4, 128)
        self.gate_mlp = nn.Sequential(
            nn.Linear(embed_dim * 4, gate_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, embed_dim),
        )
        # A weaker initial gate makes the model start close to the pretrained
        # ResNet semantic path and learn the fusion residual only when useful.
        nn.init.constant_(self.gate_mlp[-1].bias, init_residual_gate)

        detail_hidden = max(embed_dim // 2, 256)
        self.detail_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, detail_hidden, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(detail_hidden, embed_dim, bias=False),
        )

        self.out_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _avg_pool(x: Tensor) -> Tensor:
        return F.adaptive_avg_pool2d(x, 1).flatten(1)

    @staticmethod
    def _dual_pool(x: Tensor) -> Tensor:
        avg = F.adaptive_avg_pool2d(x, 1).flatten(1)
        mx = F.adaptive_max_pool2d(x, 1).flatten(1)
        return torch.cat([avg, mx], dim=1)

    def forward(self, x4: Tensor, f4: Tensor) -> Tuple[Tensor, Tensor]:
        # Semantic anchor: use GAP(x4), preserving the pretrained 2048-d stage-4
        # feature when embed_dim=2048.
        z_original = self.original_proj(self._avg_pool(x4))
        z_original = self.original_norm(z_original)

        # Multi-stage detail supplement.
        z_fused = self.fused_proj(self._dual_pool(f4))

        gate_input = torch.cat(
            [z_original, z_fused, torch.abs(z_original - z_fused), z_original * z_fused],
            dim=1,
        )
        residual_gate = torch.sigmoid(self.gate_mlp(gate_input))
        residual_detail = self.detail_proj(z_fused - z_original)

        z = z_original + residual_gate * residual_detail
        z = self.out_norm(z)
        z = self.dropout(z)
        return z, residual_gate


# -----------------------------------------------------------------------------
# Full model
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class StageDRBConfig:
    """Configuration for local--global coordination at one ResNet stage.

    ``mode`` controls how DRB is used:

    - ``"adapter"`` / ``"dual"`` / ``"local_global"``: keep the complete
      ImageNet-compatible ResNet stage as the local path, construct a separate
      DRB global-context path from the stage output, and fuse the two paths with
      dynamic complementary routing. This is the recommended mode.
    - ``"replace"``: retain the older behavior that directly replaces the last
      bottlenecks of a stage with :class:`DRBBottleneck`.

    ``replace_ratio`` is reinterpreted as the depth ratio of the DRB global path
    in the recommended local--global mode. For example, a ratio of 0.33 in
    ResNet stage 3 (six bottlenecks) creates two global DRB context units while
    leaving all six original ResNet bottlenecks intact.

    ``adapter_init`` is kept only for backward compatibility with existing
    configuration files. The new local--global mode does not use a weak scalar
    adapter gate.
    """

    replace_ratio: float
    large_kernel_size: int
    dilations: Tuple[int, ...]
    ffn_expansion: float
    mode: str = "adapter"
    adapter_init: float = -4.0
    use_se_ffn: bool = True
    global_residual_init: float = 0.05
    fusion_local_prior: float = 0.50


class GlobalDRBContextUnit(nn.Module):
    """One residual DRB unit dedicated to large-context spatial modeling.

    The unit operates in a compressed channel space. The large-kernel branch
    captures broad scene layout, while the dilated branches capture context at
    several effective receptive fields. A small layer scale stabilizes transfer
    learning without defining the DRB stream as an auxiliary branch.
    """

    def __init__(
        self,
        channels: int,
        large_kernel_size: int,
        dilations: Sequence[int],
        ffn_expansion: float = 1.0,
        use_se_ffn: bool = True,
        deploy: bool = False,
        layer_scale_init: float = 1e-2,
    ) -> None:
        super().__init__()
        self.pre_norm = nn.BatchNorm2d(channels)
        self.spatial = DilatedReparamSpatialBlock(
            channels=channels,
            stride=1,
            large_kernel_size=large_kernel_size,
            dilations=dilations,
            deploy=deploy,
        )
        self.act = nn.GELU()
        self.channel_refine = (
            SqueezeExcitation(channels, reduction=16)
            if use_se_ffn
            else nn.Identity()
        )
        self.layer_scale = nn.Parameter(
            torch.full((channels,), float(layer_scale_init))
        )
        self.ffn = (
            ConvFFN(
                channels,
                expansion=max(float(ffn_expansion), 1.0),
                dropout=0.0,
            )
            if use_se_ffn
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.pre_norm(x)
        y = self.spatial(y)
        y = self.act(y)
        y = self.channel_refine(y)
        x = x + self.layer_scale.view(1, -1, 1, 1) * y
        return self.ffn(x)


class GlobalDRBContextPath(nn.Module):
    """Build an explicit global representation from a ResNet stage feature.

    The full ResNet stage output is treated as the local representation. This
    path compresses it, applies several global DRB context units, projects it
    back to the original channel dimension, and forms a distinct global stream.
    The residual initialization makes the global stream initially close to the
    local stream, so equal-status fusion is stable from the first epoch.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        depth: int,
        large_kernel_size: int,
        dilations: Sequence[int],
        ffn_expansion: float,
        use_se_ffn: bool,
        deploy: bool,
        global_residual_init: float = 0.05,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("GlobalDRBContextPath depth must be >= 1.")
        self.reduce = ConvBNAct(
            channels,
            hidden_channels,
            kernel_size=1,
            act_layer=nn.GELU,
        )
        self.blocks = nn.Sequential(
            *[
                GlobalDRBContextUnit(
                    channels=hidden_channels,
                    large_kernel_size=large_kernel_size,
                    dilations=dilations,
                    ffn_expansion=ffn_expansion,
                    use_se_ffn=use_se_ffn,
                    deploy=deploy,
                )
                for _ in range(depth)
            ]
        )
        self.expand = nn.Sequential(
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.global_residual_scale = nn.Parameter(
            torch.tensor(float(global_residual_init))
        )

    def forward(self, local_feature: Tensor) -> Tensor:
        context = self.reduce(local_feature)
        context = self.blocks(context)
        context = self.expand(context)
        scale = torch.tanh(self.global_residual_scale)
        return local_feature + scale * context


class LocalGlobalStageFusion(nn.Module):
    """Complementary routing between local ResNet and global DRB streams.

    The two streams have equal architectural status. A scene-conditioned router
    predicts two channel-wise weights normalized by a two-way softmax, so every
    output channel explicitly allocates its representation capacity between
    local texture/structure and global multi-scale context. A low-rank
    difference path models information that is present in one stream but absent
    from the other.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        local_prior: float = 0.50,
    ) -> None:
        super().__init__()
        if not 0.0 < local_prior < 1.0:
            raise ValueError("local_prior must be in (0, 1).")
        hidden = max(channels // reduction, 32)
        self.channels = channels
        self.local_prior = float(local_prior)

        # GAP and GMP are computed for both streams: 4C -> hidden -> 2C.
        self.router = nn.Sequential(
            nn.Conv2d(channels * 4, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, channels * 2, kernel_size=1, bias=True),
        )

        # Low-rank complementary difference projection. Its final BN is
        # zero-initialized after the model-wide initialization, preserving the
        # pretrained local representation at the start of fine-tuning.
        self.difference_proj = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.out_act = nn.ReLU(inplace=False)
        self.last_branch_balance: Optional[Tensor] = None

    @staticmethod
    def _dual_descriptor(x: Tensor) -> Tuple[Tensor, Tensor]:
        avg = F.adaptive_avg_pool2d(x, 1)
        mx = F.adaptive_max_pool2d(x, 1)
        return avg, mx

    def reset_fusion_parameters(self) -> None:
        """Initialize routing with a chosen local/global prior and no diff term."""
        final_router = self.router[-1]
        nn.init.zeros_(final_router.weight)
        local_bias = float(torch.log(torch.tensor(self.local_prior)))
        global_bias = float(torch.log(torch.tensor(1.0 - self.local_prior)))
        with torch.no_grad():
            final_router.bias[: self.channels].fill_(local_bias)
            final_router.bias[self.channels :].fill_(global_bias)

        final_bn = self.difference_proj[-1]
        nn.init.zeros_(final_bn.weight)
        nn.init.zeros_(final_bn.bias)

    def forward(
        self,
        local_feature: Tensor,
        global_feature: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if local_feature.shape != global_feature.shape:
            raise ValueError(
                "Local and global stage features must have identical shapes, "
                f"got {tuple(local_feature.shape)} and {tuple(global_feature.shape)}."
            )

        local_avg, local_max = self._dual_descriptor(local_feature)
        global_avg, global_max = self._dual_descriptor(global_feature)
        descriptor = torch.cat(
            [local_avg, local_max, global_avg, global_max], dim=1
        )
        logits = self.router(descriptor)
        b = logits.size(0)
        weights = logits.view(b, 2, self.channels, 1, 1)
        weights = torch.softmax(weights, dim=1)

        local_weight = weights[:, 0]
        global_weight = weights[:, 1]
        mixture = (
            local_weight * local_feature
            + global_weight * global_feature
        )

        complementary = self.difference_proj(
            torch.abs(local_feature - global_feature)
        )
        fused = self.out_act(mixture + complementary)

        # [B, 2], useful for later visualization and paper analysis.
        balance = weights.mean(dim=2).flatten(1)
        self.last_branch_balance = balance.detach().mean(dim=0)
        return fused, balance

    @torch.no_grad()
    def get_branch_balance(self) -> Tuple[float, float]:
        if self.last_branch_balance is not None:
            return (
                float(self.last_branch_balance[0].cpu()),
                float(self.last_branch_balance[1].cpu()),
            )
        return self.local_prior, 1.0 - self.local_prior


class SADRFNet(nn.Module):
    """ResNet-50 with explicit local--global DRB coordination and fusion.

    In the recommended ``adapter`` mode, every selected stage contains:

    1. a complete ImageNet-pretrained ResNet residual group for local texture,
       edge, object-part, and stable semantic representation;
    2. a separate compressed DRB context path for broad scene layout and
       multi-scale global context;
    3. a scene-conditioned complementary fusion module that dynamically assigns
       local/global responsibility per channel.

    The previous weak ``vanilla + alpha * DRB`` bottleneck is no longer used in
    this mode. Direct DRB replacement remains available through ``mode=replace``
    for controlled ablation experiments.
    """

    def __init__(
        self,
        num_classes: int,
        in_chans: int = 3,
        layers: Sequence[int] = (3, 4, 6, 3),
        fusion_channels: int = 256,
        embed_dim: int = 2048,
        zero_init_residual: bool = True,
        groups: int = 1,
        width_per_group: int = 64,
        dropout: float = 0.2,
        return_features: bool = False,
        deploy: bool = False,
        stage_cfgs: Optional[Sequence[StageDRBConfig]] = None,
    ) -> None:
        super().__init__()
        if len(layers) != 4:
            raise ValueError("layers must contain four stage depths.")
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.return_features = return_features
        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group

        if stage_cfgs is None:
            # A conservative default for three standard remote-sensing datasets.
            # Stage 3 receives two DRB global context units (6 * 0.33 ~= 2),
            # while the complete pretrained ResNet-50 local path is retained.
            stage_cfgs = (
                StageDRBConfig(0.00, 3, (1,), 1.0, mode="adapter"),
                StageDRBConfig(0.00, 5, (1, 2), 1.0, mode="adapter"),
                StageDRBConfig(
                    0.33,
                    7,
                    (1, 2, 3),
                    1.0,
                    mode="adapter",
                    global_residual_init=0.05,
                    fusion_local_prior=0.50,
                ),
                StageDRBConfig(0.00, 5, (1, 2), 1.0, mode="adapter"),
            )
        if len(stage_cfgs) != 4:
            raise ValueError("stage_cfgs must contain four StageDRBConfig objects.")
        self.stage_cfgs = tuple(stage_cfgs)

        self.conv1 = nn.Conv2d(
            in_chans,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # In local--global mode these remain complete standard ResNet stages,
        # maximizing compatibility with ImageNet-pretrained ResNet-50 weights.
        self.layer1 = self._make_layer(
            planes=64,
            blocks=layers[0],
            stride=1,
            cfg=stage_cfgs[0],
            force_vanilla=True,
            deploy=deploy,
        )
        self.layer2 = self._make_layer(
            planes=128,
            blocks=layers[1],
            stride=2,
            cfg=stage_cfgs[1],
            deploy=deploy,
        )
        self.layer3 = self._make_layer(
            planes=256,
            blocks=layers[2],
            stride=2,
            cfg=stage_cfgs[2],
            deploy=deploy,
        )
        self.layer4 = self._make_layer(
            planes=512,
            blocks=layers[3],
            stride=2,
            cfg=stage_cfgs[3],
            deploy=deploy,
        )

        stage_channels = [256, 512, 1024, 2048]
        stage_hidden_channels = [64, 128, 256, 512]

        # Attribute names contain ``drb`` so the existing optimizer grouping in
        # main.py assigns these new modules the new-module learning rate.
        self.drb_stage_depths: List[int] = []
        self.drb_global_paths = nn.ModuleList()
        self.drb_stage_fusions = nn.ModuleList()
        for blocks, channels, hidden_channels, cfg in zip(
            layers,
            stage_channels,
            stage_hidden_channels,
            stage_cfgs,
        ):
            mode = cfg.mode.lower()
            depth = int(round(blocks * float(cfg.replace_ratio)))
            depth = min(max(depth, 0), blocks)
            use_local_global = mode in {"adapter", "dual", "local_global"}
            active_depth = depth if use_local_global else 0
            self.drb_stage_depths.append(active_depth)

            if active_depth > 0:
                self.drb_global_paths.append(
                    GlobalDRBContextPath(
                        channels=channels,
                        hidden_channels=hidden_channels,
                        depth=active_depth,
                        large_kernel_size=cfg.large_kernel_size,
                        dilations=cfg.dilations,
                        ffn_expansion=cfg.ffn_expansion,
                        use_se_ffn=cfg.use_se_ffn,
                        deploy=deploy,
                        global_residual_init=cfg.global_residual_init,
                    )
                )
                self.drb_stage_fusions.append(
                    LocalGlobalStageFusion(
                        channels=channels,
                        reduction=16,
                        local_prior=cfg.fusion_local_prior,
                    )
                )
            else:
                self.drb_global_paths.append(nn.Identity())
                self.drb_stage_fusions.append(nn.Identity())

        self.progressive_fusion = ProgressiveStageFusion(
            in_channels_list=stage_channels,
            fusion_channels=fusion_channels,
        )
        self.final_fusion = SceneAdaptiveResidualFusion(
            original_channels=stage_channels[-1],
            fusion_channels=fusion_channels,
            embed_dim=embed_dim,
            dropout=dropout,
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

        self._init_weights(zero_init_residual=zero_init_residual)
        self._reset_local_global_fusion_parameters()

    def _make_layer(
        self,
        planes: int,
        blocks: int,
        stride: int,
        cfg: StageDRBConfig,
        force_vanilla: bool = False,
        deploy: bool = False,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * VanillaBottleneck.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * VanillaBottleneck.expansion, stride),
                nn.BatchNorm2d(planes * VanillaBottleneck.expansion),
            )

        mode = cfg.mode.lower()
        if mode not in {"adapter", "dual", "local_global", "replace"}:
            raise ValueError(
                f"Unsupported StageDRBConfig.mode={cfg.mode!r}. "
                "Use 'adapter', 'dual', 'local_global', or 'replace'."
            )

        replace_count = int(round(blocks * float(cfg.replace_ratio)))
        replace_count = min(max(replace_count, 0), blocks)
        first_drb_index = blocks - replace_count

        layers_out: List[nn.Module] = []
        for block_idx in range(blocks):
            block_stride = stride if block_idx == 0 else 1
            block_downsample = downsample if block_idx == 0 else None

            # Only the explicit ablation mode directly replaces bottlenecks.
            use_replacement = (
                not force_vanilla
                and mode == "replace"
                and block_idx >= first_drb_index
            )
            if use_replacement:
                block = DRBBottleneck(
                    inplanes=self.inplanes,
                    planes=planes,
                    stride=block_stride,
                    downsample=block_downsample,
                    groups=self.groups,
                    base_width=self.base_width,
                    large_kernel_size=cfg.large_kernel_size,
                    dilations=cfg.dilations,
                    ffn_expansion=cfg.ffn_expansion,
                    ffn_dropout=0.0,
                    deploy=deploy,
                )
            else:
                block = VanillaBottleneck(
                    inplanes=self.inplanes,
                    planes=planes,
                    stride=block_stride,
                    downsample=block_downsample,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=1,
                )
            layers_out.append(block)
            self.inplanes = planes * VanillaBottleneck.expansion

        return nn.Sequential(*layers_out)

    def _init_weights(self, zero_init_residual: bool = True) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, (VanillaBottleneck, DRBBottleneck, DRBAdapterBottleneck)):
                    nn.init.constant_(m.bn3.weight, 0)

    def _reset_local_global_fusion_parameters(self) -> None:
        for module in self.drb_stage_fusions:
            if isinstance(module, LocalGlobalStageFusion):
                module.reset_fusion_parameters()

    def _coordinate_stage(
        self,
        stage_index: int,
        local_feature: Tensor,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        depth = self.drb_stage_depths[stage_index]
        if depth <= 0:
            return local_feature, local_feature, None

        global_feature = self.drb_global_paths[stage_index](local_feature)
        fused_feature, balance = self.drb_stage_fusions[stage_index](
            local_feature,
            global_feature,
        )
        return fused_feature, global_feature, balance

    def forward_features(self, x: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        local_x1 = self.layer1(x)
        x1, global_x1, balance1 = self._coordinate_stage(0, local_x1)

        local_x2 = self.layer2(x1)
        x2, global_x2, balance2 = self._coordinate_stage(1, local_x2)

        local_x3 = self.layer3(x2)
        x3, global_x3, balance3 = self._coordinate_stage(2, local_x3)

        local_x4 = self.layer4(x3)
        x4, global_x4, balance4 = self._coordinate_stage(3, local_x4)

        # Progressive fusion receives coordinated local--global representations.
        f4, progressive_gates = self.progressive_fusion([x1, x2, x3, x4])

        # Preserve the final ResNet local feature as the stable semantic anchor;
        # the progressive branch already contains DRB global context.
        z, final_weights = self.final_fusion(local_x4, f4)

        aux: Dict[str, Tensor] = {
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "x4": x4,
            "local_x1": local_x1,
            "local_x2": local_x2,
            "local_x3": local_x3,
            "local_x4": local_x4,
            "global_x1": global_x1,
            "global_x2": global_x2,
            "global_x3": global_x3,
            "global_x4": global_x4,
            "f4": f4,
            "final_residual_gate": final_weights,
            **progressive_gates,
        }
        for stage_idx, balance in enumerate(
            [balance1, balance2, balance3, balance4], start=1
        ):
            if balance is not None:
                aux[f"stage{stage_idx}_local_global_weights"] = balance
        return z, aux

    def forward(self, x: Tensor) -> Union[Tensor, Dict[str, Tensor]]:
        z, aux = self.forward_features(x)
        logits = self.classifier(z)
        if self.return_features:
            aux["embedding"] = z
            aux["logits"] = logits
            return aux
        return logits

    @torch.no_grad()
    def get_local_global_branch_weights(self) -> Dict[str, Dict[str, float]]:
        """Return the latest mean local/global routing weights for active stages."""
        result: Dict[str, Dict[str, float]] = {}
        for idx, module in enumerate(self.drb_stage_fusions, start=1):
            if isinstance(module, LocalGlobalStageFusion):
                local_weight, global_weight = module.get_branch_balance()
                result[f"stage{idx}"] = {
                    "local": local_weight,
                    "global": global_weight,
                }
        return result

    @torch.no_grad()
    def get_drb_adapter_strengths(self) -> Dict[str, float]:
        """Backward-compatible alias returning DRB global routing proportions."""
        balances = self.get_local_global_branch_weights()
        return {
            f"{stage}.global_weight": values["global"]
            for stage, values in balances.items()
        }

    @torch.no_grad()
    def switch_to_deploy(self) -> None:
        """Fuse every training-time DRB spatial branch for deployment."""
        for m in self.modules():
            if isinstance(m, DilatedReparamSpatialBlock):
                m.switch_to_deploy()

# -----------------------------------------------------------------------------
# ImageNet pretrained initialization
# -----------------------------------------------------------------------------


def _extract_state_dict_from_checkpoint(checkpoint):
    """Extract a plain state_dict from common checkpoint formats."""
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def _strip_prefix_if_present(state_dict: Dict[str, Tensor], prefix: str) -> Dict[str, Tensor]:
    if not state_dict:
        return state_dict
    if all(k.startswith(prefix) for k in state_dict.keys()):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def load_imagenet_resnet50_pretrained(
    model: nn.Module,
    pretrained_path: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> Dict[str, object]:
    """
    Partially initialize SADRFNet from ImageNet-pretrained ResNet-50 weights.

    Why partial loading is used:
    In the recommended local--global mode, the stem and all four ResNet local
    stages retain the standard torchvision ResNet-50 parameter names and tensor
    shapes. Consequently, the complete convolutional local backbone can be
    initialized from ImageNet. The separate DRB global paths, local--global
    routers, progressive fusion, scene-adaptive fusion, and classifier are new
    modules and remain randomly initialized.

    Args:
        model: SADRFNet instance.
        pretrained_path: Optional local ResNet-50 ImageNet checkpoint path.
            If empty or None, torchvision.models.resnet50 ImageNet weights are
            used and may be downloaded by torchvision on the first run.
        verbose: Print loading statistics.

    Returns:
        A dictionary containing matched and skipped parameter information.
    """
    if pretrained_path is not None and str(pretrained_path).strip():
        checkpoint = torch.load(str(pretrained_path), map_location="cpu")
        source_state = _extract_state_dict_from_checkpoint(checkpoint)
        source_name = str(pretrained_path)
    else:
        try:
            from torchvision.models import ResNet50_Weights, resnet50
            weights = ResNet50_Weights.IMAGENET1K_V1
            source_state = resnet50(weights=weights).state_dict()
            source_name = "torchvision ResNet50_Weights.IMAGENET1K_V1"
        except Exception as exc:
            raise RuntimeError(
                "Failed to load torchvision ImageNet ResNet-50 weights. "
                "Install torchvision with a compatible PyTorch version, make sure "
                "the server can access the weight file, or set IMAGENET_PRETRAINED_PATH "
                "in main.py to a local resnet50 ImageNet checkpoint."
            ) from exc

    source_state = _extract_state_dict_from_checkpoint(source_state)
    source_state = _strip_prefix_if_present(source_state, "module.")
    source_state = _strip_prefix_if_present(source_state, "backbone.")

    target_state = model.state_dict()
    matched = {}
    skipped = []
    for key, value in source_state.items():
        if key in target_state and tuple(target_state[key].shape) == tuple(value.shape):
            matched[key] = value
        else:
            skipped.append(key)

    updated_state = target_state.copy()
    updated_state.update(matched)
    missing, unexpected = model.load_state_dict(updated_state, strict=False)

    info = {
        "source": source_name,
        "matched_keys": sorted(matched.keys()),
        "num_matched": len(matched),
        "num_source_keys": len(source_state),
        "num_skipped_source_keys": len(skipped),
        "missing_keys_after_load": list(missing),
        "unexpected_keys_after_load": list(unexpected),
    }

    if verbose:
        print("[ImageNet Pretrain] Source:", source_name)
        print(
            "[ImageNet Pretrain] Loaded matched tensors: "
            f"{info['num_matched']}/{info['num_source_keys']}"
        )
        print(
            "[ImageNet Pretrain] The complete ResNet local backbone is preserved. "
            "Newly initialized modules include DRB global-context paths, local-global routers, "
            "progressive fusion, scene-adaptive fusion, and classifier."
        )
    return info


def sadrfnet50(
    num_classes: int,
    in_chans: int = 3,
    fusion_channels: int = 256,
    embed_dim: int = 2048,
    dropout: float = 0.2,
    return_features: bool = False,
    deploy: bool = False,
    stage_cfgs: Optional[Sequence[StageDRBConfig]] = None,
    pretrained: bool = False,
    pretrained_path: Optional[Union[str, Path]] = None,
    pretrained_verbose: bool = True,
) -> SADRFNet:
    """
    Build SADRFNet-50.

    Args:
        num_classes: number of scene classes.
        in_chans: input image channels. Use 3 for RGB scene datasets.
        fusion_channels: channel dimension used by progressive fusion.
        embed_dim: final embedding dimension before classifier.
        dropout: dropout ratio used in final MLP fusion/classification head.
        return_features: if True, return logits and intermediate gates/features.
        deploy: if True, instantiate DRB in deploy form. For normal training,
            keep False and call `model.switch_to_deploy()` before inference export.
        stage_cfgs: optional custom stage-wise DRB policy.
        pretrained: if True, partially load ImageNet-pretrained ResNet-50 weights.
        pretrained_path: optional local ResNet-50 ImageNet checkpoint path. If
            omitted, torchvision ImageNet weights are used.
        pretrained_verbose: whether to print loading statistics.
    """
    model = SADRFNet(
        num_classes=num_classes,
        in_chans=in_chans,
        layers=(3, 4, 6, 3),
        fusion_channels=fusion_channels,
        embed_dim=embed_dim,
        dropout=dropout,
        return_features=return_features,
        deploy=deploy,
        stage_cfgs=stage_cfgs,
    )
    if pretrained:
        load_imagenet_resnet50_pretrained(
            model,
            pretrained_path=pretrained_path,
            verbose=pretrained_verbose,
        )
    return model


# Explicit local--global names for the suffixed model file.
SADRFNetLG = SADRFNet

def sadrfnet50_lg(**kwargs) -> SADRFNetLG:
    """Build the local--global SADRFNet-LG model."""
    return sadrfnet50(**kwargs)


# Backward-compatible aliases retained for ablation and older scripts.
AdaptiveDRBResNet = SADRFNet

def adaptive_drb_resnet50(**kwargs) -> SADRFNet:
    return sadrfnet50(**kwargs)

def drb_resnet50(**kwargs) -> SADRFNet:
    return sadrfnet50(**kwargs)


if __name__ == "__main__":
    # Minimal sanity check. Remove this block if integrating into a large project.
    # Setting the thread number keeps this check fast on small CPU-only machines.
    torch.set_num_threads(1)
    model = sadrfnet50_lg(num_classes=45, in_chans=3, return_features=True)
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224)
        outputs = model(dummy)
    print("logits:", outputs["logits"].shape)
    print("final residual gate:", outputs["final_residual_gate"].shape)
    print("stage2 pair weights:", outputs["stage2_pair_weights"].shape)
    print("stage3 pair weights:", outputs["stage3_pair_weights"].shape)
    print("stage4 pair weights:", outputs["stage4_pair_weights"].shape)
    print("local-global balances:", model.get_local_global_branch_weights())
