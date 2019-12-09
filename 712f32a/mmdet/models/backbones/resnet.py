import collections
import logging
import re
from typing import Dict, List

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.models.plugins import GeneralizedAttention
from mmdet.models.utils import norm as norm_utils
from mmdet.ops import ContextBlock, DeformConv, ModulatedDeformConv
from ..registry import BACKBONES
from ..utils import build_norm_layer, get_conv_layer_module

logger = logging.getLogger(__name__)


class BasicBlock(nn.Module):
    expansion = 1
    _version = 2

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 gcb=None,
                 gen_attention=None):
        super(BasicBlock, self).__init__()
        assert dcn is None, "Not implemented yet."
        assert gen_attention is None, "Not implemented yet."
        assert gcb is None, "Not implemented yet."

        _, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        _, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = get_conv_layer_module(conv_cfg)(
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.norm1 = norm1
        self._register_load_state_dict_pre_hook(_convert_norm_names)
        self.conv2 = get_conv_layer_module(conv_cfg)(
            planes, planes, 3, padding=1, bias=False)
        self.norm2 = norm2

        self.relu = nn.ReLU(inplace=True)
        if downsample is None:
            self.downsample = nn.Identity()
        else:
            self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        assert not with_cp

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        identity_or_downsample = self.downsample(x)

        out += identity_or_downsample
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    _version = 2
    __constants__ = [
        "context_block",
        "conv2_offset",
        "expansion",
        "gen_attention_block",
    ]

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 gcb=None,
                 gen_attention=None):
        """Bottleneck block for ResNet.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer,
        if it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__()
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)
        assert gcb is None or isinstance(gcb, dict)
        assert gen_attention is None or isinstance(gen_attention, dict)

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.gcb = gcb
        self.with_gcb = gcb is not None
        self.gen_attention = gen_attention
        self.with_gen_attention = gen_attention is not None

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        _, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        _, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        _, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)

        self.conv1 = get_conv_layer_module(conv_cfg)(
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.norm1 = norm1
        fallback_on_stride = False
        self.with_modulated_dcn = False
        if self.with_dcn:
            fallback_on_stride = dcn.get('fallback_on_stride', False)
            self.with_modulated_dcn = dcn.get('modulated', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = get_conv_layer_module(conv_cfg)(
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)
            self.conv2_offset = None
        else:
            assert conv_cfg is None, 'conv_cfg must be None for DCN'
            self.deformable_groups = dcn.get('deformable_groups', 1)
            if not self.with_modulated_dcn:
                conv_op = DeformConv
                offset_channels = 18
            else:
                conv_op = ModulatedDeformConv
                offset_channels = 27
            self.conv2_offset = nn.Conv2d(
                planes,
                self.deformable_groups * offset_channels,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation)
            self.conv2 = conv_op(
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                deformable_groups=self.deformable_groups,
                bias=False)
        self.norm2 = norm2
        self.conv3 = get_conv_layer_module(conv_cfg)(
            planes, planes * self.expansion, kernel_size=1, bias=False)
        self.norm3 = norm3
        self._register_load_state_dict_pre_hook(_convert_norm_names)

        self.relu = nn.ReLU(inplace=True)
        if downsample is None:
            self.downsample = nn.Identity()
        else:
            self.downsample = downsample

        if self.with_gcb:
            gcb_inplanes = planes * self.expansion
            self.context_block = ContextBlock(inplanes=gcb_inplanes, **gcb)
        else:
            self.context_block = None

        # gen_attention
        if self.with_gen_attention:
            self.gen_attention_block = GeneralizedAttention(
                planes, **gen_attention)
        else:
            self.gen_attention_block = None

    def forward(self, x):

        if self.with_cp and x.requires_grad:
            out = self._checkpointed_forward(x)
        else:
            out = self._inner_forward(x)

        out = self.relu(out)

        return out

    @torch.jit.ignore
    def _checkpointed_forward(self, x):
        return cp.checkpoint(self._inner_forward, x)

    def _inner_forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        if not self.with_dcn:
            out = self.conv2(out)
        else:
            if self.conv2_offset is not None:
                if self.with_modulated_dcn:
                    offset_mask = self.conv2_offset(out)
                    offset = offset_mask[:, :18 * self.deformable_groups, :, :]
                    mask = offset_mask[:, -9 * self.deformable_groups:, :, :]
                    mask = mask.sigmoid()
                    out = self.conv2(out, offset, mask)
                else:
                    offset = self.conv2_offset(out)
                    out = self.conv2(out, offset)
            else:
                raise RuntimeError("conv2_offset shouldn't be None")
        out = self.norm2(out)
        out = self.relu(out)

        if self.gen_attention_block is not None:
            out = self.gen_attention_block(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.context_block is not None:
            out = self.context_block(out)

        identity_or_downsample = self.downsample(x)

        out += identity_or_downsample

        return out


def make_res_layer(block,
                   inplanes,
                   planes,
                   blocks,
                   stride=1,
                   dilation=1,
                   style='pytorch',
                   with_cp=False,
                   conv_cfg=None,
                   norm_cfg=dict(type='BN'),
                   dcn=None,
                   gcb=None,
                   gen_attention=None,
                   gen_attention_blocks=[]):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            get_conv_layer_module(conv_cfg)(
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False),
            build_norm_layer(norm_cfg, planes * block.expansion)[1],
        )

    layers = []
    layers.append(
        block(
            inplanes=inplanes,
            planes=planes,
            stride=stride,
            dilation=dilation,
            downsample=downsample,
            style=style,
            with_cp=with_cp,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            dcn=dcn,
            gcb=gcb,
            gen_attention=gen_attention if
            (0 in gen_attention_blocks) else None))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(
            block(
                inplanes=inplanes,
                planes=planes,
                stride=1,
                dilation=dilation,
                style=style,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                gcb=gcb,
                gen_attention=gen_attention if
                (i in gen_attention_blocks) else None))

    return nn.Sequential(*layers)


@BACKBONES.register_module
class ResNet(nn.Module):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Normally 3.
        num_stages (int): Resnet stages, normally 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        norm_cfg (dict): dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): whether to use zero init for last norm layer
            in resblocks to let them behave as identity.

    Example:
        >>> from mmdet.models import ResNet
        >>> import torch
        >>> self = ResNet(depth=18)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)
        >>> jitted = torch.jit.script(ResNet(depth=18))
        >>> jitted.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = jitted.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)
    """

    _version = 2
    __constants__ = ["dcn", "gcb"]

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 gcb=None,
                 stage_with_gcb=(False, False, False, False),
                 gen_attention=None,
                 stage_with_gen_attention=((), (), (), ()),
                 with_cp=False,
                 zero_init_residual=True):
        super(ResNet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))
        self.depth = depth
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.out_indices_map = {i: True for i in out_indices}
        self.style = style
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.gen_attention = gen_attention
        self.gcb = gcb
        self.stage_with_gcb = stage_with_gcb
        if gcb is not None:
            assert len(stage_with_gcb) == num_stages
        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = 64

        self._make_stem_layer(in_channels)
        res_layers = []

        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            gcb = self.gcb if self.stage_with_gcb[i] else None
            planes = 64 * 2**i
            res_layer = make_res_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                gcb=gcb,
                gen_attention=gen_attention,
                gen_attention_blocks=stage_with_gen_attention[i])
            self.inplanes = planes * self.block.expansion
            res_layers.append(res_layer)

        self.res_layers = torch.nn.ModuleList(res_layers)
        self._freeze_stages()

        self.feat_dim = self.block.expansion * 64 * 2**(
            len(self.stage_blocks) - 1)
        self._register_load_state_dict_pre_hook(_convert_res_layers)
        self._register_load_state_dict_pre_hook(_convert_norm_names)

    def _make_stem_layer(self, in_channels):
        self.conv1 = get_conv_layer_module(self.conv_cfg)(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        _, norm1 = build_norm_layer(self.norm_cfg, 64, postfix=1)
        self.norm1 = norm1
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.norm1.eval()
            for m in [self.conv1, self.norm1]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = self.res_layers[i]  #getattr(self, 'layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.dcn is not None:
                for m in self.modules():
                    if isinstance(m, Bottleneck) and hasattr(
                            m, 'conv2_offset'):
                        constant_init(m.conv2_offset, 0)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = []

        i = 0

        for res_layer in self.res_layers:
            x = res_layer(x)
            if i in self.out_indices_map:
                outs.append(x)
            i += 1
        return outs

    def train(self, mode=True):
        super(ResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


def _convert_res_layers(state_dict, prefix: str, local_metadata: Dict,
                        strict: bool, missing_keys: List[str],
                        unexpected_keys: List[str], error_msgs: List[str]):
    """Convert res_layers keys

    Rename key with prefix `backbone.layer1.` to the key with prefix
    `backbone.res_layers.0.`

    """
    MODULE_LIST_NAME = 'res_layers'

    keys_to_convert = []
    converted_prefixes = collections.OrderedDict()
    for key in state_dict.keys():
        if not key.startswith(prefix):
            # it's a parameter/buffer for another module
            continue
        else:
            unprefixed = key.split(prefix)[1]
            if unprefixed.startswith('layer'):
                # we don't convert it
                parts = unprefixed.split('.')
                layer_part = parts[0]
                assert layer_part.startswith('layer')
                index = layer_part.split('layer')[1]
                start_from_zero_index = str(int(index) - 1)
                converted_parts = [MODULE_LIST_NAME, start_from_zero_index]
                converted_prefixes[prefix + layer_part] = \
                    prefix + '.'.join(converted_parts)
                converted_parts.extend(parts[1:])
                converted_key = prefix + '.'.join(converted_parts)
                keys_to_convert.append((key, converted_key))

    for key, converted_key in keys_to_convert:
        state_dict[converted_key] = state_dict.pop(key)
    if converted_prefixes:
        msg = "\n".join(
            ["{} -> {}".format(k, v) for k, v in converted_prefixes.items()])
        logger.info("Converted key prefixes:\n %s", msg)


def _convert_norm_names(state_dict, prefix: str, local_metadata: Dict,
                        strict: bool, missing_keys: List[str],
                        unexpected_keys: List[str], error_msgs: List[str]):
    """Convert norm names to 'norm'"""
    NORM_NAMES = {v[0] for v in norm_utils.norm_cfg.values()}
    NORM_NAMES_PATTERN = '|'.join('({})'.format(n) for n in NORM_NAMES)

    keys_to_convert = []
    converted_prefixes = collections.OrderedDict()
    for key in state_dict.keys():
        if not key.startswith(prefix):
            # it's a parameter/buffer for another module
            continue
        else:
            unprefixed = key.split(prefix)[1]
            m = re.match(NORM_NAMES_PATTERN, unprefixed)
            if m:
                converted_key = prefix + re.sub(
                    NORM_NAMES_PATTERN, 'norm', unprefixed, count=1)
                keys_to_convert.append((key, converted_key))
                converted_prefixes[prefix + m.group()] = prefix + 'norm'

    for key, converted_key in keys_to_convert:
        state_dict[converted_key] = state_dict.pop(key)
    if converted_prefixes:
        msg = "\n".join(
            ["{} -> {}".format(k, v) for k, v in converted_prefixes.items()])
        logger.info("Converted key prefixes:\n %s", msg)
