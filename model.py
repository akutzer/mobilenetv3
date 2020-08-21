import math
from typing import TypeVar, Union, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F



def _make_divisible(v: Union[int, float], divisor: int = 8) -> int:
    return int(math.ceil(v * 1. / divisor) * divisor)


class ConvBN(nn.Module):

    def __init__(
            self,
            in_c: int,
            out_c: int,
            k_size: int,
            stride: int = 1,
            groups: int = 1,
            activation : Optional[nn.Module] = None,
            squeeze_excite: bool = False
    ):
        super().__init__()
        self.padding = (k_size - 1) // 2        # same padding
        self.squeeze_excite = squeeze_excite

        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_c, out_c, k_size, stride, self.padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_c)
        )
        if activation:
            self.conv_bn.add_module("2", activation)

        if self.squeeze_excite:
            self.squeeze = SqueezeExcite(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_bn(x)
        if self.squeeze_excite:
            x = x * self.squeeze(x)
        return x


class SqueezeExcite(nn.Module):

    def __init__(self, channels: int, factor: Union[int, float] = 1/4):
        super().__init__()
        self.squeeze_excite = nn.Sequential(
            nn.Linear(channels, math.floor(channels * factor)),
            nn.ReLU(),
            nn.Linear(math.floor(channels * factor), channels),
            nn.Hardsigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, *_ = x.size()
        x = F.adaptive_max_pool2d(x, 1).view(b, c)
        return self.squeeze_excite(x).view(b, c, 1, 1)


class InvertedResidual(nn.Module):

    def __init__(
        self,
        in_c: int,
        exp_c: int,
        out_c: int,
        k_size: int,
        stride: int,
        activation: nn.Module,
        squeeze_excite: bool = False
    ):
        super().__init__()
        self.residual = (stride == 1 or stride == (1, 1)) and (in_c == out_c)
        self.squeeze_excite = squeeze_excite

        self.block = nn.Sequential(
            ConvBN(in_c, exp_c, 1, activation=activation),
            ConvBN(exp_c, exp_c, k_size, stride,
                   groups=exp_c,
                   activation=activation,
                   squeeze_excite=squeeze_excite),
            ConvBN(exp_c, out_c, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ = self.block(x)
        if self.residual:
            x_ = x_ + x
        return x_


class MobileNetV3(nn.Module):

    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: Union[int, float] = 1.0,
        architecture: str = "small"
    ):
        super().__init__()

        if architecture not in ["small", "large"]:
            raise ValueError("Only architecture 'small' and 'large' supported!")

        hs = nn.Hardswish()
        relu = nn.ReLU()
        last_channels = _make_divisible(16 * width_mult)

        if architecture == "small":
            features_arch = (
                ( 16,  16, 3, 2, relu,  True),
                ( 72,  24, 3, 2, relu, False),
                ( 88,  24, 3, 1, relu, False),
                ( 96,  40, 5, 2,   hs,  True),
                (240,  40, 5, 1,   hs,  True),
                (240,  40, 5, 1,   hs,  True),
                (120,  48, 5, 1,   hs,  True),
                (144,  48, 5, 1,   hs,  True),
                (288,  96, 5, 2,   hs,  True),
                (576,  96, 5, 1,   hs,  True),
                (576,  96, 5, 1,   hs,  True),
            )
        elif architecture == "large":
            features_arch = (
                (16,  16, 3, 1, relu, False),
                (64,  24, 3, 2, relu, False),
                (72,  24, 3, 1, relu, False),
                (72,  40, 5, 2, relu,  True),
                (120,  40, 5, 1, relu,  True),
                (120,  40, 5, 1, relu,  True),
                (240,  80, 3, 2,   hs, False),
                (200,  80, 3, 1,   hs, False),
                (184,  80, 3, 1,   hs, False),
                (184,  80, 3, 1,   hs, False),
                (480, 112, 3, 1,   hs,  True),
                (672, 112, 3, 1,   hs,  True),
                (672, 160, 5, 2,   hs,  True),
                (960, 160, 5, 1,   hs,  True),
                (960, 160, 5, 1,   hs,  True),
            )

        features = [ConvBN(3, last_channels, 3, 2, activation=hs)]
        for exp_c, out_c, k, s, activation, se in features_arch:
            in_c = last_channels
            exp_c = _make_divisible(exp_c * width_mult)
            out_c = _make_divisible(out_c * width_mult)
            features.append(InvertedResidual(in_c, exp_c, out_c, k, s,
                                             activation, se))
            last_channels = out_c

        if architecture == "small":
            in_c = last_channels
            exp_c = _make_divisible(576 * width_mult)
            out_c = _make_divisible(1024 * width_mult)
            features.extend([
                ConvBN(in_c, exp_c, 1, 1, activation=hs, squeeze_excite=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(exp_c, out_c, 1, 1),
                nn.Hardswish(),
            ])
            classifier = nn.Sequential(
                nn.Dropout(p=.8),
                nn.Conv2d(out_c, num_classes, 1, 1)
            )
        elif architecture == "large":
            in_c = last_channels
            exp_c = _make_divisible(960 * width_mult)
            out_c = _make_divisible(1280 * width_mult)
            features.extend([
                ConvBN(in_c, exp_c, 1, 1, activation=hs, squeeze_excite=False),
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(exp_c, out_c, 1, 1),
                nn.Hardswish(),
            ])
            classifier = nn.Sequential(
                nn.Dropout(p=.8),
                nn.Conv2d(out_c, num_classes, 1, 1)
            )

        self.features = nn.Sequential(*features)
        self.classifier = classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.classifier(x)

        if x.dim() > 2:
            x = x.view(x.shape[0], -1)
        return x


def mobilenet_v2(pretrained=False, progress=True, **kwargs):
    model = MobileNetV3(**kwargs)
    if pretrained:
        raise NotImplementedError("No pretrained model available")
    return model


if __name__ == "__main__":
    model = mobilenet_v2(architecture="large", num_classes=1000, width_mult=1)
    print(model)
    print("Param:", sum(p.numel() for p in model.parameters()))
