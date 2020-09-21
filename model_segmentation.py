import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import MobileNetV3, ConvBN, InvertedResidual



class LR_ASPP(nn.Module):

    def __init__(self, shallow_in_c: int, deep_in_c: int, out_c: int,
                 hidden_c: int = 128):
        super().__init__()

        self.deep_conv_1 = ConvBN(deep_in_c, hidden_c, 1, activation=nn.ReLU6())
        self.squeeze_excite = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Conv2d(deep_in_c, hidden_c, 1),
            nn.Hardsigmoid(),
        )
        self.deep_conv_2 = nn.Conv2d(hidden_c, out_c, 1)
        self.shallow_conv = nn.Conv2d(shallow_in_c, out_c, 1)

    def forward(self, shallow_x, deep_x):
        # prepare skip connection from low-level features
        shallow_x = self.shallow_conv(shallow_x)

        # prepare high-level features (1x1-Conv, Pooling with large kernel)
        deep_x_ = self.deep_conv_1(deep_x)
        se_x = self.squeeze_excite(deep_x)
        se_x = F.interpolate(se_x, size=deep_x_.shape[-2:],
                             mode="bilinear", align_corners=True)
        excited_deep_x = deep_x_ * se_x
        excited_deep_x = F.interpolate(se_x, size=shallow_x.shape[-2:],
                                       mode="bilinear", align_corners=True)
        excited_deep_x = self.deep_conv_2(excited_deep_x)

        # add skip connection from low-level features
        segmentation_x = excited_deep_x + shallow_x

        return segmentation_x


class R_ASPP(nn.Module):

    def __init__(self, shallow_in_c: int, deep_in_c: int, out_c: int,
                 hidden_c: int = 32, dilations: list = [1, 3, 5]):
        super().__init__()
        self.shallow_conv = nn.Conv2d(shallow_in_c, out_c, 1)

        self.pyramid = nn.ModuleList()
        self.pyramid.append(nn.Conv2d(deep_in_c, hidden_c, 1))
        kernel_size = 3
        for dilation in dilations:
            padding = int((kernel_size - 1) * dilation / 2)
            self.pyramid.append(nn.Conv2d(deep_in_c, hidden_c, kernel_size,
                                          dilation=dilation, padding=padding))

        self.deep_conv = nn.Conv2d(hidden_c * len(self.pyramid) + deep_in_c,
                                   out_c, 1)

    def forward(self, shallow_x, deep_x):
        # prepare skip connection from low-level features
        shallow_x = self.shallow_conv(shallow_x)

        # prepare high-level features
        # (skip-connection, 1x1-conv, atrous spatial pyramid pooling)
        deep_x_cat = [deep_x] + [module(deep_x)for module in self.pyramid]
        deep_x_cat = torch.cat(deep_x_cat, dim=1)
        deep_x = self.deep_conv(deep_x_cat)
        deep_x = F.interpolate(deep_x, size=shallow_x.shape[-2:],
                                       mode="bilinear", align_corners=True)

        # add skip connection from low-level features
        segmentation_x = deep_x + shallow_x
        segmentation_x = F.interpolate(segmentation_x, scale_factor=2,
                                       mode="bilinear", align_corners=True)

        return segmentation_x


class MobileNetV3Segmentation(nn.Module):

    shallow_x: torch.Tensor

    def __init__(
            self,
            out_c: int,
            architecture: str = "small",
            head: str = "lr_aspp",
            width_mult: float = 1.0,
            shallow_stride: int = 8,
            deep_stride: int = 16,
            head_c: int = 128
    ):
        super().__init__()
        self.out_c = out_c
        self.architecture = architecture
        self.width_mult = width_mult
        self.shallow_stride = shallow_stride
        self.deep_stride = deep_stride
        self.head_c = head_c

        assert shallow_stride < deep_stride, \
            "Shallow stride should be smaller than deep stride"

        head = head.lower()
        assert head in ["lr_aspp", "r_aspp"], \
            "Head must be LR_ASPP or R_ASPP!"

        self.backbone, self.segm_head = self._prepare_model(
            architecture, head, shallow_stride, deep_stride, width_mult)

    def forward(self, x: torch.Tensor):
        deep_x = self.backbone(x)
        out = self.segm_head(self.shallow_x, deep_x)
        out_interp = F.interpolate(out, x.shape[-2:], mode='bicubic',
                                   align_corners=True)
        return out_interp

    def _prepare_model(
            self,
            architecture: str,
            head: str,
            shallow_stride: int,
            deep_stride: int,
            width_mult: float = 1.0
    ):
        mobile_net = MobileNetV3(architecture=architecture,
                                 width_mult=width_mult)

        shallow_hook_bool = False
        shallow_channels, deep_channels = None, None
        backbone = [mobile_net.features[0]]
        output_stride = max(backbone[0].conv_bn[0].stride)

        # append every inverted residual block to the backbone, until we reach
        # the last module with our desired output stride (deep_stride);
        # also apply a forward-hook to the last module with the desired
        # shallow stride, for a low-level feature skip connection
        for module in mobile_net.features.modules():
            if isinstance(module, InvertedResidual):
                output_stride *= module.stride

                if output_stride == shallow_stride * 2 \
                        and not shallow_hook_bool:
                    backbone[-1].register_forward_hook(self._set_shallow_hook())
                    shallow_channels = backbone[-1].out_c
                    shallow_hook_bool = True
                if output_stride == deep_stride * 2 and shallow_hook_bool:
                    deep_channels = backbone[-1].out_c
                    break
                backbone.append(module)

        # if the very last module had the desired deep stride
        # extract its output channels
        if output_stride == deep_stride and shallow_hook_bool:
            deep_channels = backbone[-1].out_c

        assert shallow_channels, \
            "Shallow stride is to big, could not place hook!"
        assert deep_channels, \
            f"Deep stride is to big! Max stride possible {output_stride}"

        if head == "lr_aspp":
            # reduce channels in last block by factor of 2 and set dilation=2
            backbone[-1] = self._reduce_last_block_by_factor(backbone[-1], 2, 2)
            head = LR_ASPP(shallow_channels, deep_channels // 2,
                           self.out_c, self.head_c)
        if head == "r_aspp":
            head = R_ASPP(shallow_channels, deep_channels,
                           self.out_c, self.head_c)

        return nn.Sequential(*backbone), head

    def _set_shallow_hook(self):
        def shallow_hook(module, t_in, t_out):
            self.shallow_x = t_out
            return t_out
        return shallow_hook

    @staticmethod
    def _reduce_last_block_by_factor(inv_residual: nn.Module,
                                     factor: int, dilation: int):
        assert isinstance(inv_residual, InvertedResidual), \
            "Block is not of type InvertedResidual"

        args = inspect.getfullargspec(inv_residual.__init__).args
        module_args = {}
        for arg in args:
            if hasattr(inv_residual, arg):
                module_args[arg] = getattr(inv_residual, arg)

        module_args["dilation"] = dilation
        module_args["exp_c"] = int(module_args["exp_c"] // factor)
        module_args["out_c"] = int(module_args["out_c"] // factor)

        new_inv_residual = inv_residual.__class__(**module_args)

        return new_inv_residual


def mobilenet_v3_segmentation(pretrained=False, progress=True, **kwargs):
    model = MobileNetV3Segmentation(**kwargs)
    if pretrained:
        raise NotImplementedError("No pretrained model available")
    return model


if __name__ == "__main__":
    model = mobilenet_v3_segmentation(out_c=20, architecture="small",
                                                head="lr_aspp",
                                                shallow_stride=8, deep_stride=16,
                                                head_c=128, width_mult=1.0)
    print(model)
    print("Param:", sum(p.numel() for p in model.parameters()))
