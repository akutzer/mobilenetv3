# Optional list of dependencies required by the package
dependencies = ['torch']

from model import MobileNetV3
from model_segmentation import MobileNetV3Segmentation


def mobilenet_v3(pretrained=False, **kwargs):
    model = MobileNetV3(**kwargs)
    if pretrained:
        raise NotImplementedError("No pretrained model available")
    return model


def mobilenet_v3_segmentation(pretrained=False, **kwargs):
    model = MobileNetV3Segmentation(**kwargs)
    if pretrained:
        raise NotImplementedError("No pretrained model available")
    return model
