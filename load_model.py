import torch

model = torch.hub.load('akutzer/mobilenetv3', 'mobilenet_v3', pretrained=False,
                       architecture="large", num_classes=420)
print(model)
