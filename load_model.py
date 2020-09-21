import torch

model = torch.hub.load('akutzer/mobilenetv3', 'mobilenet_v3', pretrained=False,
                       architecture="large", num_classes=420)
print(model)
print("Param:", sum(p.numel() for p in model.parameters()))


model = torch.hub.load('akutzer/mobilenetv3', 'mobilenet_v3_segmentation',
                       pretrained=False, architecture="small", head="aspp",
                       shallow_stride=8, deep_stride=16,
                       out_c=20, head_c=128, width_mult=1.0)
print(model)
print("Param:", sum(p.numel() for p in model.parameters()))
