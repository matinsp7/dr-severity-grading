import torch

from src.models.efficientnet import EfficientNetDR

model = EfficientNetDR()

images = torch.randn(16, 3, 384, 384)

outputs = model(images)

print(outputs.shape)