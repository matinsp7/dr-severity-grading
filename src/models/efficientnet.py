import timm
import torch.nn as nn


class EfficientNetDR(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        self.model = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=num_classes,
        )

    def forward(self, x):
        return self.model(x)