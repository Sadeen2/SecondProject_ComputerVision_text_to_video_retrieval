# models/backbone.py
import torch
import torch.nn as nn
import torchvision.models as tvm


class CNNBackbone(nn.Module):
    """
    Loads pretrained CNN (ResNet) and outputs frame-level feature vectors.
    For ResNet50:
      input:  [B, 3, H, W]
      output: [B, feat_dim]  (feat_dim = 2048)
    """

    def __init__(self, backbone_name="resnet50", pretrained=True, freeze=False):
        super().__init__()

        if backbone_name == "resnet18":
            m = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None)
            feat_dim = 512
        elif backbone_name == "resnet34":
            m = tvm.resnet34(weights=tvm.ResNet34_Weights.DEFAULT if pretrained else None)
            feat_dim = 512
        elif backbone_name == "resnet50":
            m = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT if pretrained else None)
            feat_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # remove classification layer, keep pooling + features
        m.fc = nn.Identity()
        self.model = m
        self.feat_dim = feat_dim

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
