import torch
from torch import nn
import torch.nn.functional as F

import torchvision
import torchvision.models as models

class BaseModel(nn.Module):
    def __init__(self, num_classes, model):
        super().__init__()

        if model == 'efficientnet_b7':
            self.backbone = models.efficientnet_b7(weights='IMAGENET1K_V1')
        elif model == 'densenet121':
            self.backbone = models.densenet121(weights='IMAGENET1K_V1')
        elif model == 'densenet201':
            self.backbone = models.densenet201(weights='IMAGENET1K_V1')
        elif model == 'resnet18':
            self.backbone = models.resnet18(weights='IMAGENET1K_V1')
            self.backbone.conv1.requires_grad_(False)
            self.backbone.layer1.requires_grad_(False)

        self.classifier = nn.Linear(1000, num_classes)


    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x