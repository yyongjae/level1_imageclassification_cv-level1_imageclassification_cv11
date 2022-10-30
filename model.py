from torchvision import nn
import torch.nn.functional as F

import torchvision
import torchvision.models as models


class BaseModel(nn.Module):
    def __init__(self, num_classes, model):
        super().__init__()

        if model == 'efficientnet_b7':
            self.backbone = models.efficientnet_b7(weights='IMAGENET1K_V1')
            
        elif model == 'resnet50':
            self.backbone = models.resnet50(weights='IMAGENET1K_V1')

        elif model == 'densenet121':
            self.backbone = models.densenet121(weights='IMAGENET1K_V1')
            
        elif model == 'densenet169':
            self.backbone = models.densenet169(weights='IMAGENET1K_V1')
        
        elif model == 'swin_t':
            self.backbone = models.swin_t(weights='IMAGENET1K_V1')
        
        elif model == 'vit_l_16':
            self.backbone = models.vit_l_16(weights='IMAGENET1K_V1')
        
        elif model == 'mobilenet_v2':
            self.backbone = models.mobilenet_v2(weights='IMAGENET1K_V1')
        
        elif model == 'squeezenet1_1':
            self.backbone = models.squeezenet1_1(weights='IMAGENET1K_V1')
        
        elif model == 'shufflenet_v2_x2_0':
            self.backbone = models.shufflenet_v2_x2_0(weights='IMAGENET1K_V1')
            
        elif model == 'alexnet':
            self.backbone = models.alexnet(weights='IMAGENET1K_V1')
            
        elif model == 'resnext101_32x8d':
            self.backbone = models.resnext101_32x8d(weights='IMAGENET1K_V1')
            
        elif model == 'efficientnet_v2_m':
            self.backbone = models.efficientnet_v2_m(weights='IMAGENET1K_V1')
            
        elif model == 'vgg19_bn':
            self.backbone = models.vgg19_bn(weights='IMAGENET1K_V1')
            
        elif model == 'regnet_y_32gf':
            self.backbone = models.regnet_y_32gf(weights='IMAGENET1K_V1')
        
        elif model == 'convnext_base':
            self.backbone = models.convnext_base(weights='IMAGENET1K_V1')
        
        elif model == 'efficientnet_b5':
            self.backbone = models.efficientnet_b5(weights='IMAGENET1K_V1')
        
        elif model == 'mnasnet1_0':
            self.backbone = models.mnasnet1_0(weights='IMAGENET1K_V1')
        
        elif model == 'shufflenet_v2_x1_0':
            self.backbone = models.shufflenet_v2_x1_0(weights='IMAGENET1K_V1')
        
        elif model == 'resnext101_64x4d':
            self.backbone = models.resnext101_64x4d(weights='IMAGENET1K_V1')
            
        elif model == 'densenet201':
            self.backbone = models.densenet201(weights='IMAGENET1K_V1')
            
        elif model == 'vit_b_32':
            self.backbone = models.vit_b_32(weights='IMAGENET1K_V1')
            
        elif model == 'convnext_large':
            self.backbone = models.convnext_large(weights='IMAGENET1K_V1')

        elif model == 'resnet18':
            self.backbone = models.resnet18(weights='IMAGENET1K_V1')
            
        elif model == 'densenet161':
            self.backbone = models.densenet161(weights='IMAGENET1K_V1')
            
        elif model == 'mobilenet_v3_small':
            self.backbone = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
            
        elif model == 'mobilenet_v3_large':
            self.backbone = models.mobilenet_v3_large(weights='IMAGENET1K_V1')
            
        elif model == 'resnext50_32x4d':
            self.backbone = models.resnext50_32x4d(weights='IMAGENET1K_V1')
            
        elif model == 'resnext101_64x4d':
            self.backbone = models.resnext101_64x4d(weights='IMAGENET1K_V1')
        
        elif model == 'convnext_tiny':
            self.backbone = models.convnext_tiny(weights='IMAGENET1K_V1')
            
        elif model == 'swin_s':
            self.backbone = models.swin_s(weights='IMAGENET1K_V1')
            
        elif model == 'swin_b':
            self.backbone = models.swin_b(weights='IMAGENET1K_V1')
            
            
        self.classifier = nn.Linear(1000, num_classes)
            
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x