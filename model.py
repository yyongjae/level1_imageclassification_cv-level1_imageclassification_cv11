import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.models as models


class BaseModel(nn.Module):
    def __init__(self, num_classes, model):
        super().__init__()

        if model == 'efficientnet_b7':
            self.backbone = models.efficientnet_b7(pretrained=True)
            
        elif model == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)

        elif model == 'densenet121':
            self.backbone = models.densenet121(pretrained=True)
            
        elif model == 'densenet169':
            self.backbone = models.densenet169(pretrained=True)
        
        elif model == 'swin_t':
            self.backbone = models.swin_t(weights='IMAGENET1K_V1')
        
        elif model == 'vit_l_16':
            self.backbone = models.vit_l_16(pretrained=True)
        
        elif model == 'mobilenet_v2':
            self.backbone = models.mobilenet_v2(pretrained=True)
        
        elif model == 'squeezenet1_1':
            self.backbone = models.squeezenet1_1(pretrained=True)
        
        elif model == 'shufflenet_v2_x2_0':
            self.backbone = models.shufflenet_v2_x2_0(pretrained=True)
            
        elif model == 'alexnet':
            self.backbone = models.alexnet(pretrained=True)
            
        elif model == 'resnext101_32x8d':
            self.backbone = models.resnext101_32x8d(pretrained=True)
            
        elif model == 'efficientnet_v2_m':
            self.backbone = models.efficientnet_v2_m(pretrained=True)
            
        elif model == 'vgg19_bn':
            self.backbone = models.vgg19_bn(pretrained=True)
            
        elif model == 'regnet_y_32gf':
            self.backbone = models.regnet_y_32gf(pretrained=True)
        
        elif model == 'convnext_base':
            self.backbone = models.convnext_base(pretrained=True)
        
        elif model == 'efficientnet_b5':
            self.backbone = models.efficientnet_b5(pretrained=True)
        
        elif model == 'mnasnet1_0':
            self.backbone = models.mnasnet1_0(pretrained=True)
        
        elif model == 'shufflenet_v2_x1_0':
            self.backbone = models.shufflenet_v2_x1_0(pretrained=True)
        
        elif model == 'resnext101_64x4d':
            self.backbone = models.resnext101_64x4d(weights='IMAGENET1K_V1')
            
        elif model == 'densenet201':
            self.backbone = models.densenet201(pretrained=True)
            
        elif model == 'vit_b_32':
            self.backbone = models.vit_b_32(pretrained=True)
            
        elif model == 'convnext_large':
            self.backbone = models.convnext_large(pretrained=True)
            
        ##
        elif model == 'resnet18':
            self.backbone = models.resnet18(weights='DEFAULT')
            
        elif model == 'densenet161':
            self.backbone = models.densenet161(weights='DEFAULT')
            
        elif model == 'mobilenet_v3_small':
            self.backbone = models.mobilenet_v3_small(weights='DEFAULT')
            
        elif model == 'mobilenet_v3_large':
            self.backbone = models.mobilenet_v3_large(weights='DEFAULT')
            
        elif model == 'resnext50_32x4d':
            self.backbone = models.resnext50_32x4d(weights='DEFAULT')
            
        elif model == 'resnext101_64x4d':
            self.backbone = models.resnext101_64x4d(weights='DEFAULT')
        
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