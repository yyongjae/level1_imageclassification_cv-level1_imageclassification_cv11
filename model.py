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

        elif model == 'darknet53':
            self.backbone = darknet53(num_classes)
            
        self.classifier = nn.Linear(1000, num_classes)


    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x



####DarkNet53#############
def conv_batch(in_num, out_num, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_num, out_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_num),
        nn.LeakyReLU())

# Residual block
class DarkResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(DarkResidualBlock, self).__init__()

        reduced_channels = int(in_channels/2)

        self.layer1 = conv_batch(in_channels, reduced_channels, kernel_size=1, padding=0)
        self.layer2 = conv_batch(reduced_channels, in_channels)

    def forward(self, x):
        residual = x

        out = self.layer1(x)
        out = self.layer2(out)
        out += residual
        return out

class Darknet53(nn.Module):
    def __init__(self, block, num_classes):
        super(Darknet53, self).__init__()

        self.num_classes = num_classes

        self.features = nn.Sequential(
            conv_batch(3, 32),
            conv_batch(32, 64, stride=2),
            self.make_layer(block, in_channels=64, num_blocks=1),
            conv_batch(64, 128, stride=2),
            self.make_layer(block, in_channels=128, num_blocks=2),
            conv_batch(128, 256, stride=2),
            self.make_layer(block, in_channels=256, num_blocks=8),
            conv_batch(256, 512, stride=2),
            self.make_layer(block, in_channels=512, num_blocks=8),
            conv_batch(512, 1024, stride=2),
            self.make_layer(block, in_channels=1024, num_blocks=4),
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(1024, 1000)

    def forward(self, x):
        out = self.features(x)
        out = self.global_avg_pool(out)
        out = out.view(-1, 1024)
        out = self.classifier(out)

        return out

    def make_layer(self, block, in_channels, num_blocks):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels))
        return nn.Sequential(*layers)


def darknet53(num_classes):
    return Darknet53(DarkResidualBlock, num_classes)
#########################