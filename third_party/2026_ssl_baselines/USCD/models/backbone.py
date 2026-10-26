import torch
import torch.nn as nn
import torchvision.models as models


class ResNetEncoder(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True, in_channels=3):
        super().__init__()
        if backbone == 'resnet50':
            net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1
                                  if pretrained else None)
        elif backbone == 'resnet101':
            net = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1
                                   if pretrained else None)
        else:
            raise ValueError(backbone)

        if in_channels != 3:
            net.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)

        self.layer0 = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.out_channels = 2048
        self._set_dilations()

    def _set_dilations(self):
        for m in self.layer3.modules():
            if hasattr(m, 'stride') and m.stride == (2, 2):
                m.stride = (1, 1)
            if hasattr(m, 'dilation') and m.kernel_size == (3, 3):
                m.dilation, m.padding = (2, 2), (2, 2)
        for m in self.layer4.modules():
            if hasattr(m, 'stride') and m.stride == (2, 2):
                m.stride = (1, 1)
            if hasattr(m, 'dilation') and m.kernel_size == (3, 3):
                m.dilation, m.padding = (4, 4), (4, 4)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class SiameseEncoder(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True, in_channels=3):
        super().__init__()
        self.encoder = ResNetEncoder(backbone, pretrained, in_channels)
        self.fusion = nn.Sequential(
            nn.Conv2d(self.encoder.out_channels * 2, self.encoder.out_channels, 1),
            nn.BatchNorm2d(self.encoder.out_channels),
            nn.ReLU(inplace=True),
        )
        self.out_channels = self.encoder.out_channels

    def forward(self, img_A, img_B):
        feat_A = self.encoder(img_A)
        feat_B = self.encoder(img_B)
        return self.fusion(torch.cat([feat_A, feat_B], dim=1))
