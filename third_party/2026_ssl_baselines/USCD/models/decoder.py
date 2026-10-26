import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPP(nn.Module):
    def __init__(self, in_ch=2048, out_ch=256):
        super().__init__()
        def conv_bn_relu(dilation):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3 if dilation > 1 else 1,
                          padding=dilation, dilation=dilation, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

        self.c1   = conv_bn_relu(1)
        self.c6   = conv_bn_relu(6)
        self.c12  = conv_bn_relu(12)
        self.c18  = conv_bn_relu(18)
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.proj = nn.Sequential(
            nn.Conv2d(out_ch * 5, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout(0.5))

    def forward(self, x):
        h, w = x.shape[2:]
        out = [self.c1(x), self.c6(x), self.c12(x), self.c18(x),
               F.interpolate(self.pool(x), (h, w), mode='bilinear', align_corners=False)]
        return self.proj(torch.cat(out, dim=1))


class DeepLabDecoder(nn.Module):
    def __init__(self, in_channels=2048, num_classes=2, feat_dim=256):
        super().__init__()
        self.aspp = ASPP(in_channels, feat_dim)
        self.head = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim), nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(feat_dim, num_classes, 1),
        )

    def forward(self, enc_feat, output_size):
        feat = self.aspp(enc_feat)
        logits = F.interpolate(self.head(feat), size=output_size,
                               mode='bilinear', align_corners=False)
        return logits, F.softmax(logits, dim=1), feat
