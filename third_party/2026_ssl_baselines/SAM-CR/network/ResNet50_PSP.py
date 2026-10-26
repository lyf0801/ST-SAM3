import warnings

import torch
from torch import nn
from network import resnet
from network.changerEx import ChangerEx

import torch.nn.functional as F
def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)
class Convmodule(nn.Module):
    def __init__(self,in_channels,
        out_channels,
        kernel_size = 1,
        stride = 1,
        norm_cfg = None,
        act_cfg = None):
        super(Convmodule, self).__init__()
        self.act_cfg=act_cfg
        self.norm_cfg = norm_cfg
        self.conv=nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=True)
        self.act=nn.ReLU()
        self.norm=nn.BatchNorm2d(out_channels)
        for param in self.norm.parameters():
            param.requires_grad = True
    def forward(self, x):
        x = self.conv(x)
        if(self.norm_cfg!=None):
            x = self.norm(x)
        if(self.act_cfg!=None):
            x = self.act(x)
        return x
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class _PSPModule(nn.Module):
    def __init__(self, in_channels, bin_sizes):
        super(_PSPModule, self).__init__()

        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s) for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + (out_channels * len(bin_sizes)), out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear',
                                       align_corners=False) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output
class ResNet__PSP(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet__PSP, self).__init__()

        self.base = resnet.resnet50(pretrained=pretrained,progress=True,interaction_cfg=(0, 1, 2, 2))
        self.in_index = [0, 1, 2, 3]
        self.in_channels=[256,512,1024,2048]
        self.channels=512
        num_inputs = len(self.in_index)
        self.conv_reduce = BasicConv2d(512*2,512,3,1,1)

        self.conv_fu = BasicConv2d(512, 512, 3, 1, 1)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                Convmodule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg='bn',
                    act_cfg='relu'))

        self.fusion_conv = Convmodule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg='bn')
        self.psp = _PSPModule(512, bin_sizes=[2, 3, 4, 6])
        mid_channels = 64
        upscale=4
        self.diff_conv1x1 = nn.Conv2d(128, mid_channels, kernel_size=3, padding=1, bias=False)
        nn.init.kaiming_normal_(self.diff_conv1x1.weight.data)
        self.up = nn.Upsample(scale_factor=upscale, mode='bilinear')
        self.conv1x1 = nn.Conv2d(mid_channels, 1, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.conv1x1.weight.data)

    def freeze_backbone_param(self):
        for param in self.base.parameters():
            param.requires_grad = False

    def base_forward(self, inputs):
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=True))

        out = self.fusion_conv(torch.cat(outs, dim=1))
        return out


    def forward(self, A, B):
        inputs1=[]
        inputs2 = []
        inputs = self.base(A, B)
        inputs = [inputs[i] for i in self.in_index]

        for i,input in enumerate(inputs):
            f1, f2 = torch.chunk(input, 2, dim=1)
            inputs1.append(f1)
            inputs2.append(f2)

        out1 = self.base_forward(inputs1)
        out2 = self.base_forward(inputs2)

        x = self.conv_reduce(torch.cat([out1, out2], dim=1))
        x = self.conv_fu(x)
        x = self.psp(x)
        x = self.diff_conv1x1(x)
        x = self.up(x)
        x = self.conv1x1(x)
        return x,torch.abs(out1-out2)

    def get_backbone_params(self):
        return self.base.parameters()

    def get_module_params(self):
        return self.changer_cd.parameters()