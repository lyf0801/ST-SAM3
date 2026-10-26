import copy
import torch
import torch.nn as nn
from .backbone import SiameseEncoder
from .decoder import DeepLabDecoder


class ChangeDetector(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True,
                 in_channels=3, num_classes=2):
        super().__init__()
        self.encoder = SiameseEncoder(backbone, pretrained, in_channels)
        self.decoder = DeepLabDecoder(self.encoder.out_channels, num_classes)

    def forward(self, img_A, img_B):
        h, w = img_A.shape[2:]
        return self.decoder(self.encoder(img_A, img_B), (h, w))


class USCDModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        mc = cfg['model']
        self.ema_momentum = cfg['train'].get('ema_momentum', 0.999)

        self.student = ChangeDetector(
            mc.get('backbone', 'resnet50'),
            mc.get('pretrained', True),
            cfg['data'].get('in_channels', 3),
            mc.get('num_classes', 2))

        self.teacher = copy.deepcopy(self.student)
        for p in self.teacher.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_teacher(self):
        a = self.ema_momentum
        for t, s in zip(self.teacher.parameters(), self.student.parameters()):
            t.data.mul_(a).add_(s.data * (1 - a))

    def forward_student(self, img_A, img_B):
        return self.student(img_A, img_B)

    @torch.no_grad()
    def forward_teacher(self, img_A, img_B):
        return self.teacher(img_A, img_B)

    def compute_uncertainty(self, prob):
        return 1.0 - torch.abs(prob[:, 0] - prob[:, 1])

    def generate_pseudo_labels(self, prob, threshold=0.9):
        max_prob, pseudo = prob.max(dim=1)
        return pseudo, max_prob >= threshold
