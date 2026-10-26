import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import torch.distributions as dist
from torch.distributions.uniform import Uniform

import random
import math
import cv2
import numpy as np
from copy import deepcopy

from utils import BCELoss, IoULoss, Evaluator, upsample_scale, upsample_size
from models import CreateModel


class CCT(nn.Module):
    def __init__(self, args, device):
        super(CCT, self).__init__()
        self.device = device
        self.name = 'CCT'
        self.num_classes = args.num_classes
        self.ignore_index = args.ignore_index
        self.metric = Evaluator(self.num_classes+1)

        self.init_lr = args.learning_rate
        self.model = nn.DataParallel(CreateModel(args.model, self.num_classes).to(self.device))
        feat_dim = self.model.module.embedding_dim * 4
        upscale = 4
        vat_decoder = [VATDecoder(upscale, feat_dim, self.num_classes, xi=1e-6,
                                    eps=2.0) for _ in range(2)]
        drop_decoder = [DropOutDecoder(upscale, feat_dim, self.num_classes,
                                    drop_rate=0.5, spatial_dropout=True)
                                    for _ in range(6)]
        cut_decoder = [CutOutDecoder(upscale, feat_dim, self.num_classes, erase=0.4)
                                    for _ in range(6)]
        context_m_decoder = [ContextMaskingDecoder(upscale, feat_dim, self.num_classes)
                                    for _ in range(2)]
        object_masking = [ObjectMaskingDecoder(upscale, feat_dim, self.num_classes)
                                    for _ in range(2)]
        feature_drop = [FeatureDropDecoder(upscale, feat_dim, self.num_classes)
                                    for _ in range(6)]
        feature_noise = [FeatureNoiseDecoder(upscale, feat_dim, self.num_classes,
                                    uniform_range=0.3)
                                    for _ in range(6)]
        self.aux_decoders = nn.ModuleList([*vat_decoder, *drop_decoder, *cut_decoder, \
                                           *context_m_decoder, *object_masking, \
                                           *feature_drop, *feature_noise]).to(self.device)

        model_params = self.model.module.optim_parameters(args.learning_rate)
        aux_decoder_params = {'params': self.aux_decoders.parameters(), 'lr': args.learning_rate * 10}
        all_params = model_params + [aux_decoder_params]

        self.optimizer = optim.AdamW(all_params, lr=self.init_lr)
        self.bce_criterion = BCELoss(self.ignore_index)
        self.conf_thr = 0.95

    def adjust_learning_rate(self, init_lr, optimizer, iters, total_iters):
        decay = (1 - iters / total_iters) ** 0.9
        for i, param_group in enumerate(optimizer.param_groups):
            if i == 0:
                param_group['lr'] = init_lr * decay  
            else:
                param_group['lr'] = init_lr * 10 * decay  

    def update(self, image_l, target_l, image_u, image_us, target_u, curr_iter, total_iters):
        self.optimizer.zero_grad()                                  
        self.model.train()
        self.aux_decoders.train()
        self.adjust_learning_rate(self.init_lr, self.optimizer, curr_iter, total_iters)

        # labeled learning
        prob_l = self.model(image_l)           
        loss_l = self.bce_criterion(prob_l, target_l) 

        # unlabeled learning
        feat_u = self.get_feats(image_u)
        logit_u = self.get_logits(feat_u)
        prob_u = torch.sigmoid(logit_u)         
        pseudo_labels = (prob_u.detach() > 0.5).float()

        # Get auxiliary predictions
        loss_u = 0
        feat_u = upsample_scale(feat_u, scale=0.25)
        prob_u = upsample_scale(prob_u, scale=0.25)
        pseudo_labels = upsample_scale(pseudo_labels, scale=0.25)
        for aux_decoder in self.aux_decoders:
            prob_u_aux = torch.sigmoid(aux_decoder(feat_u, prob_u.detach()))
            loss_u += F.mse_loss(prob_u_aux, pseudo_labels.detach())
        loss_u /= len(self.aux_decoders) 

        # overall loss
        loss = loss_l + loss_u
        loss.backward()
        self.optimizer.step()
        return {'loss_l': loss_l.item(), 'loss_u': loss_u.item()}

    def forward(self, image):
        prob = self.model(image)           
        return prob
        
    def get_feats(self, image):
        feat = self.model.module.get_feats(image)           
        return feat

    def get_probs(self, feat):
        prob = self.model.module.get_probs(feat)           
        return prob
        
    def get_logits(self, feat):
        logit = self.model.module.get_logits(feat)           
        return logit




class Decoder(nn.Module):
    def __init__(self, in_channels, num_classes, upscale=4, dropout_ratio=0.1):
        super(Decoder, self).__init__()
        self.linear_pred = self._make_last_layer(in_channels, num_classes)
        self.dropout = nn.Dropout2d(dropout_ratio)
        self.upscale = upscale

    def _make_last_layer(self, in_channels, out_channels):
        last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.BatchNorm2d(in_channels, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0)
        )
        return last_layer 

    def forward(self, x):
        x = self.linear_pred(x)
        x = upsample_scale(x, self.upscale)
        return x


class DropOutDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes, drop_rate=0.3, spatial_dropout=True):
        super(DropOutDecoder, self).__init__()
        self.dropout = nn.Dropout2d(p=drop_rate) if spatial_dropout else nn.Dropout(drop_rate)
        self.upsample = Decoder(conv_in_ch, num_classes, upscale=upscale)

    def forward(self, x, _):
        x = self.upsample(self.dropout(x))
        return x


class FeatureDropDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes):
        super(FeatureDropDecoder, self).__init__()
        self.upsample = Decoder(conv_in_ch, num_classes, upscale=upscale)

    def feature_dropout(self, x):
        attention = torch.mean(x, dim=1, keepdim=True)
        max_val, _ = torch.max(attention.view(x.size(0), -1), dim=1, keepdim=True)
        threshold = max_val * np.random.uniform(0.7, 0.9)
        threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
        drop_mask = (attention < threshold).float()
        return x.mul(drop_mask)

    def forward(self, x, _):
        x = self.feature_dropout(x)
        x = self.upsample(x)
        return x


class FeatureNoiseDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes, uniform_range=0.3):
        super(FeatureNoiseDecoder, self).__init__()
        self.upsample = Decoder(conv_in_ch, num_classes, upscale=upscale)
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x, _):
        x = self.feature_based_noise(x)
        x = self.upsample(x)
        return x



def _l2_normalize(d):
    # Normalizing per batch axis
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


def get_r_adv(x, decoder, it=1, xi=1e-1, eps=10.0):
    """
    Virtual Adversarial Training
    https://arxiv.org/abs/1704.03976
    """
    with torch.no_grad():
        pred = torch.sigmoid(decoder(x))

    d = torch.rand(x.shape).sub(0.5).to(x.device)
    d = _l2_normalize(d)

    for _ in range(it):
        d.requires_grad_()
        pred_hat = decoder(x.detach_() + xi * d)
        logp_hat = torch.sigmoid(pred_hat)
        adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
        adv_distance.backward(retain_graph=True)
        d = _l2_normalize(d.grad)
        decoder.zero_grad()

    r_adv = d * eps
    return r_adv


class VATDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes, xi=1e-1, eps=10.0, iterations=1):
        super(VATDecoder, self).__init__()
        self.xi = xi
        self.eps = eps
        self.it = iterations
        self.upsample = Decoder(conv_in_ch, num_classes, upscale=upscale)

    def forward(self, x, _):
        r_adv = get_r_adv(x, self.upsample, self.it, self.xi, self.eps)
        x = self.upsample(x + r_adv)
        return x


def guided_cutout(output, upscale, resize, erase=0.4, use_dropout=False):
    masks = (output > 0.5).float().squeeze(dim=1)

    if use_dropout:
        p_drop = random.randint(3, 6)/10
        maskdroped = (F.dropout(masks, p_drop) > 0).float()
        maskdroped = maskdroped + (1 - masks)
        maskdroped.unsqueeze_(0)
        maskdroped = F.interpolate(maskdroped, size=resize, mode='nearest')

    masks_np = []
    for mask in masks:
        mask_np = np.uint8(mask.cpu().numpy())
        mask_ones = np.ones_like(mask_np)
        try: # Version 3.x
            _, contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except: # Version 4.x
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        polys = [c.reshape(c.shape[0], c.shape[-1]) for c in contours if c.shape[0] > 50]
        for poly in polys:
            min_w, max_w = poly[:, 0].min(), poly[:, 0].max()
            min_h, max_h = poly[:, 1].min(), poly[:, 1].max()
            bb_w, bb_h = max_w-min_w, max_h-min_h
            rnd_start_w = random.randint(0, int(bb_w*(1-erase)))
            rnd_start_h = random.randint(0, int(bb_h*(1-erase)))
            h_start, h_end = min_h+rnd_start_h, min_h+rnd_start_h+int(bb_h*erase)
            w_start, w_end = min_w+rnd_start_w, min_w+rnd_start_w+int(bb_w*erase)
            mask_ones[h_start:h_end, w_start:w_end] = 0
        masks_np.append(mask_ones)
    masks_np = np.stack(masks_np)

    maskcut = torch.from_numpy(masks_np).float().unsqueeze_(1)
    maskcut = F.interpolate(maskcut, size=resize, mode='nearest')

    if use_dropout:
        return maskcut.to(output.device), maskdroped.to(output.device)
    return maskcut.to(output.device)


class CutOutDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes, drop_rate=0.3, spatial_dropout=True, erase=0.4):
        super(CutOutDecoder, self).__init__()
        self.erase = erase
        self.upscale = upscale 
        self.upsample = Decoder(conv_in_ch, num_classes, upscale=upscale)

    def forward(self, x, pred=None):
        maskcut = guided_cutout(pred, upscale=self.upscale, erase=self.erase, resize=(x.size(2), x.size(3)))
        x = x * maskcut
        x = self.upsample(x)
        return x


def guided_masking(x, output, upscale, resize, return_msk_context=True):
    masks_context = (output > 0.5).float()
    masks_context = F.interpolate(masks_context, size=resize, mode='nearest')

    x_masked_context = masks_context * x
    if return_msk_context:
        return x_masked_context

    masks_objects = (1 - masks_context)
    x_masked_objects = masks_objects * x
    return x_masked_objects


class ContextMaskingDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes):
        super(ContextMaskingDecoder, self).__init__()
        self.upscale = upscale
        self.upsample = Decoder(conv_in_ch, num_classes, upscale=upscale)

    def forward(self, x, pred=None):
        x_masked_context = guided_masking(x, pred, resize=(x.size(2), x.size(3)),
                                          upscale=self.upscale, return_msk_context=True)
        x_masked_context = self.upsample(x_masked_context)
        return x_masked_context


class ObjectMaskingDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes):
        super(ObjectMaskingDecoder, self).__init__()
        self.upscale = upscale
        self.upsample = Decoder(conv_in_ch, num_classes, upscale=upscale)

    def forward(self, x, pred=None):
        x_masked_obj = guided_masking(x, pred, resize=(x.size(2), x.size(3)),
                                      upscale=self.upscale, return_msk_context=False)
        x_masked_obj = self.upsample(x_masked_obj)

        return x_masked_obj