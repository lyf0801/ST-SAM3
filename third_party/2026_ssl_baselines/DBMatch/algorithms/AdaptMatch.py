import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import torch.distributions as dist

import random
import numpy as np
from copy import deepcopy
from utils import BCELoss, IoULoss, Evaluator, upsample_scale, upsample_size
from models import CreateModel


class AdaptMatch(nn.Module):
    def __init__(self, args, device):
        super(AdaptMatch, self).__init__()
        self.device = device
        self.name = 'AdaptMatch'
        self.num_classes = args.num_classes
        self.ignore_index = args.ignore_index

        self.init_lr = args.learning_rate
        self.model = nn.DataParallel(CreateModel(args.model, self.num_classes).to(device))
        self.optimizer = optim.AdamW(self.model.module.optim_parameters(self.init_lr), lr=self.init_lr)
        self.bce_criterion = BCELoss(self.ignore_index)
        self.metric = Evaluator(self.num_classes+1)

        self.back_idx, self.fore_idx = 0, 1
        self.thr_back, self.thr_fore = 0, 1
        self.probs_list_l = {self.fore_idx: [], self.back_idx: []}
        self.probs_list_u = {self.fore_idx: [], self.back_idx: []}
        self.momentum = 0.99

    def update_mean(self, probs_mean, probs, labels, class_idx, momentum=0.99):
        probs = probs.detach()[labels==class_idx].reshape(-1)
        if probs.shape[0] > 0:
            probs_mean = momentum * probs_mean + (1 - momentum) * probs.mean() 
        return probs_mean

    def update(self, image_l, target_l, image_u, image_us, target_u, curr_iter, total_iters):
        self.optimizer.zero_grad()                                  
        self.model.train()
        self.model.module.adjust_learning_rate(
            self.init_lr, self.optimizer, curr_iter, total_iters)    

        # supervised learning
        prob_l = self.model(image_l)           
        loss_l = self.bce_criterion(prob_l, target_l) 
        
        # unsupervised learning
        prob_us = self.model(image_us)         
        with torch.no_grad():
            prob_u = self.model(image_u)       
            pseudo_labels = (prob_u > 0.5).float()
            mask = (prob_u > self.thr_fore) | (prob_u < 1 - self.thr_back)
            
            # accumulate weights   
            list_length = 100
            self.probs_list_l = accumulate_probs_list(list_length, \
                self.probs_list_l, prob_l.detach(), target_l, self.fore_idx, self.back_idx)
            self.probs_list_u = accumulate_probs_list(list_length*5, \
                self.probs_list_u, prob_us.detach(), pseudo_labels, self.fore_idx, self.back_idx)

            # calculate thr: labeled & unlabeled
            if curr_iter > list_length:
                self.thr_fore = calc_threshold(
                    self.probs_list_l[self.fore_idx] + self.probs_list_u[self.fore_idx])
                self.thr_back = calc_threshold(
                    self.probs_list_l[self.back_idx] + self.probs_list_u[self.back_idx])
        
        loss_u = (F.binary_cross_entropy(prob_us, pseudo_labels, reduction='none') * mask).mean()

        target_u = target_u.float().numpy()
        prob_u = pseudo_labels.squeeze(dim=1).cpu().detach().numpy()
        self.metric.add_batch(target_u, prob_u)

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



def accumulate_probs_list(list_length, probs_hist, prob, target, fore_idx, back_idx):
    h, w = 64, 64
    b = prob.size()[0]

    target = F.interpolate(target, size=[h, w], mode='nearest').squeeze(dim=1).view(b, h*w).view(b*h*w)
    prob = F.interpolate(prob, size=[h, w], mode='nearest').squeeze(dim=1).view(b, h*w).view(b*h*w)
    mask_fore, mask_back = (target==fore_idx), (target==back_idx)
    prob_fore = prob[mask_fore].tolist()
    prob_back = prob[mask_back].tolist()

    if prob_fore != []:
        probs_hist[fore_idx].append(prob_fore)
    if prob_back != []:
        probs_hist[back_idx].append(prob_back)

    if len(probs_hist[fore_idx]) > list_length:
        probs_hist[fore_idx].pop(0)
    if len(probs_hist[back_idx]) > list_length:
        probs_hist[back_idx].pop(0)

    return probs_hist


def calc_threshold(probs_hist):
    total_sum = 0
    total_num = 1e-7
    for inner_list in probs_hist:
        total_num += len(inner_list)
        total_sum += sum(inner_list)
    thr = total_sum / total_num
    return thr