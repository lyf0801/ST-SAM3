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


class DebiasPL(nn.Module):
    def __init__(self, args, device):
        super(DebiasPL, self).__init__()
        self.device = device
        self.name = 'DebiasPL'
        self.num_classes = args.num_classes
        self.ignore_index = args.ignore_index

        self.init_lr = args.learning_rate
        self.model = nn.DataParallel(CreateModel(args.model, self.num_classes).to(device))
        self.optimizer = optim.AdamW(self.model.module.optim_parameters(self.init_lr), lr=self.init_lr)
        self.bce_criterion = BCELoss(self.ignore_index)
        self.metric = Evaluator(self.num_classes+1)

        self.conf_thr = 0.95
        self.bias, self.mean = 0, 0
        self.momentum = 0.99
        self.lamda = 0.5

    def update_mean(self, bias, logits, momentum=0.99):
        logit_mean = logits.detach().reshape(-1).mean()
        bias = momentum * bias + (1 - momentum) * logit_mean 
        return bias

    def update_std(self, std, logits, momentum=0.99):
        logit_std = logits.detach().reshape(-1).std()
        std = momentum * std + (1 - momentum) * logit_std 
        return std

    def update(self, image_l, target_l, image_u, image_us, target_u, curr_iter, total_iters):
        self.optimizer.zero_grad()                                  
        self.model.train()
        self.model.module.adjust_learning_rate(
            self.init_lr, self.optimizer, curr_iter, total_iters)    

        # supervised learning
        feat_l = self.get_feats(image_l)         
        logit_l = self.get_logits(feat_l)         
        prob_l = torch.sigmoid(logit_l)
        loss_l = self.bce_criterion(prob_l, target_l) 

        # unsupervised learning
        prob_us = self.model(image_us)         
        with torch.no_grad():
            # calculate pseudo-labels
            feat_u = self.get_feats(image_u)
            logit_u = self.get_logits(feat_u)

            # update class mean & std with labeled data
            self.mean = self.update_mean(self.mean, logit_u, self.momentum)
            
            # calculate class bias
            self.bias = self.lamda * self.mean

            # calculate masks
            logit_u = logit_u - self.bias
            prob_u = torch.sigmoid(logit_u)
            pseudo_labels = (prob_u > 0.5).float()
            mask = ((prob_u > self.conf_thr) | (prob_u < 1 - self.conf_thr)).float()

        loss_u = (F.binary_cross_entropy(prob_us, pseudo_labels, reduction='none') * mask).mean()

        # overall loss
        loss = loss_l + loss_u
        loss.backward()
        self.optimizer.step()

        # performance of unlabeled data
        target_u = target_u.float().numpy()
        pred_u = pseudo_labels.squeeze(dim=1).cpu().detach().numpy()
        self.metric.add_batch(target_u, pred_u)

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
