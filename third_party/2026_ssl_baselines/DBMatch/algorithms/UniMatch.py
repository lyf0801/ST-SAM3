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


class UniMatch(nn.Module):
    def __init__(self, args, device):
        super(UniMatch, self).__init__()
        self.device = device
        self.name = 'UniMatch'
        self.num_classes = args.num_classes
        self.ignore_index = args.ignore_index
        self.metric = Evaluator(self.num_classes+1)

        self.init_lr = args.learning_rate
        self.model = nn.DataParallel(CreateModel(args.model, self.num_classes).to(self.device))
        self.optimizer = optim.AdamW(self.model.module.optim_parameters(self.init_lr), lr=self.init_lr)
        self.bce_criterion = BCELoss(self.ignore_index)
        self.conf_thr = 0.95

    def update(self, image_l, target_l, image_u, image_us, \
                image_us2, target_u, curr_iter, total_iters):
        self.optimizer.zero_grad()                                  
        self.model.train()
        self.model.module.adjust_learning_rate(
            self.init_lr, self.optimizer, curr_iter, total_iters)    

        # labeled learning
        prob_l = self.model(image_l)   
        loss_l = self.bce_criterion(prob_l, target_l) 

        # unlabeled learning
        with torch.no_grad():
            feat_u = self.get_feats(image_u)   
        prob_u = self.get_probs(feat_u).detach()

        pseudo_labels = (prob_u > 0.5).float()
        mask = ((prob_u > self.conf_thr) | (prob_u < 1 - self.conf_thr)).float()

        prob_up = self.get_probs(F.dropout(feat_u, p=0.5))
        prob_us = self.model(image_us)   
        prob_us2 = self.model(image_us2)   

        loss_u = ((F.binary_cross_entropy(prob_us, pseudo_labels, reduction='none')*mask).mean() \
               +  (F.binary_cross_entropy(prob_us2, pseudo_labels, reduction='none')*mask).mean() \
               +  (F.binary_cross_entropy(prob_up, pseudo_labels, reduction='none')*mask).mean()) / 3

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

