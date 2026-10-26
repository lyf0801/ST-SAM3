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


class DBMatch(nn.Module):
    def __init__(self, args, device):
        super(DBMatch, self).__init__()
        self.device = device
        self.name = 'DBMatch'
        self.num_classes = args.num_classes
        self.ignore_index = args.ignore_index

        self.init_lr = args.learning_rate
        self.model = nn.DataParallel(CreateModel(args.model, self.num_classes).to(device))
        self.optimizer = optim.AdamW(self.model.module.optim_parameters(self.init_lr), lr=self.init_lr)
        self.bce_criterion = BCELoss(self.ignore_index)
        self.metric = Evaluator(self.num_classes+1)

        self.fore_idx, self.back_idx = 1, 0
        self.fore_l_mean, self.back_l_mean = torch.tensor(20).to(device), torch.tensor(-20).to(device)
        self.fore_u_mean, self.back_u_mean = torch.tensor(20).to(device), torch.tensor(-20).to(device)
        
        self.bias = nn.Parameter(torch.tensor(0.0).to(device))
        self.conf_thr = 0.95
        self.momentum_bias = 0.999

        if 'Massachusetts_Buildings' in args.dataset: 
            self.lamda = 0.2
        elif 'Shenzhen' in args.dataset:  
            self.lamda = 0.2
        elif 'Tyrol' in args.dataset: 
            self.lamda = 0.4
        elif 'Vienna' in args.dataset:  
            self.lamda = 0.4 
        elif 'Chongqing' in args.dataset:  
            self.lamda = 0.6
        elif 'Kitsap' in args.dataset: 
            self.lamda = 1.2
        else:
            self.lamda = 1.0

    def update_mean(self, class_mean, logits, labels, class_idx, momentum=0.99):
        logits = logits.detach()[labels==class_idx].reshape(-1)
        if logits.shape[0] > 0:
            class_mean = momentum * class_mean + (1 - momentum) * logits.mean() 
        return class_mean

    def update(self, image_l, target_l, image_u, image_us, target_u, curr_iter, total_iters):
        self.optimizer.zero_grad()                                  
        self.model.train()
        self.model.module.adjust_learning_rate(
            self.init_lr, self.optimizer, curr_iter, total_iters)    

        # supervised learning
        feat_l = self.get_feats(image_l)         
        logit_l = self.get_logits(feat_l)         
        prob_l = torch.sigmoid(logit_l)
        pred_l = (prob_l > 0.5).float()
        loss_l = self.bce_criterion(prob_l, target_l) 

        # unsupervised learning      
        prob_us = self.model(image_us)

        with torch.no_grad():
            self.model.eval()
            feat_u = self.get_feats(image_u)         
            logit_u = self.get_logits(feat_u)         
            pseudo_labels = (logit_u > 0).float()

            # update bias
            self.fore_l_mean = self.update_mean(
                self.fore_l_mean, logit_l, pred_l, self.fore_idx, self.momentum_bias)
            self.back_l_mean = self.update_mean(
                self.back_l_mean, logit_l, pred_l, self.back_idx, self.momentum_bias)

            self.fore_u_mean = self.update_mean(
                self.fore_u_mean, logit_u, pseudo_labels, self.fore_idx, self.momentum_bias)
            self.back_u_mean = self.update_mean(
                self.back_u_mean, logit_u, pseudo_labels, self.back_idx, self.momentum_bias)

            self.bias_mean = (self.fore_u_mean / self.back_u_mean).abs() \
                           - (self.fore_l_mean / self.back_l_mean).abs() 
            self.bias.data = self.lamda * self.bias_mean * torch.abs(self.back_u_mean)

            # calculate pseudo-labels and masks
            logit_u = logit_u - self.bias.data
            prob_u = torch.sigmoid(logit_u)
            pseudo_labels = (prob_u > 0.5).float()
            mask = (prob_u > self.conf_thr) | (prob_u < 1 - self.conf_thr)
            
        self.model.train()

        loss_u = (F.binary_cross_entropy(prob_us, pseudo_labels, reduction='none') * mask).mean()

        # overall loss
        loss = loss_l + loss_u
        loss.backward()
        self.optimizer.step()

        # performance of unlabeled data
        target_u = target_u.float().numpy()
        pseudo_labels = pseudo_labels.squeeze(dim=1).cpu().detach().numpy()
        self.metric.add_batch(target_u, pseudo_labels)
        return {'loss_l': loss_l.item(), 'loss_u': loss_u.item()}

    def forward(self, image):
        feat = self.get_feats(image)         
        logit = self.get_logits(feat) 
        logit = self.debias_logits(logit)
        prob = torch.sigmoid(logit)
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

    def debias_logits(self, logit):
        logit = logit - self.bias    
        return logit



