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


class OnlySup(nn.Module):
    def __init__(self, args, device):
        super(OnlySup, self).__init__()
        self.device = device
        self.name = 'OnlySup'
        self.num_classes = args.num_classes
        self.ignore_index = args.ignore_index

        self.init_lr = args.learning_rate
        self.metric = Evaluator(self.num_classes+1)
        self.model = nn.DataParallel(CreateModel(args.model, self.num_classes).to(self.device))
        self.optimizer = optim.AdamW(self.model.module.optim_parameters(self.init_lr), lr=self.init_lr)
        self.bce_criterion = BCELoss(self.ignore_index)

    def update(self, image_l, target_l, curr_iter, total_iters):
        self.optimizer.zero_grad()                                  
        self.model.train()
        self.model.module.adjust_learning_rate(
            self.init_lr, self.optimizer, curr_iter, total_iters)    

        # low-resolution learning
        prob_l = self.model(image_l)           
        loss_l = self.bce_criterion(prob_l, target_l) 

        # overall loss
        loss = loss_l
        loss.backward()
        self.optimizer.step()

        return {'loss_l': loss_l.item(), 'loss_u': 0}

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

