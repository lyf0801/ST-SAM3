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


def cutmix(imgs, gts):
    batch_size, _, img_h, img_w = imgs.size()
    indices = torch.linspace(batch_size-1, 0, batch_size).long()
    shuffled_imgs = imgs[indices]
    shuffled_gts = gts[indices]

    lam = np.random.uniform(0, 1)
    cut_h, cut_w = int(img_h * lam), int(img_w * lam)

    x1, y1 = np.random.randint(img_h-cut_h), np.random.randint(img_w-cut_w)
    x2, y2 = x1 + cut_h, y1 + cut_w
    mask_cutmix = torch.zeros((batch_size, 1, img_h, img_w)).cuda()

    imgs = imgs*mask_cutmix + shuffled_imgs*(1 - mask_cutmix)
    gts = gts*mask_cutmix + shuffled_gts*(1 - mask_cutmix)
    return imgs.detach(), gts.detach()


class CutMix(nn.Module):
    def __init__(self, args, device):
        super(CutMix, self).__init__()
        self.device = device
        self.name = 'CutMix'
        self.num_classes = args.num_classes
        self.ignore_index = args.ignore_index
        self.metric = Evaluator(self.num_classes+1)

        self.init_lr = args.learning_rate
        self.model = nn.DataParallel(CreateModel(args.model, self.num_classes).to(self.device))
        self.optimizer = optim.AdamW(self.model.module.optim_parameters(self.init_lr), lr=self.init_lr)
        self.bce_criterion = BCELoss(self.ignore_index)
        self.conf_thr = 0.95

    def update(self, image_l, target_l, image_u, image_us, target_u, curr_iter, total_iters):
        self.optimizer.zero_grad()                                  
        self.model.train()
        self.model.module.adjust_learning_rate(
            self.init_lr, self.optimizer, curr_iter, total_iters)    

        # labeled learning
        prob_l = self.model(image_l)           
        loss_l = self.bce_criterion(prob_l, target_l) 

        # unlabeled learning
        prob_u = self.model(image_u)         
        pseudo_labels = (prob_u.detach() > 0.5).float()

        image_u_mix, pseudo_labels_mix = cutmix(image_u, pseudo_labels)
        prob_u_mix = self.model(image_u_mix)         
        loss_u = F.mse_loss(prob_u_mix, pseudo_labels_mix) 

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

