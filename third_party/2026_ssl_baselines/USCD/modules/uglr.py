import torch
import torch.nn as nn
import torch.nn.functional as F


class UGLR(nn.Module):
    def __init__(self, gamma_labeled=2.0, gamma_unlabeled=-1.0):
        super().__init__()
        self.gL = gamma_labeled
        self.gU = gamma_unlabeled

    def _weights(self, unc, gamma):
        w = torch.exp(gamma * unc)
        norm = w.reshape(w.shape[0], -1).sum(1).reshape(-1, 1, 1)
        return w / (norm + 1e-8)

    def supervised_loss(self, logits, labels, unc):
        w = self._weights(unc, self.gL)
        ce = F.cross_entropy(logits, labels, reduction='none')
        return (w * ce).sum()

    def unsupervised_loss(self, logits, pseudo, unc, confident_mask=None):
        w = self._weights(unc, self.gU)
        ce = F.cross_entropy(logits, pseudo, reduction='none')
        if confident_mask is not None:
            ce = ce * confident_mask.float()
            w  = w  * confident_mask.float()
        return (w * ce).sum()

    def forward(self, logits_L, labels, unc_L,
                logits_U, pseudo, unc_U, confident_mask=None):
        l_sup   = self.supervised_loss(logits_L, labels, unc_L)
        l_unsup = self.unsupervised_loss(logits_U, pseudo, unc_U, confident_mask)
        return l_sup, l_unsup
