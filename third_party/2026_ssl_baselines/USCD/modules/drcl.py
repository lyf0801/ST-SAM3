from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryBank:
    def __init__(self, size=512):
        self.bank = deque(maxlen=size)

    def update(self, proto):
        self.bank.append(proto.detach().cpu())

    def get(self, n, device):
        if not self.bank:
            return None
        n = min(n, len(self.bank))
        return torch.stack(list(self.bank)[-n:]).to(device)


class DRCL(nn.Module):
    def __init__(self, feat_dim=256, num_anchors=32, num_samples=64,
                 temperature=0.1, memory_bank_size=512, global_weight=0.5):
        super().__init__()
        self.Nr = num_anchors
        self.Ns = num_samples
        self.tau = temperature
        self.gw = global_weight

        self.mem_fg = MemoryBank(memory_bank_size)
        self.mem_bg = MemoryBank(memory_bank_size)

        self.proj = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 1),
            nn.BatchNorm2d(feat_dim), nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim, feat_dim, 1),
        )

    def _reliable(self, prob_ori, prob_aug):
        return prob_ori.argmax(1) == prob_aug.argmax(1)

    def _difficult(self, unc, thr=0.5):
        return unc > thr

    def _infonce(self, q, pos, neg):
        q = F.normalize(q, dim=-1)
        pos = F.normalize(pos, dim=-1)
        neg = F.normalize(neg, dim=-1)
        p = torch.exp((pos @ q) / self.tau).sum()
        n = torch.exp((neg @ q) / self.tau).sum()
        return -torch.log(p / (p + n + 1e-8))

    def _hard_mine(self, feats, unc):
        n = min(2 * self.Ns, feats.shape[0])
        perm = torch.randperm(feats.shape[0], device=feats.device)[:n]
        _, top = unc[perm].topk(min(self.Ns, n))
        return feats[perm][top]

    def _resize(self, t, size):
        return F.interpolate(t.unsqueeze(1).float(), size=size,
                             mode='nearest').squeeze(1)

    def local_loss(self, feat, labels, reliable, difficult, unc):
        B, D, Hf, Wf = feat.shape
        size = (Hf, Wf)
        lab  = self._resize(labels, size).long()
        rel  = self._resize(reliable.float(), size).bool()
        diff = self._resize(difficult.float(), size).bool()
        unc_ = F.interpolate(unc.unsqueeze(1), size=size,
                             mode='bilinear', align_corners=False).squeeze(1)

        loss, cnt = torch.tensor(0., device=feat.device), 0
        for b in range(B):
            valid = rel[b] & diff[b]
            f = feat[b].permute(1, 2, 0)   # [Hf, Wf, D]

            for cls, other in [(1, 0), (0, 1)]:
                anc_mask = valid & (lab[b] == cls)
                pos_mask = valid & (lab[b] == cls)
                neg_mask = valid & (lab[b] == other)
                if anc_mask.sum() < 1 or pos_mask.sum() < 1 or neg_mask.sum() < 1:
                    continue

                anchors = f[anc_mask]
                n_anc = min(self.Nr, anchors.shape[0])
                anchors = anchors[torch.randperm(anchors.shape[0],
                                                  device=feat.device)[:n_anc]]

                pos_s = self._hard_mine(f[pos_mask], unc_[b][pos_mask])
                neg_s = self._hard_mine(f[neg_mask], unc_[b][neg_mask])

                bl = sum(self._infonce(anchors[k], pos_s, neg_s)
                         for k in range(n_anc)) / n_anc
                loss = loss + bl
                cnt += 1

        return loss / max(cnt, 1)

    def global_loss(self, feat, labels, reliable, difficult):
        B, D, Hf, Wf = feat.shape
        size = (Hf, Wf)
        lab  = self._resize(labels, size).long()
        rel  = self._resize(reliable.float(), size).bool()
        diff = self._resize(difficult.float(), size).bool()
        valid = rel & diff

        device = feat.device
        loss, cnt = torch.tensor(0., device=device), 0

        for b in range(B):
            f = feat[b].permute(1, 2, 0)
            fg = valid[b] & (lab[b] == 1)
            bg = valid[b] & (lab[b] == 0)
            if fg.sum() < 1 or bg.sum() < 1:
                continue

            m_fg = f[fg].mean(0)
            m_bg = f[bg].mean(0)
            self.mem_fg.update(m_fg)
            self.mem_bg.update(m_bg)

            neg_fg = self.mem_bg.get(self.Ns, device)
            neg_bg = self.mem_fg.get(self.Ns, device)
            if neg_fg is None or neg_bg is None:
                continue

            loss = (loss
                    + self._infonce(m_fg, m_fg.unsqueeze(0), neg_fg)
                    + self._infonce(m_bg, m_bg.unsqueeze(0), neg_bg))
            cnt += 1

        return loss / max(cnt, 1)

    def forward(self, feat, labels, prob_ori, prob_aug, unc):
        proj = self.proj(feat)
        rel  = self._reliable(prob_ori, prob_aug)
        diff = self._difficult(unc)
        l_local  = self.local_loss(proj, labels, rel, diff, unc)
        l_global = self.global_loss(proj, labels, rel, diff)
        return l_local + self.gw * l_global, l_local, l_global
