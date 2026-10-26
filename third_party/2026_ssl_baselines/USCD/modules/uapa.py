import math
import torch


class UAPA:
    def __init__(self, window_n=16, beta=0.3,
                 paste_ratio_max=0.5, paste_ratio_min=0.1):
        self.N = window_n
        self.beta = beta
        self.rho_max = paste_ratio_max
        self.rho_min = paste_ratio_min

    def _window_scores(self, unc_map):
        B, H, W = unc_map.shape
        N = self.N
        h, w = H // N, W // N
        u = unc_map[:, :h * N, :w * N].reshape(B, N, h, N, w)
        return u.mean(dim=[2, 4])

    def _protected(self, scores, epoch, total):
        B, N, _ = scores.shape
        K = max(1, math.ceil(self.beta * (epoch / max(total, 1)) * N * N))
        flat = scores.reshape(B, -1)
        _, top = flat.topk(K, dim=1)
        mask = torch.zeros(B, N * N, dtype=torch.bool, device=scores.device)
        mask.scatter_(1, top, True)
        return mask.reshape(B, N, N)

    def _sources(self, prob, protected, epoch, total):
        B, H, W = prob.shape
        N = self.N
        h, w = H // N, W // N
        density = prob[:, :h * N, :w * N].reshape(B, N, h, N, w).mean(dim=[2, 4])

        progress = epoch / max(total, 1)
        rho = self.rho_max * (1 - progress) + self.rho_min

        src = torch.zeros(B, N, N, dtype=torch.bool, device=prob.device)
        flat_d = density.reshape(B, -1)
        flat_p = protected.reshape(B, -1)
        for b in range(B):
            avail = ~flat_p[b]
            n = avail.sum().item()
            if n == 0:
                continue
            M = max(1, int(rho * n))
            d = flat_d[b].clone()
            d[flat_p[b]] = -1.0
            _, top = d.topk(M)
            s = torch.zeros(N * N, dtype=torch.bool, device=prob.device)
            s.scatter_(0, top, True)
            src[b] = s.reshape(N, N)
        return src

    def _pixel_mask(self, protected, source_b, H, W):
        B, N, _ = protected.shape
        h, w = H // N, W // N
        mask = torch.ones(B, 1, H, W, device=protected.device)
        for b in range(B):
            for i in range(N):
                for j in range(N):
                    y1, y2 = i * h, (i + 1) * h
                    x1, x2 = j * w, (j + 1) * w
                    if source_b[b, i, j]:
                        mask[b, 0, y1:y2, x1:x2] = 0.0
                    if protected[b, i, j]:
                        mask[b, 0, y1:y2, x1:x2] = 1.0
        return mask

    def __call__(self, img_A_a, img_B_a, pseudo_a,
                 img_A_b, img_B_b, pseudo_b,
                 unc_a, prob_a, epoch, total):
        scores_a = self._window_scores(unc_a)
        protected = self._protected(scores_a, epoch, total)
        source = self._sources(prob_a, protected, epoch, total)

        M = self._pixel_mask(protected, source, img_A_a.shape[2], img_A_a.shape[3])
        mix_A = M * img_A_a + (1 - M) * img_A_b
        mix_B = M * img_B_a + (1 - M) * img_B_b
        mix_L = (M.squeeze(1) * pseudo_a.float()
                 + (1 - M.squeeze(1)) * pseudo_b.float()).long()
        return mix_A, mix_B, mix_L
