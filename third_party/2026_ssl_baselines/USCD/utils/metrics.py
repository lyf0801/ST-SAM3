class ChangeDetectionMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.TP = self.FP = self.FN = self.TN = 0

    def update(self, pred, target):
        pred   = pred.cpu().long()
        target = target.cpu().long()
        self.TP += ((pred == 1) & (target == 1)).sum().item()
        self.FP += ((pred == 1) & (target == 0)).sum().item()
        self.FN += ((pred == 0) & (target == 1)).sum().item()
        self.TN += ((pred == 0) & (target == 0)).sum().item()

    @property
    def precision(self):
        d = self.TP + self.FP
        return self.TP / d if d else 0.0

    @property
    def recall(self):
        d = self.TP + self.FN
        return self.TP / d if d else 0.0

    @property
    def f1(self):
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def iou(self):
        d = self.TP + self.FP + self.FN
        return self.TP / d if d else 0.0

    def summary(self):
        return {k: round(v * 100, 2) for k, v in
                {'F1': self.f1, 'IoU': self.iou,
                 'Precision': self.precision, 'Recall': self.recall}.items()}
