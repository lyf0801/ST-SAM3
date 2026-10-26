import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from sklearn.metrics import f1_score


def upsample_size(x, size, mode='bilinear'):
    if mode == 'bilinear':
        return F.interpolate(x, size, mode=mode, align_corners=False)
    else:
        return F.interpolate(x, size, mode=mode)

def upsample_scale(x, scale, mode='bilinear'):
    if mode == 'bilinear':
        return F.interpolate(x, scale_factor=scale, mode=mode, align_corners=False)
    else:
        return F.interpolate(x, scale_factor=scale, mode=mode)

def unnormalize(image):
    # 归一化使用的均值和标准差
    mean = torch.tensor([0.485, 0.456, 0.406]).cuda().view(1, 3, 1, 1)  # Bx3x1x1 以适应 Bx3xHxW
    std = torch.tensor([0.229, 0.224, 0.225]).cuda().view(1, 3, 1, 1)  # Bx3x1x1

    # 逆归一化操作
    image = image * std + mean
    return image

def update_soft_conf_matrix(conf_matrix, logits, labels):
    probs = torch.softmax(logits, dim=1)  # 将 logits 转换为概率
    labels_one_hot = torch.nn.functional.one_hot(labels, num_classeses=probs.size(1)).float()
    # 使用外积来更新混淆矩阵
    conf_matrix += torch.matmul(labels_one_hot.transpose(0, 1), probs)
    return conf_matrix
    

class Evaluator(object):
    def __init__(self, num_class, ignore_index=None):
        self.num_class = num_class
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc_class = np.nanmean(Acc)  # nanmean is used to ignore NaN values which may arise if any class was ignored completely
        return Acc_class

    def Precision_Recall(self):
        Precision = np.diag(self.confusion_matrix) / (np.sum(self.confusion_matrix, axis=0) + 1e-10)
        Recall = np.diag(self.confusion_matrix) / (np.sum(self.confusion_matrix, axis=1) + 1e-10)
        return Precision, Recall

    def F1_Score(self):
        Precision, Recall = self.Precision_Recall()
        F1 = 2 * Precision * Recall / (Precision + Recall)
        return F1

    def Mean_F1_Score(self):
        F1 = self.F1_Score()
        Mean_F1 = np.nanmean(F1)
        return Mean_F1

    def Mean_Intersection_over_Union(self):
        IoU = self.Intersection_over_Union()
        MIoU = np.nanmean(IoU)
        return MIoU

    def Intersection_over_Union(self):
        IoU = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix))
        return IoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = self.Intersection_over_Union()
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class) & (gt_image != self.ignore_index)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        gt_image = gt_image.astype(np.int64)
        pre_image = pre_image.astype(np.int64)
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class IoULoss(nn.Module):
    """Jaccard/IoU loss of binary or multi-class
    Args:
        num_classes: number of classes
        ignore_index: label value to ignore
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
    """
    def __init__(self, num_classes=2, ignore_index=-1, smooth=1):
        super(IoULoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, predict, target):
        # predict = F.softmax(predict, dim=1)
        # target = make_one_hot(target.unsqueeze(dim=1), self.num_classes)

        total_loss = 0
        for i in range(self.num_classes):
            inputs = predict[:, i].reshape(-1)
            targets = target[:, i].reshape(-1)
            
            # Create a mask where both inputs and targets are not equal to ignore_index
            valid_mask = (targets != self.ignore_index)
            if torch.any(valid_mask):
                inputs_valid = inputs[valid_mask]
                targets_valid = targets[valid_mask]

                # Intersection is equivalent to True Positive count
                intersection = (inputs_valid * targets_valid).sum()
                total = (inputs_valid + targets_valid).sum()
                union = total - intersection
                IoU = (intersection + self.smooth) / (union + self.smooth)
                loss = 1 - IoU
                total_loss += loss

        total_loss /= self.num_classes
        return  total_loss



class BCELoss(nn.Module):
    def __init__(self, ignore_index=-1):
        super(BCELoss, self).__init__()
        self.ignore_index = ignore_index
        self.bce_loss = nn.BCELoss(reduction='none')  # 使用'none'以便后续应用掩码

    def forward(self, inputs, targets):
        """
        计算除了忽略索引之外的BCE损失。
        
        Args:
            inputs (torch.Tensor): 预测值张量，形状为 [B, 1, H, W]
            targets (torch.Tensor): 目标值张量，形状为 [B, 1, H, W]
        
        Returns:
            torch.Tensor: 计算后的损失值。
        """
        # 创建掩码，这里忽略特定值
        mask = (targets != self.ignore_index)

        # 应用掩码，这里为了避免错误计算，未被掩码的位置设置为合理的值
        inputs_masked = inputs * mask
        targets_masked = targets * mask
        loss = self.bce_loss(inputs_masked, targets_masked)
        loss = torch.mean(loss[mask])  # 只在掩码为True的地方计算平均值
        return loss