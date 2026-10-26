import os
import json
import torch
import torch.nn as nn
import numpy as np
import os.path as osp
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
from utils import Evaluator, unnormalize, upsample_scale
import cv2


def evaluate(loader, algorithm, device='cuda', num_classes=2, ignore_index=None):
    metric = Evaluator(num_classes)

    algorithm.eval()
    with torch.no_grad():
        for i, (image, label, _) in enumerate(loader):
            image, label = image.to(device), label.unsqueeze(dim=1).float().to(device)     
            prob = algorithm(image)   
            pred = (prob > 0.5).cpu().detach().numpy()
            label = label.cpu().detach().numpy()
            metric.add_batch(label, pred)

    acc = metric.Pixel_Accuracy()
    Precision, Recall = metric.Precision_Recall()
    IoU = metric.Intersection_over_Union()
    mIoU = metric.Mean_Intersection_over_Union()
    algorithm.train()

    return acc, Recall, Precision, IoU, mIoU



