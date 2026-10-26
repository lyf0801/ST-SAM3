import os 
import torch
import torch.nn as nn
from .deeplab_v3plus import Deeplab_V3plus
from .fcn8s import VGG16_FCN8s
from .hrnet import HRNet
from .segformer import SegFormer


def CreateModel(model, num_classes):
    return globals()[model](num_classes=num_classes)
            





