import random
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image


class WeakAugmentation:
    def __init__(self, img_size=256):
        self.img_size = img_size

    def __call__(self, img_A, img_B, label=None):
        if random.random() > 0.5:
            img_A, img_B = TF.hflip(img_A), TF.hflip(img_B)
            if label is not None:
                label = TF.hflip(label)

        if random.random() > 0.5:
            img_A, img_B = TF.vflip(img_A), TF.vflip(img_B)
            if label is not None:
                label = TF.vflip(label)

        angle = random.choice([0, 90, 180, 270])
        if angle != 0:
            img_A, img_B = TF.rotate(img_A, angle), TF.rotate(img_B, angle)
            if label is not None:
                label = TF.rotate(label, angle)

        img_A = TF.resize(img_A, [self.img_size, self.img_size])
        img_B = TF.resize(img_B, [self.img_size, self.img_size])
        if label is not None:
            label = TF.resize(label, [self.img_size, self.img_size],
                              interpolation=TF.InterpolationMode.NEAREST)
        return img_A, img_B, label


class ToTensor:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img_A, img_B, label=None):
        img_A = TF.normalize(TF.to_tensor(img_A), self.mean, self.std)
        img_B = TF.normalize(TF.to_tensor(img_B), self.mean, self.std)
        if label is not None:
            label = (torch.from_numpy(np.array(label)) > 0).long()
        return img_A, img_B, label


class TrainTransform:
    def __init__(self, img_size=256, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.aug = WeakAugmentation(img_size)
        self.to_tensor = ToTensor(mean, std)

    def __call__(self, img_A, img_B, label=None):
        img_A, img_B, label = self.aug(img_A, img_B, label)
        return self.to_tensor(img_A, img_B, label)


class TestTransform:
    def __init__(self, img_size=256, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.img_size = img_size
        self.to_tensor = ToTensor(mean, std)

    def __call__(self, img_A, img_B, label=None):
        img_A = TF.resize(img_A, [self.img_size, self.img_size])
        img_B = TF.resize(img_B, [self.img_size, self.img_size])
        if label is not None:
            label = TF.resize(label, [self.img_size, self.img_size],
                              interpolation=TF.InterpolationMode.NEAREST)
        return self.to_tensor(img_A, img_B, label)
