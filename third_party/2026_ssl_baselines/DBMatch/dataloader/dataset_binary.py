import os
import os.path as osp

import random
import numpy as np
from copy import deepcopy
from PIL import Image

import torch
from torch.utils import data
from torchvision import transforms

from .rand_augment import horizontal_flip, vertical_flip, rotate, random_resize_and_crop
from .rand_augment import color_swap, hsv_shift, adjust_color_hue


class RS_Binary_DataSet(data.Dataset):
    def __init__(self, root, dataset, list_path, algorithm, \
                    base_size=512, set_mode='test', ignore_index=-1):
        self.root = root  
        self.algorithm = algorithm
        self.ignore_index = ignore_index

        self.base_size = [base_size, base_size]
        self.set_mode = set_mode
        set_mode_list = ['test', 'labeled_training', 'unlabeled_training', 'unlabeled_training_multi']
        assert self.set_mode in set_mode_list, f"Invalid name: {name}. Must be one of {set_mode_list}"

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        self.paths = []         
        curr_list_path = osp.join(root, dataset, list_path)
        f = open(curr_list_path)            
        for path in f.readlines():
            image_path, label_path = path.strip('\n').split(' ')
            image_path = osp.join(dataset, image_path)
            label_path = osp.join(dataset, label_path)
            path = image_path + ' ' + label_path
            self.paths.append(path)
        f.close()  
        print (dataset, ': ', len(self.paths))
        if len(self.paths) < 50:
            self.paths = self.paths + self.paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image_path, label_path = self.paths[index].split(' ')
        image = Image.open(osp.join(self.root, image_path)).convert('RGB')
        label = Image.open(osp.join(self.root, label_path.strip('\n'))).convert('L')
        image = image.resize(self.base_size, Image.BILINEAR)
        label = label.resize(self.base_size, Image.NEAREST)

        # spatial transform
        if self.set_mode=='labeled_training':
            image, label = horizontal_flip(image, label)
            image, label = vertical_flip(image, label)
            image, label = rotate(image, label)
            image, label = random_resize_and_crop(image, label, self.ignore_index)
            
            image = self.normalize(self.to_tensor(image))
            label = np.asarray(label).copy()
            label[label==255] = 1
            return image, label, [image_path, label_path]

        # strong augmented unlabeled data
        elif self.set_mode=='unlabeled_training':
            image, label = horizontal_flip(image, label)
            image, label = vertical_flip(image, label)
            image, label = rotate(image, label)
            image, label = random_resize_and_crop(image, label, self.ignore_index)
            
            image_s = deepcopy(image)
            image_s = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.0)(image_s)
            image_s = adjust_color_hue(image_s, max_temp_change=0.5)
            image_s = transforms.RandomGrayscale(p=0.2)(image_s)
            image_s = transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))(image_s)

            image_s = self.normalize(self.to_tensor(image_s))
            image = self.normalize(self.to_tensor(image))
            label = np.asarray(label).copy()
            label[label==255] = 1
            return image, image_s, label, [image_path, label_path]

        elif self.set_mode=='unlabeled_training_multi':
            image, label = horizontal_flip(image, label)
            image, label = vertical_flip(image, label)
            image, label = rotate(image, label)
            image, label = random_resize_and_crop(image, label, self.ignore_index)
            
            image_s = deepcopy(image)
            image_s = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.0)(image_s)
            image_s = adjust_color_hue(image_s, max_temp_change=0.5)
            image_s = transforms.RandomGrayscale(p=0.2)(image_s)
            image_s = transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))(image_s)

            image_s_bar = deepcopy(image)
            image_s_bar = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.0)(image_s_bar)
            image_s_bar = adjust_color_hue(image_s_bar, max_temp_change=0.5)
            image_s_bar = transforms.RandomGrayscale(p=0.2)(image_s_bar)
            image_s_bar = transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))(image_s_bar)
            
            image_s = self.normalize(self.to_tensor(image_s))
            image_s_bar = self.normalize(self.to_tensor(image_s_bar))
            image = self.normalize(self.to_tensor(image))
            label = np.asarray(label).copy()
            label[label==255] = 1
            return image, image_s, image_s_bar, label, [image_path, label_path]

        else:
            image = self.normalize(self.to_tensor(image))
            label = np.asarray(label).copy()
            label[label==255] = 1
            return image, label, [image_path, label_path]

