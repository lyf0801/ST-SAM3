import argparse
import json
import logging
import os
import pprint
import math
import random
import time
from copy import deepcopy

import numpy as np
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import LambdaLR
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from tqdm import tqdm
import yaml

from models.dinov3_sam_student import get_dinov3_sam_student
from datasets.ORSI_SOD_dataset import ORSI_SOD_Dataset
from evaluator import Eval_thread
from misc import AvgMeter, check_mkdir
import models

# =========================
# Data Augmentation Helpers lyf: the same as train_fixmatch.py and train_unimatch.py
# =========================

def normalize(img, mask=None):
    # Using EORSSD mean/std
    mean = [0.3412, 0.3798, 0.3583]
    std = [0.1148, 0.1042, 0.0990]
    
    if isinstance(img, torch.Tensor):
        img = TF.normalize(img, mean, std)
    else:
        img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])(img)
        
    if mask is not None:
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(np.array(mask)).float() / 255.0
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        return img, mask
    return img


def blur(img, p=0.5):
    if random.random() < p:
        if isinstance(img, torch.Tensor):
            sigma = np.random.uniform(0.1, 2.0)
            kernel_size = int(2 * math.ceil(2 * sigma) + 1)
            img = TF.gaussian_blur(img, [kernel_size, kernel_size], [sigma, sigma])
        else:
            sigma = np.random.uniform(0.1, 2.0)
            img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img

def apply_random_rotation(img, mask=None, p=0.5):
    if random.random() < p:
        angle = random.uniform(-180, 180)
        img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)
        if mask is not None:
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
            return img, mask, angle
    return img if mask is None else (img, mask, 0)

def apply_random_scale(img, mask=None, scales=[0.25, 0.5], p=0.5):
    if random.random() < p:
        scale = random.choice(scales)
        # 此时 img 是 [B, C, H, W]
        h, w = img.shape[-2:]
        new_h, new_w = int(h * scale), int(w * scale)
        
        # 直接缩放整个 Batch [B, C, H, W] -> [B, C, new_h, new_w]
        img = F.interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=False)
        if mask is not None:
            # mask 是 [B, 1, H, W]
            mask = F.interpolate(mask, size=(new_h, new_w), mode='nearest')
            return img, mask
    return img if mask is None else (img, mask)

def obtain_cutmix_box(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    mask = torch.zeros(1, img_size, img_size)
    if random.random() > p:
        return mask

    size = np.random.uniform(size_min, size_max) * img_size * img_size
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)

        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break

    mask[:, y:y + cutmix_h, x:x + cutmix_w] = 1

    return mask

# =========================
# Train Dataset Class  lyf: the same as train_fixmatch.py and train_unimatch.py
# =========================

class SemiDataset(Dataset):
    def __init__(self, dataset_name, mode, size=448, labeled_percent='5%', nsample=None):
        self.dataset_name = dataset_name
        self.mode = mode
        self.size = size

        if dataset_name == "ORSSD":
            self.root = "/home/liuyanfeng/RSI_SOD/ORSSD dataset"
        elif dataset_name == "EORSSD":
            self.root = "/home/liuyanfeng/RSI_SOD/EORSSD dataset"
        elif dataset_name == "ORS_4199":
            self.root = "/home/liuyanfeng/RSI_SOD/ORS_4199 dataset"
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # 基础路径  最终三次迭代训练的数据都放在 Re_Re_Retrain_SAM3_Adapter_XXX_5% 目录下
        #ORSSD, EORSSD, ORS_4199 的SAM3_Adapter迭代3次后都放在 Re_Re_Retrain_SAM3_Adapter_XXX_5% 目录下
        self.dataset_base = f'./data/Re_Re_Retrain_SAM3_Adapter_{dataset_name}_{labeled_percent}'

        extended_list_path = os.path.join(self.dataset_base, f'extended_train{labeled_percent}.txt')
        extended_mask_dir = os.path.join(self.dataset_base, 'extended_labeled_mask')

        if not os.path.exists(extended_list_path):
            raise FileNotFoundError(f"Critical: Extended list {extended_list_path} not found! Please run select_reliable_subset.py first.")

        # 加载扩充后的所有标注列表
        extended_labeled_ids = [line.strip() for line in open(extended_list_path)]

        if mode == 'train_l':
            print(f"===> [SemiDataset] Loading EXTENDED labeled set from: {extended_list_path}")
            self.ids = extended_labeled_ids
            self.mask_root = extended_mask_dir
        elif mode == 'train_u':
            all_train_file = os.path.join(self.root, 'train.txt')
            all_prefixes = [line.strip() for line in open(all_train_file)]
            self.ids = [p for p in all_prefixes if p not in extended_labeled_ids]
            self.mask_root = os.path.join(self.root, 'labels')
            print(f"===> [SemiDataset] Loading Unlabeled set by excluding {len(extended_labeled_ids)} samples, remaining: {len(self.ids)}")

        if nsample is not None and len(self.ids) > 0:
            self.ids *= math.ceil(nsample / len(self.ids))
            self.ids = self.ids[:nsample]
            print(f"===> [SemiDataset] {mode} set resampled to {len(self.ids)}")

    def __getitem__(self, item):
        prefix = self.ids[item]
        img_path = os.path.join(self.root, 'images', prefix + '.jpg')
        lbl_path = os.path.join(self.mask_root, prefix + '.png')
        
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(lbl_path).convert('L')

        # Basic augmentations (Geometric)
        img = TF.resize(img, (self.size, self.size))
        mask = TF.resize(mask, (self.size, self.size), interpolation=Image.BILINEAR)             
        if random.random() > 0.5:
            img, mask = TF.hflip(img), TF.hflip(mask)
        if random.random() > 0.5:
            img, mask = TF.vflip(img), TF.vflip(mask)

        angle = random.choice([0, 90, 180, 270])
        if angle != 0:
            img, mask = TF.rotate(img, angle), TF.rotate(mask, angle)

        if self.mode == 'train_l':
            return normalize(img, mask)

        # Unlabeled data: Weak and Strong augmentations
        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)

        # Strong augmentation 1
        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(self.size, p=0.5)

        # Strong augmentation 2
        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(self.size, p=0.5)

        return normalize(img_w), normalize(img_s1), normalize(img_s2), cutmix_box1, cutmix_box2

    def __len__(self):
        return len(self.ids)

# =========================
# Loss and Helpers
# =========================

def structure_loss(pred, mask):
    wbce = F.binary_cross_entropy_with_logits(pred, mask)
    pred_sig = torch.sigmoid(pred)
    inter = (pred_sig * mask).sum(dim=(2, 3))
    union = (pred_sig + mask).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return wbce.mean() + wiou.mean()

# =========================
# Main Training Script lyf add: https://github.com/LiheYoung/UniMatch-V2/blob/main/unimatch_v2.py
# =========================

parser = argparse.ArgumentParser(description='刘焰锋拟投稿中国图象图形学报 DistillMatch-DINOv3 for ORSI-SOD')
parser.add_argument('--dataset', type=str, default='ORSSD', choices=['ORSSD', 'EORSSD', 'ORS_4199'], help='dataset name')
parser.add_argument('--labeled_percent', type=str, default='5%', help='Percentage of labeled data')
parser.add_argument('--save-path', type=str, default='', help='path to save checkpoints')
parser.add_argument('--model-name', type=str, default='DistillMatch_DINOv3', help='model name')
parser.add_argument('--img-size', type=int, default=448, help='input image size')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--batch-size', type=int, default=4, help='batch size')
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
parser.add_argument('--lr-min', type=float, default=0.0, help='minimum task-head learning rate')
parser.add_argument('--backbone-lr', type=float, default=5e-6, help='DINOv3 backbone learning rate')
parser.add_argument('--backbone-lr-min', type=float, default=0.0, help='minimum DINOv3 backbone learning rate')
parser.add_argument('--poly-power', type=float, default=0.9, help='per-iteration polynomial LR decay power')
parser.add_argument('--dinov3-checkpoint', type=str, default='pretrained/dinov3_vits16_pretrain_lvd1689m.pth')
parser.add_argument('--mobile-sam-checkpoint', type=str, default='pretrained/mobile_sam.pt')
parser.add_argument('--conf-thresh', type=float, default=0.95, help='confidence threshold')
parser.add_argument('--profile-compute-json', type=str, default='',
                    help='write one-epoch training time and peak CUDA memory, then skip evaluation')

"""
define ST++SAM3_Adapter model: lyf add:
"""
def get_SAM3_Adapter_model(args):
    config_path = 'configs/cod-sam-vit-l.yaml'
    
    args.model_path = './weights/ST-SAM3_{}_{}.pth'.format(args.dataset, args.labeled_percent)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 1. 加载配置和模型
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    model = models.make(config['model']).to(device)
    
    # lyf fix: 显式加载权重
    if os.path.exists(args.model_path):
        print(f"===> Loading SAM3 Teacher weights from {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint)
    else:
        raise FileNotFoundError(f"SAM3 teacher weights not found: {args.model_path}")
        
    return model


import torch
import torch.nn.functional as F

def compute_higher_order_moments(feature_map, orders=[1, 2, 3, 4]):
    """
    计算特征图的高阶矩
    
    Args:
        feature_map: 输入特征图, shape [B, C, H, W]
        orders: 要计算的矩的阶数列表
    
    Returns:
        moments: 各阶矩的字典
    """
    B, C, H, W = feature_map.shape
    
    # 展平空间维度 [B, C, H*W]
    flat_feat = feature_map.view(B, C, -1)
    
    moments = {}
    
    # 一阶矩 (均值)
    if 1 in orders:
        mean = flat_feat.mean(dim=2, keepdim=True)  # [B, C, 1]
        moments[1] = mean.squeeze(2)  # [B, C]
    
    # 二阶矩 (方差)
    if 2 in orders:
        mean = flat_feat.mean(dim=2, keepdim=True)
        variance = ((flat_feat - mean) ** 2).mean(dim=2)  # [B, C]
        moments[2] = variance
    
    # 三阶矩 (偏度)
    if 3 in orders:
        mean = flat_feat.mean(dim=2, keepdim=True)
        centered = flat_feat - mean
        skewness = (centered ** 3).mean(dim=2)  # [B, C]
        moments[3] = skewness
    
    # 四阶矩 (峰度)
    if 4 in orders:
        mean = flat_feat.mean(dim=2, keepdim=True)
        centered = flat_feat - mean
        kurtosis = (centered ** 4).mean(dim=2)  # [B, C]
        moments[4] = kurtosis
    
    return moments



def semantic_distillation(student_features, teacher_features, 
                        w_moments=0.25, w_mmd=1.0, moment_orders=[1, 2, 3, 4]):
    """
    语义蒸馏函数
    
    Args:
        student_features: 学生金字塔特征列表 [feat_1/4, feat_1/8, feat_1/16]
        teacher_features: SAM3教师FPN特征列表 [feat_1/4, feat_1/8, feat_1/16]
        w_moments: 高阶矩损失权重
        w_mmd: MMD损失权重
        moment_orders: 要使用的矩阶数
    
    Returns:
        total_loss: 总的语义蒸馏损失
    """
    total_loss = 0.0
    valid_layers = 0
    
    for s_feat, t_feat in zip(student_features, teacher_features):
        if s_feat is None or t_feat is None:
            continue
            
        # 尺寸匹配
        if s_feat.shape[2:] != t_feat.shape[2:]:
            t_feat = F.interpolate(t_feat, size=s_feat.shape[2:], mode='bilinear', align_corners=False)
        
        # 高阶矩损失
        s_moments = compute_higher_order_moments(s_feat, moment_orders)
        t_moments = compute_higher_order_moments(t_feat, moment_orders)
        
        moment_loss = 0.0
        for order in moment_orders:
            if order in s_moments and order in t_moments:
                moment_loss += F.mse_loss(s_moments[order], t_moments[order])
        

        # 累加所有有效层，返回前再按有效层数取平均
        total_loss += w_moments * moment_loss
        valid_layers += 1
    if valid_layers == 0:
        if student_features:
            return student_features[0].new_zeros(())
        return torch.tensor(0.0)
    return total_loss / valid_layers





def build_dinov3_optimizer(model, args, iterations_per_epoch):
    backbone_params = [param for param in model.backbone_parameters() if param.requires_grad]
    backbone_param_ids = {id(param) for param in backbone_params}
    task_params = [
        param for param in model.parameters()
        if param.requires_grad and id(param) not in backbone_param_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": task_params, "lr": args.lr, "name": "fusion_and_sam_decoder"},
            {"params": backbone_params, "lr": args.backbone_lr, "name": "dinov3_backbone"},
        ],
        weight_decay=1e-4,
    )
    total_iterations = max(1, args.epochs * iterations_per_epoch)

    def polynomial_factor(base_lr, min_lr):
        min_ratio = min_lr / base_lr

        def factor(step):
            progress = min(float(step) / total_iterations, 1.0)
            return min_ratio + (1.0 - min_ratio) * ((1.0 - progress) ** args.poly_power)

        return factor

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=[
            polynomial_factor(args.lr, args.lr_min),
            polynomial_factor(args.backbone_lr, args.backbone_lr_min),
        ],
    )
    return optimizer, scheduler, total_iterations


def main():
    args = parser.parse_args()
    print(f"\n=== Training on {args.dataset} with DistillMatch-DINOv3 framework ===")
    if args.img_size != 448:
        raise ValueError("This response experiment is fixed to 448x448 input")
    if args.lr <= 0 or args.backbone_lr <= 0:
        raise ValueError("Task and DINOv3 backbone learning rates must be positive")
    if not (0.0 <= args.lr_min <= args.lr and 0.0 <= args.backbone_lr_min <= args.backbone_lr):
        raise ValueError("Minimum learning rates must be within [0, base_lr]")
    if not args.save_path:
        args.save_path = os.path.join('./ablation_study/data', f"{args.model_name}_{args.dataset}_{args.labeled_percent}")
    os.makedirs(args.save_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #define ST++SAM3_Adapter model
    model_teacher = get_SAM3_Adapter_model(args)
    #Frozen teacher
    for param in model_teacher.parameters():
        param.requires_grad = False
    
    # DINOv3-S multi-layer fusion + SAM Mask Decoder student and its EMA copy
    model = get_dinov3_sam_student(
        dinov3_checkpoint=args.dinov3_checkpoint,
        mobile_sam_checkpoint=args.mobile_sam_checkpoint,
        img_size=args.img_size,
    ).to(device)
    model_ema = deepcopy(model).to(device)
    model_ema.eval()
    for param in model_ema.parameters():
        param.requires_grad = False

    # 1. 先获取真实的原始集合
    raw_dataset_l = SemiDataset(args.dataset, 'train_l', args.img_size, args.labeled_percent)
    raw_dataset_u = SemiDataset(args.dataset, 'train_u', args.img_size, args.labeled_percent)
    
    # 计算需要的总 batch 数：取折算后“轮数”较多的作为基准
    num_batches = max(math.ceil(len(raw_dataset_l) / args.batch_size), math.ceil(len(raw_dataset_u) / args.batch_size))
    
    # 最终对齐的目标长度
    target_len_l = num_batches * args.batch_size
    target_len_u = num_batches * args.batch_size
    
    print(f"--- Dataset Statistics ---")
    print(f"Actual Reliable Labeled: {len(raw_dataset_l)}")
    print(f"Remaining Unlabeled: {len(raw_dataset_u)}")
    print(f"Target Total Batches: {num_batches}")
    print(f"Resampled Lengths -> Labeled: {target_len_l}, Unlabeled: {target_len_u}")
    print(f"--------------------------")
    
    # 2. 根据比例进行重采样填充
    train_dataset_l = SemiDataset(args.dataset, 'train_l', args.img_size, args.labeled_percent, nsample=target_len_l)
    train_dataset_u = SemiDataset(args.dataset, 'train_u', args.img_size, args.labeled_percent, nsample=target_len_u)
    
    test_dataset = ORSI_SOD_Dataset(dataset_name=args.dataset, mode='test', img_size=args.img_size, aug=False)

    train_loader_l = DataLoader(train_dataset_l, batch_size=args.batch_size, shuffle=True, 
                               pin_memory=True, num_workers=4, drop_last=True)
    train_loader_u = DataLoader(train_dataset_u, batch_size=args.batch_size, shuffle=True, 
                               pin_memory=True, num_workers=4, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    print(f"--- Data Loader Ready ---")
    print(f"Batch Size: {args.batch_size + args.batch_size} (Labeled: {args.batch_size}, Unlabeled: {args.batch_size})")
    print(f"Iterations per Epoch: {len(train_loader_l)}")
    print(f"--------------------------")

    optimizer, lr_scheduler, total_iterations = build_dinov3_optimizer(
        model, args, len(train_loader_u)
    )
    task_param_count = sum(p.numel() for p in optimizer.param_groups[0]["params"])
    backbone_param_count = sum(p.numel() for p in optimizer.param_groups[1]["params"])
    print(f"DINOv3 optimizer: backbone_lr={args.backbone_lr:.2e}, "
          f"task_lr={args.lr:.2e}, poly_power={args.poly_power}, "
          f"total_iterations={total_iterations}")
    print(f"Trainable parameters: backbone={backbone_param_count:,}, "
          f"fusion_and_decoder={task_param_count:,}")

    best_fm = 0.0

    for epoch in range(args.epochs):
        if args.profile_compute_json and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        epoch_train_started = time.perf_counter()
        
        total_loss = AvgMeter()
        total_loss_x = AvgMeter()
        total_loss_u = AvgMeter()
        total_loss_feature_distillation = AvgMeter()
        total_loss_prediction_distillation = AvgMeter()

        total_mask_ratio = AvgMeter()
        batch_time = AvgMeter()
        
        loader = zip(train_loader_l, train_loader_u)

        model.train()
        
        for i, ((img_x, mask_x),
                (img_u_w, img_u_s1, img_u_s2, cutmix_box1, cutmix_box2)) in enumerate(loader):
            
            start_batch = time.time()

            img_x, mask_x = img_x.to(device), mask_x.to(device)
            img_u_w, img_u_s1, img_u_s2 = img_u_w.to(device), img_u_s1.to(device), img_u_s2.to(device)
            #cutmix_box1, cutmix_box2 = cutmix_box1.to(device), cutmix_box2.to(device)

            # 1. Generate Pseudo-labels using SAM3-Adapter Teacher Model and EMA Student Model (on weakly augmented images)
            with torch.no_grad():
                model_teacher.eval()
                prob_u_w = model_teacher(img_u_w).detach()
                model_ema.eval()
                prob_u_w_ema = model_ema(img_u_w).detach()
                # 取两者平均作为最终伪标签概率
                prob_u_w = (prob_u_w + prob_u_w_ema) / 2
                mask_u_w = (prob_u_w > 0.5).float()
                conf_u_w = torch.max(prob_u_w, 1 - prob_u_w)


            # 教师模型输入（保持 448 原始尺寸）
            img_u_s1_teacher = img_u_s1.clone()
            img_u_s2_teacher = img_u_s2.clone()

            # 学生模型输入应用扰动（缩放或旋转）
            img_u_s1_student, mask_u_s1_disturbed = apply_random_scale(img_u_s1, mask_u_w.clone(), scales=[0.25, 0.5, 2.0], p=0.5)
            img_u_s2_student, mask_u_s2_disturbed, rot_angle2 = apply_random_rotation(img_u_s2, mask_u_w.clone(), p=0.5)

    
            model.train()
            
            # Labeled loss
            pred_x, feature_x_student = model(img_x, return_feature=True)
            if isinstance(pred_x, tuple): pred_x = pred_x[0]

            # 学生模型前向传播（使用扰动后的输入）
            # 由于 batch 中图片尺寸可能被 apply_random_scale 改变得不一致，这里需要分开 forward 或者统一尺寸
            pred_u_s1_student, features_u_s1_student = model(img_u_s1_student, return_feature=True)
            pred_u_s2_student, features_u_s2_student = model(img_u_s2_student, return_feature=True)
           
            # 教师模型前向传播（使用原始 448 输入）
            pred_u_s1_teacher, features_u_s1_teacher = model_teacher((img_u_s1_teacher), return_feature=True)
            pred_u_s2_teacher, features_u_s2_teacher = model_teacher((img_u_s2_teacher), return_feature=True)
    
            for idx in range(len(features_u_s1_teacher)):
                features_u_s1_teacher[idx] = features_u_s1_teacher[idx].detach()
            for idx in range(len(features_u_s2_teacher)):
                features_u_s2_teacher[idx] = features_u_s2_teacher[idx].detach()

            # 对于旋转过的 s2 特征和预测，需要逆旋转回来与教师特征和一致性目标对齐
            if rot_angle2 != 0:
                features_u_s2_student_aligned = []
                for f_s in features_u_s2_student:
                    f_s_aligned = TF.rotate(f_s, -rot_angle2, interpolation=TF.InterpolationMode.BILINEAR)
                    features_u_s2_student_aligned.append(f_s_aligned)
                # 预测掩码逆旋转回原始 0 度坐标系
                pred_u_s2_student_aligned = TF.rotate(pred_u_s2_student, -rot_angle2, interpolation=TF.InterpolationMode.BILINEAR)
            else:
                features_u_s2_student_aligned = features_u_s2_student
                pred_u_s2_student_aligned = pred_u_s2_student

            # 特征蒸馏损失（使用对齐后的特征）
            loss_feature_distillation = (
                semantic_distillation(features_u_s1_student, features_u_s1_teacher) +
                semantic_distillation(features_u_s2_student_aligned, features_u_s2_teacher)
            ) / 2.0
            loss_feature_distillation = loss_feature_distillation * 0.01 



            loss_x = structure_loss(pred_x, mask_x)


            # 使用逆旋转对齐后的预测计算损失
            # 对于 s1（缩放扰动）：pred_u_s1_student 是缩放后的，mask_u_s1_disturbed 也是缩放后的
            loss_u_s1 = F.binary_cross_entropy_with_logits(pred_u_s1_student, mask_u_s1_disturbed, reduction='none')
            # 缩放后的置信度也需要对齐
            conf_mask_s1 = F.interpolate((conf_u_w >= args.conf_thresh).float(), size=pred_u_s1_student.shape[-2:], mode='nearest')
            loss_u_s1 = (loss_u_s1 * conf_mask_s1).mean()

            # 对于 s2（旋转扰动）：pred_u_s2_student_aligned 已经回到了原始坐标系，直接与原始 CutMix 伪标签对比
            loss_u_s2 = F.binary_cross_entropy_with_logits(pred_u_s2_student_aligned, mask_u_w, reduction='none')
            loss_u_s2 = (loss_u_s2 * (conf_u_w >= args.conf_thresh).float()).mean()

            loss_u =  (loss_u_s1 + loss_u_s2) / 2.0
            # UniMatch V2 Total Loss: lyf: change weights=1 for RSI_SOD
            loss = loss_x + loss_u + loss_feature_distillation

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # 5. Update EMA Model
            iters = epoch * len(train_loader_u) + i
            ema_ratio = min(1 - 1 / (iters + 1), 0.996)

            with torch.no_grad():
                for param, param_ema in zip(model.parameters(), model_ema.parameters()):
                    param_ema.copy_(param_ema * ema_ratio + param.detach() * (1 - ema_ratio))
                for buffer, buffer_ema in zip(model.buffers(), model_ema.buffers()):
                    buffer_ema.copy_(buffer_ema * ema_ratio + buffer.detach() * (1 - ema_ratio))

            total_loss.update(loss.item(), img_x.size(0))
            total_loss_x.update(loss_x.item(), img_x.size(0))
            total_loss_u.update(loss_u.item(), img_u_w.size(0))
            total_loss_feature_distillation.update(loss_feature_distillation.item(), img_u_w.size(0))
            
            mask_ratio = (conf_u_w >= args.conf_thresh).float().mean()
            total_mask_ratio.update(mask_ratio.item(), img_u_w.size(0))
            
            batch_time.update((time.time() - start_batch) * 1000)
            end = time.time()

            if i % 20 == 0:
                print(f'[epoch: {epoch:03d}] [iter: {i:05d}/{len(train_loader_u)}] [loss {total_loss.avg:.4f}] [loss_x {total_loss_x.avg:.4f}] [loss_u {total_loss_u.avg:.4f}] [loss_feature {total_loss_feature_distillation.avg:.4f}] [mask_ratio {total_mask_ratio.avg:.4f}] [task_lr {optimizer.param_groups[0]["lr"]:.10f}] [backbone_lr {optimizer.param_groups[1]["lr"]:.10f}] [time {batch_time.avg:.1f}ms]')


        if args.profile_compute_json:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                gib = 1024.0 ** 3
                peak_allocated_gib = torch.cuda.max_memory_allocated() / gib
                peak_reserved_gib = torch.cuda.max_memory_reserved() / gib
            else:
                peak_allocated_gib = peak_reserved_gib = 0.0
            epoch_train_seconds = time.perf_counter() - epoch_train_started
            profile_payload = {
                'dataset': args.dataset,
                'labeled_percent': args.labeled_percent,
                'epoch': epoch,
                'iterations': len(train_loader_u),
                'labeled_batch_size': args.batch_size,
                'unlabeled_batch_size': args.batch_size,
                'effective_images_per_iteration': args.batch_size * 2,
                'training_images_per_epoch': len(train_loader_u) * args.batch_size * 2,
                'reliable_labeled_unique': len(raw_dataset_l),
                'remaining_unlabeled_unique': len(raw_dataset_u),
                'total_unique_training_images': len(raw_dataset_l) + len(raw_dataset_u),
                'training_seconds_per_epoch': epoch_train_seconds,
                'training_ms_per_iteration': 1000.0 * epoch_train_seconds / max(len(train_loader_u), 1),
                'peak_allocated_gib': peak_allocated_gib,
                'peak_reserved_gib': peak_reserved_gib,
                'input_size': [args.img_size, args.img_size],
            }
            profile_path = os.path.abspath(args.profile_compute_json)
            os.makedirs(os.path.dirname(profile_path), exist_ok=True)
            temporary = profile_path + '.tmp'
            with open(temporary, 'w', encoding='utf-8') as handle:
                json.dump(profile_payload, handle, ensure_ascii=False, indent=2)
                handle.write('\n')
            os.replace(temporary, profile_path)
            print(f'===> Saved one-epoch compute profile: {profile_path}')
            return

        # Evaluation
        if epoch % 10 == 0 or epoch >= 70:
            # lyf add: According to the source paper: the UnimatchV2's authors report the mean Intersectionover-Union (mIoU) metric (↑) of the EMA teacher model, However, we find testing student model performs better for RSI-SOD.
            model_ema.eval()
            output_dir = os.path.join(args.save_path, 'output')
            os.makedirs(output_dir, exist_ok=True)
            thread = Eval_thread(epoch=epoch, model=model_ema, loader=test_loader, dataset=args.dataset, 
                                 method=args.model_name, txt_name=f"{args.model_name}_{args.dataset}.txt", output_dir=output_dir, cuda=True)
            log_line, fm = thread.run()
            print(log_line)
            
            if fm > best_fm:
                best_fm = fm
                torch.save(model_ema.state_dict(), os.path.join(args.save_path, f'best.pth'))
            
            print(f"Epoch {epoch} F-measure: {fm:.4f} Best F-measure: {best_fm:.4f}")



if __name__ == '__main__':
    main()
