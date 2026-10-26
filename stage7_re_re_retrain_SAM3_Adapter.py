import argparse
import logging
import os
import pprint
import math
import random
from copy import deepcopy

import numpy as np
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from tqdm import tqdm

import yaml
import models
from datasets.ORSI_SOD_dataset import ORSI_SOD_Dataset
from evaluator import Eval_thread
from misc import AvgMeter, check_mkdir

# =========================
# Data Augmentation Helpers lyf: the same as train_fixmatch.py and train_unimatch.py
# =========================

def normalize(img, mask=None):
    # Using EORSSD mean/std
    mean = [0.3412, 0.3798, 0.3583]
    std = [0.1148, 0.1042, 0.0990]
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])(img)
    if mask is not None:
        mask = torch.from_numpy(np.array(mask)).float() / 255.0
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        return img, mask
    return img


def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img

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

        self.root = os.environ.get("STMATCH_DATASET_ROOT", self.root)

        # 基础路径
        # 载入stage4生成的扩充标注列表和对应的mask目录
        self.dataset_base = f'./data/Re_Retrain_SAM3_Adapter_{dataset_name}_{labeled_percent}'
        extended_list_path = os.path.join(self.dataset_base, f'extended_train{labeled_percent}.txt')
        extended_mask_dir = os.path.join(self.dataset_base, 'extended_labeled_mask')
        labeled_list_path = os.path.join(self.root, f'train{labeled_percent}.txt')
        if not os.path.exists(extended_list_path):
            raise FileNotFoundError(f"Critical: Extended list {extended_list_path} not found! Please run select_reliable_subset.py first.")

        # 加载扩充后的所有标注列表
        extended_labeled_ids = [line.strip() for line in open(extended_list_path)]
        labeled_ids_list = [line.strip() for line in open(labeled_list_path)]
        self.labeled_ids_set = set(labeled_ids_list)
        
        if mode == 'train_l':
            print(f"===> [SemiDataset] Loading EXTENDED labeled set from: {extended_list_path}")
            self.ids = extended_labeled_ids
            self.mask_root = extended_mask_dir
        elif mode == 'train_u':
            all_train_file = os.path.join(self.root, 'train.txt')
            all_prefixes = [line.strip() for line in open(all_train_file)]
            self.ids = [p for p in all_prefixes if p not in self.labeled_ids_set]
            self.mask_root = os.path.join(self.root, 'labels')
            print(f"===> [SemiDataset] Loading Unlabeled set from: {all_train_file} with {len(self.ids)} samples")

        if nsample is not None and len(self.ids) > 0:
            self.ids *= math.ceil(nsample / len(self.ids))
            self.ids = self.ids[:nsample]
            print(f"===> [SemiDataset] {mode} set resampled to {len(self.ids)}")

    def __getitem__(self, item):
        prefix = self.ids[item]
        img_path = os.path.join(self.root, 'images', prefix + '.jpg')
        lbl_path = os.path.join(self.root, 'labels', prefix + '.png')
        
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(lbl_path).convert('L')

        img = TF.resize(img, (self.size, self.size))
        mask = TF.resize(mask, (self.size, self.size), interpolation=Image.BILINEAR)  
        
        # For unlabeled data, we only return the image and prefix (for saving pseudo labels later)
        #no need for basic geometric augmentations since we will generate pseudo labels at the original size and then apply augmentations in the next stage
        if self.mode == 'train_u':
            return normalize(img), prefix

        # Basic augmentations (Geometric)          
        if random.random() > 0.5:
            img, mask = TF.hflip(img), TF.hflip(mask)
        if random.random() > 0.5:
            img, mask = TF.vflip(img), TF.vflip(mask)

        angle = random.choice([0, 90, 180, 270])
        if angle != 0:
            img, mask = TF.rotate(img, angle), TF.rotate(mask, angle)

        if self.mode == 'train_l':
            return normalize(img, mask)


    def __len__(self):
        return len(self.ids)

# =========================
# Core Functions
# =========================

def structure_loss(pred, mask):
    valid_mask = (mask != 255).float() # 255 is ignore
    wbce = F.binary_cross_entropy_with_logits(pred, mask.clamp(0, 1), reduction='none')
    wbce = (wbce * valid_mask).sum(dim=(2, 3)) / (valid_mask.sum(dim=(2, 3)) + 1e-8)
    pred_sig = torch.sigmoid(pred)
    inter = (pred_sig * mask * valid_mask).sum(dim=(2, 3))
    union = (pred_sig * valid_mask + mask * valid_mask).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return wbce.mean() + wiou.mean()

def init_SAM3_Adapter(args, device):
   
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('Config loaded.')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model
    torch.manual_seed(args.seed)

    # Build model from config
    if 'model' not in config:
        raise RuntimeError('Config must contain a model specification.')
    model = models.make(config['model']).to(device).train()

    # Load SAM checkpoint if provided (partial load with key remapping similar to train.py)
    if config.get('sam_checkpoint'):
        ckpt = torch.load(config['sam_checkpoint'], map_location='cpu')
        
        # Support older checkpoint formats where weights may be wrapped
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]

        new_state_dict = {}
        ref_state_dict = model.state_dict()

        print(f"Loading checkpoint: {config.get('sam_checkpoint')}")

        for k, v in ckpt.items():
            #lyf add
            if 'freqs_cis' in k:
                print(f"Skipping freqs_cis key: {k}")  # 跳过，不加载
                continue
            # remap common prefixes from detector-style checkpoints to this model
            if k.startswith("detector.backbone."):
                new_k = k.replace("detector.backbone.", "image_encoder.")
            elif "mask_decoder" in k:
                suffix = k.split("mask_decoder.")[-1]
                new_k = f"mask_decoder.{suffix}"
            elif "pe_layer" in k:
                suffix = k.split("pe_layer.")[-1]
                new_k = f"pe_layer.{suffix}"
            elif "no_mask_embed" in k:
                new_k = "no_mask_embed.weight"
            else:
                new_k = k

            # only keep keys that exist in the reference model and have matching shape
            if new_k in ref_state_dict:
                ref_shape = ref_state_dict[new_k].shape
                if v.shape != ref_shape:
                    print(f"Warning: Skipping {new_k} due to shape mismatch. Ckpt: {v.shape} vs Model: {ref_shape}")
                    continue

            if new_k:
                new_state_dict[new_k] = v

        msg = model.load_state_dict(new_state_dict, strict=False)

    # Freeze image encoder (optional behavior retained)
    for name, para in model.named_parameters():
        if 'image_encoder' in name and 'prompt_generator' not in name:
            para.requires_grad_(False)

    return model

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n>>>> Training on {args.dataset} (Labeled {args.labeled_percent}) <<<<")
    
    save_path = f"./data/{args.model_name}_{args.dataset}_{args.labeled_percent}"
    os.makedirs(save_path, exist_ok=True)
    pseudo_path = os.path.join(save_path, "pseudo_labels")
    os.makedirs(pseudo_path, exist_ok=True)
    
    # 1. Prepare Data
    # Labeled Train
    train_dataset_l = SemiDataset(args.dataset, 'train_l', args.img_size, args.labeled_percent, nsample=None)
    # 如果样本太少（比如 < 200），进行简单扩充，保证 epoch 有足够迭代次数
    if len(train_dataset_l) < 200:
        train_dataset_l.ids *= math.ceil(200 / len(train_dataset_l))
        train_dataset_l.ids = train_dataset_l.ids[:200]

    print(f"{args.labeled_percent} {args.dataset} dataset has Extended Labeled samples: {len(train_dataset_l)}")

    train_loader_l = DataLoader(train_dataset_l, batch_size=args.batch_size, shuffle=True, 
                                pin_memory=True, num_workers=4, drop_last=True)
    
    # Unlabeled for Pseudo Labeling
    train_dataset_u = SemiDataset(args.dataset, 'train_u', args.img_size, args.labeled_percent)
    loader_u = DataLoader(train_dataset_u, batch_size=1, drop_last=False)
    
    # Test
    test_dataset = ORSI_SOD_Dataset(dataset_name=args.dataset, mode='test', img_size=args.img_size, aug=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # 2. Init Model & Optimizer
    model = init_SAM3_Adapter(args, device)
    
    # 极简 EMA 实现
    ema_model = deepcopy(model).to(device)
    for param in ema_model.parameters():
        param.requires_grad = False
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)
    
    best_fm = 0.0

    # 3. Training Loop
    for epoch in range(args.epochs):
        model.train()
        loss_meter = AvgMeter()
        for i, (img, mask) in enumerate(train_loader_l):
            img, mask = img.to(device), mask.to(device)

            pred = model(img)
            if isinstance(pred, tuple): pred = pred[0]
            
            # 使用标准的结构损失
            loss = structure_loss(pred, mask)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update EMA Model
            iters = epoch * len(train_loader_l) + i
            ema_ratio = min(1 - 1 / (iters + 1), 0.996)

            with torch.no_grad():
                for param, param_ema in zip(model.parameters(), ema_model.parameters()):
                    param_ema.copy_(param_ema * ema_ratio + param.detach() * (1 - ema_ratio))
                for buffer, buffer_ema in zip(model.buffers(), ema_model.buffers()):
                    buffer_ema.copy_(buffer_ema * ema_ratio + buffer.detach() * (1 - ema_ratio))

            loss_meter.update(loss.item(), img.shape[0])

            if i % 20 == 0:
                print(f'[{args.dataset} - epoch: {epoch:03d}] [iter: {i:05d}] [loss {loss_meter.avg:.5f}] [lr {optimizer.param_groups[0]["lr"]:.10f}]')
        
        scheduler.step()
        
        # 4. Evaluation
        if epoch % 10 == 0 or epoch >= 15:
            output_dir = os.path.join(save_path, 'output')
            os.makedirs(output_dir, exist_ok=True)
            
            # 仅评估 EMA 模型性能
            ema_model.eval()
            thread_ema = Eval_thread(epoch=epoch, model=ema_model, loader=test_loader, dataset=args.dataset, 
                                     method=args.model_name, txt_name=f"{args.model_name}_{args.dataset}.txt", 
                                     output_dir=output_dir, cuda=True)
            log_line, fm_ema = thread_ema.run()
            print(f"[{args.dataset}] EMA: {log_line}")
            
            if fm_ema > best_fm: 
                best_fm = fm_ema
                save_name = os.path.join(save_path, f"best.pth")
                torch.save(ema_model.state_dict(), save_name)
                print(f"===> [{args.dataset}] New Best EMA FM: {fm_ema:.4f}, saved to {save_name}")

    # 5. Pseudo Labeling for THIS dataset using its BEST weights
    print(f"===> Generating Pseudo Labels for {args.dataset}...")
    best_pth = os.path.join(save_path, "best.pth")
    if os.path.exists(best_pth):
        model.load_state_dict(torch.load(best_pth, map_location=device))
        model.eval()
        
        with torch.no_grad():
            for img, prefix in tqdm(loader_u, desc=f"Labeling {args.dataset}"):    
                pred = model(img.to(device))
                if isinstance(pred, tuple): pred = pred[0] 
                pred = (pred > 0.5).float() * 255  #model.eval() has torch.sigmoid
                Image.fromarray(pred.squeeze().cpu().numpy().astype(np.uint8)).save(
                    os.path.join(pseudo_path, prefix[0] + ".png"))
    else:
        print(f"Skip labeling: No best.pth found.")


def parse_args():
    parser = argparse.ArgumentParser(description='Stage7: Re_Re_Retraining SAM3_Adapter on labeled RSI SOD datasets.')
    parser.add_argument('--dataset', type=str, default='ORSSD', choices=['ORSSD', 'EORSSD', 'ORS_4199'], help='Dataset to train on.')
    parser.add_argument('--labeled_percent', type=str, default='5%', help='Percentage of labeled data: 5%, 10%, etc.')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--lr_min', type=float, default=4e-5)
    parser.add_argument('--img-size', type=int, default=448)
    parser.add_argument('--model-name', type=str, default='Re_Re_Retrain_SAM3_Adapter')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--config', default='configs/cod-sam-vit-l.yaml', help='YAML config containing model spec.')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    print(f"\n=== Training on {args.dataset} with Re_Retrain_SAM3_Adapter ===")
    datasets = ['ORS_4199']
    for ds in datasets:
        args.dataset = ds
        print(f"\n=== Training on {args.dataset} with Re_Retrain_SAM3_Adapter ===")
        main(args)
