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
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image, ImageFilter, ImageOps, ImageDraw, ImageFont
from torchvision import transforms
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from tqdm import tqdm

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

        # Load labeled prefixes
        labeled_file = os.path.join(self.root, f'train{labeled_percent}.txt')
        labeled_prefixes = [line.strip() for line in open(labeled_file)]

        if mode == 'train_l' and nsample is not None:
            self.ids = labeled_prefixes
            # Ensure labeled set is at least as large as unlabeled set for zip (by repeating)
            self.ids *= math.ceil(nsample / len(self.ids))
            self.ids = self.ids[:nsample]
        elif mode == 'train_u':
            # Load all training prefixes and exclude labeled ones
            all_train_file = os.path.join(self.root, 'train.txt')
            all_prefixes = [line.strip() for line in open(all_train_file)]
            self.ids = [p for p in all_prefixes if p not in labeled_prefixes]
        else:
            # For validation, we use the standard ORSI_SOD_Dataset logic
            pass

    def __getitem__(self, item):
        prefix = self.ids[item]
        img_path = os.path.join(self.root, 'images', prefix + '.jpg')
        lbl_path = os.path.join(self.root, 'labels', prefix + '.png')
        
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(lbl_path).convert('L')

        img, mask = normalize(img, mask)
        return img, mask, prefix

    def __len__(self):
        return len(self.ids)

import os
import cv2
import torch
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader

# 导入 SAM3 相关模块
from models.sam3.model.geometry_encoders import Prompt
from models.sam3.model.box_ops import box_xyxy_to_cxcywh
from models import make # 假设模型创建函数在此

def get_prompts_from_mask(mask_np):
    """
    从二进制掩码中提取多点和多框 Prompt
    多框：所有面积 > 100 的连通区域各提取一个框
    前景点：每个连通区域提取一个质心
    背景点：四个角点 + 随机背景点
    """
    H, W = mask_np.shape
    mask_8u = (mask_np * 255).astype(np.uint8)
    
    # 1. 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_8u, connectivity=8)
    
    # 筛选面积大于 100 的区域
    valid_indices = [i for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= 100]
    
    if len(valid_indices) == 0:
        if num_labels > 1:
            valid_indices = [1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])]
        else:
            return None

    # --- 提取 Multi-Box ---
    boxes_cxcywh = []
    for i in valid_indices:
        x, y, w, h, _ = stats[i]
        # 转换为 cxcywh 并归一化到 [0, 1]
        cx, cy = (x + w/2) / W, (y + h/2) / H
        nw, nh = w / W, h / H
        boxes_cxcywh.append([cx, cy, nw, nh])
    
    multi_boxes = torch.tensor(boxes_cxcywh, dtype=torch.float32) # [N, 4]

    # --- 提取 Points ---
    fg_points = []
    for i in valid_indices:
        cx, cy = centroids[i]
        fg_points.append([cx / W, cy / H])

    # 3. 负样本点 (背景)
    bg_points = [] 
    bg_coords = np.argwhere(mask_np <= 0.5)
    if len(bg_coords) > 0:
        # 保持背景点数量至少与前景点相当
        n_bg = len(fg_points)
        n_to_sample = min(len(bg_coords), n_bg)
        curr_state = np.random.get_state()
        np.random.seed(42)
        idx_bg = np.random.choice(len(bg_coords), n_to_sample, replace=False)
        np.random.set_state(curr_state)
        for idx in idx_bg:
            y, x = bg_coords[idx]
            bg_points.append([x / W, y / H])

    points = fg_points + bg_points
    labels = [1] * len(fg_points) + [0] * len(bg_points)
    
    return {
        "points": torch.tensor([points], dtype=torch.float32), 
        "point_labels": torch.tensor([labels], dtype=torch.long)
    }

def calculate_gpa_score(pred_mask, ref_mask):
    """
    精细化像素级质量评估 (PLADC): 针对局部缺失和背景幻觉
    """
    pred_mask = (pred_mask > 0.5).astype(np.float32)
    ref_mask = (ref_mask > 0.5).astype(np.float32)
    pred_sum = pred_mask.sum()
    
    inter = (pred_mask * ref_mask).sum()
    union = (pred_mask + ref_mask).sum() - inter
    s_iou = inter / (union + 1e-8)
    
    p_8u = (pred_mask * 255).astype(np.uint8)
    r_8u = (ref_mask * 255).astype(np.uint8)
    p_edge = cv2.Laplacian(p_8u, cv2.CV_32F)
    r_edge = cv2.Laplacian(r_8u, cv2.CV_32F)
    s_fit = 1.0 - np.clip(np.abs(p_edge - r_edge).mean() / 255.0, 0, 1)
    
    if pred_sum < 10:
        s_miss = 0.0 if ref_mask.sum() > 10 else 1.0
        s_hallu = 1.0
    else:
        # 缺失检测
        missing_mask = (pred_mask == 0) & (ref_mask == 1)
        s_miss = 1.0 - np.clip(missing_mask.sum() / (pred_sum + 1e-8), 0, 1)
        # 幻觉检测
        hallucination_mask = (pred_mask == 1) & (ref_mask == 0)
        s_hallu = 1.0 - np.clip(hallucination_mask.sum() / (pred_sum + 1e-8), 0, 1)
    
    total_score = (s_iou + s_fit + s_miss + s_hallu) / 4.0
    return total_score, (s_iou, s_fit, s_miss, s_hallu), (pred_sum, ref_mask.sum())

def filter_reliable_samples(model_name, model, dataloader, device, output_txt, labeled_list_path):
    model.eval()
    gap = 10
    
    with open(labeled_list_path, 'r') as f:
        labeled_ids = [line.strip() for line in f.readlines()]

    dataset_base = f'./data/{model_name}_{dataloader.dataset.dataset_name}_{args.labeled_percent}'
    comp_dir = os.path.join(dataset_base, 'pseudo_labels_comparison')
    target_dir = os.path.join(dataset_base, 'extended_labeled_mask')
    
    print(f"===> Clearing and Preparing directories in {dataset_base}...")
    for d in [comp_dir, target_dir]:
        if os.path.exists(d): shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    print(f"===> Processing samples to rank by (SAM_Score + Stability)...")
    
    all_candidates = []

    with torch.no_grad():
        for img_tensor, _, prefix in tqdm(dataloader):
            img_tensor = F.interpolate(img_tensor, size=(448, 448), mode='bilinear', align_corners=False).to(device)
            prefix_id = prefix[0] if isinstance(prefix, (list, tuple)) else prefix
            
            initial_pseudo_path = getattr(dataloader.dataset, 'pseudo_path', None)
            pseudo_label_path = os.path.join(initial_pseudo_path, prefix_id + '.png')
            if not os.path.exists(pseudo_label_path): continue
                
            curr_mask_np = np.array(Image.open(pseudo_label_path).convert('L'))
            curr_mask_np = cv2.resize(curr_mask_np, (448, 448))
            original_pseudo = (curr_mask_np > 127).astype(np.float32)
            
            # --- 准备工作 ---
            features = model.image_encoder(img_tensor)
            high_res_features = [
                model.mask_decoder.conv_s0(features["backbone_fpn"][0]),
                model.mask_decoder.conv_s1(features["backbone_fpn"][1])
            ]
            image_embed = features["backbone_fpn"][-1]
            if model.directly_add_no_mem_embed:
                image_embed = image_embed + model.no_mask_embed.weight.view(1, -1, 1, 1)
            
            image_pe = model.get_dense_pe()
            if image_pe.shape[-2:] != image_embed.shape[-2:]:
                image_pe = F.interpolate(image_pe, size=image_embed.shape[-2:], mode='bilinear', align_corners=False)

            # --- TTA 计算 ---
            img_flipped = torch.flip(img_tensor, [-1])
            pred_flipped_logits = model(img_flipped)
            prob_flipped_back = torch.flip(pred_flipped_logits, [-1]).squeeze()
            tta_mask_np = (prob_flipped_back > 0.5).float().cpu().numpy()
            stability_score, tta_details, _ = calculate_gpa_score(original_pseudo, tta_mask_np)

            # --- SAM 迭代 ---
            step_data = []
            iter_masks = [original_pseudo]
            temp_mask_np = curr_mask_np.copy()

            for iter_idx in range(2):
                try:
                    prompts = get_prompts_from_mask((temp_mask_np > 127).astype(np.float32))
                    if prompts is None: break
                    
                    pt_coords = prompts["points"].to(device)
                    pt_labels = prompts["point_labels"].to(device)
                    sparse_embeddings = model.pe_layer.forward_with_coords(pt_coords * 447.0 + 0.5, (448, 448))
                    
                    for i in range(pt_labels.shape[1]):
                        label = pt_labels[0, i].item()
                        sparse_embeddings[:, i, :] += model.point_embeddings[label].weight
                    
                    dense_h, dense_w = image_embed.shape[-2:]
                    dense_embeddings = model.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(1, -1, dense_h, dense_w)
                    
                    low_res_masks, _, _, _, = model.mask_decoder(
                        image_embeddings=image_embed, image_pe=image_pe,
                        sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings,
                        multimask_output=True, repeat_image=False, high_res_features=high_res_features,
                    )
                    
                    refined_logits = model.postprocess_masks(low_res_masks, 448, 448)
                    prob_map = torch.sigmoid(refined_logits).squeeze()
                    avg_entropy = (-prob_map * torch.log(prob_map + 1e-8) - (1 - prob_map) * torch.log(1 - prob_map + 1e-8)).mean().item()
                    
                    refined_mask_np = (prob_map > 0.5).float().cpu().numpy()
                    score, details, _ = calculate_gpa_score(original_pseudo, refined_mask_np)
                    
                    step_data.append({'score': score, 'details': details, 'ent': avg_entropy})
                    temp_mask_np = (refined_mask_np * 255).astype(np.uint8)
                    iter_masks.append(refined_mask_np)
                except: break

            if len(step_data) == 2:
                current_sam_avg = step_data[1]['score'] 
                total_sum_score = current_sam_avg + stability_score
                
                all_candidates.append({
                    'id': prefix_id,
                    'total_score': total_sum_score,
                    'sam_score': current_sam_avg,
                    'stability': stability_score,
                    'step_data': step_data,
                    'iter_masks': iter_masks,
                    'tta_mask_np': tta_mask_np,
                    'tta_details': tta_details,
                    'original_pseudo': original_pseudo
                })

    # --- TOP-100% 筛选 ---
    all_candidates.sort(key=lambda x: x['total_score'], reverse=True)
    num_to_keep = len(all_candidates)
    reliable_unlabeled = []

    print(f"===> Ranking finished. Selecting TOP {num_to_keep} samples (100% of {len(all_candidates)} candidates).")

    # 准备字体
    font_path = "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"
    if not os.path.exists(font_path): font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
    try:
        font_status = ImageFont.truetype(font_path, 34)
        font_main = ImageFont.truetype(font_path, 32)
        font_detail = ImageFont.truetype(font_path, 30)
    except:
        font_status = font_main = font_detail = ImageFont.load_default()

    for idx, cand in enumerate(tqdm(all_candidates, desc="Saving Results")):
        is_reliable = idx < num_to_keep
        prefix_id = cand['id']
        
        # Final Mask (Voting + Post-process)
        vote_map = cand['original_pseudo'] + cand['iter_masks'][-1] + cand['tta_mask_np']
        consensus_mask = (vote_map > 1.5).astype(np.float32)
        c_8u = (consensus_mask * 255).astype(np.uint8)
        
        contours, hierarchy = cv2.findContours(c_8u, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is not None:
            for c_idx, h in enumerate(hierarchy[0]):
                if h[3] != -1: cv2.drawContours(c_8u, [contours[c_idx]], -1, 255, -1)
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(c_8u, connectivity=8)
        for l_idx in range(1, num_labels):
            if stats[l_idx, cv2.CC_STAT_AREA] < 100: c_8u[labels == l_idx] = 0
        
        if is_reliable:
            reliable_unlabeled.append(prefix_id)
            cv2.imwrite(os.path.join(target_dir, prefix_id + '.png'), c_8u)
        
        # 可视化 logic
        img_path = os.path.join(dataloader.dataset.root, 'images', prefix_id + '.jpg')
        img_pil = Image.open(img_path).convert('RGB').resize((448, 448))
        canvas = Image.new('RGB', (448*6 + gap*5, 448 + 140), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        
        titles = ["Original Image", "Initial Prediction", "SAM Iterative Prediction1", "SAM Iterative Prediction2", "TTA Prediction", "Final Pseudo-Label"]
        for i, title in enumerate(titles):
            sx = i * (448 + gap)
            # 针对较长标题调整起始偏移量以对齐居中感
            offset_x = 10 if i < 2 else 5
            draw.text((sx + offset_x, 2), title, font=font_main, fill=(0, 0, 0))
            if i == 0:
                img_draw = img_pil.copy()
                p_viz = get_prompts_from_mask(cand['original_pseudo'])
                if p_viz is not None:
                    pts = p_viz["points"][0].cpu().numpy()
                    lbls = p_viz["point_labels"][0].cpu().numpy()
                    for (px, py), pl in zip(pts, lbls):
                        color = (0, 255, 0) if pl == 1 else (255, 0, 0)
                        draw_v = ImageDraw.Draw(img_draw)
                        draw_v.ellipse([px*448-5, py*448-5, px*448+5, py*448+5], fill=color, outline=(255,255,255))
                canvas.paste(img_draw, (sx, 40))
            elif i <= 3: canvas.paste(Image.fromarray((cand['iter_masks'][i-1]*255).astype(np.uint8)).convert('RGB'), (sx, 40))
            elif i == 4: canvas.paste(Image.fromarray((cand['tta_mask_np']*255).astype(np.uint8)).convert('RGB'), (sx, 40))
            elif i == 5: canvas.paste(Image.fromarray(c_8u).convert('RGB'), (sx, 40))

        base_y = 448 + 40
        status_color = (0, 150, 0) if is_reliable else (200, 0, 0)
        draw.text((15, base_y + 8), f"Reliable: {is_reliable} (Rank: {idx+1}/{len(all_candidates)})", font=font_status, fill=status_color)
        draw.text((15, base_y + 52), f"Total Score: {cand['total_score']:.3f} (SAM:{cand['sam_score']:.2f} + TTA:{cand['stability']:.2f})", font=font_detail, fill=(0, 0, 0))
        
        cv2.imwrite(os.path.join(comp_dir, f"{prefix_id}_comp.png"), cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR))

    # 最后写入 txt
    final_list = labeled_ids + reliable_unlabeled
    with open(os.path.join(dataset_base, output_txt), 'w') as f:
        for pid in final_list: f.write(pid + '\n')

    labeled_root = os.path.join(dataloader.dataset.root, 'labels')
    for pid in labeled_ids:
        src = os.path.join(labeled_root, pid + '.png')
        if os.path.exists(src): shutil.copy(src, os.path.join(target_dir, pid + '.png'))

    print(f"===> Done! Original: {len(labeled_ids)}, Added: {len(reliable_unlabeled)}, Total: {len(final_list)}")

if __name__ == "__main__":
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/cod-sam-vit-l.yaml')
    parser.add_argument('--labeled_percent', type=str, default='5%',  help="labeled percentage: 5%% or 10%%")
    parser.add_argument('--model_name', type=str, default='Re_Retrain_SAM3_Adapter', help="stage name for directory naming")
    parser.add_argument('--img_size', type=int, default=448)
    parser.add_argument('--dataset', choices=['ORSSD', 'EORSSD', 'ORS_4199'], default=None)
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 0. 设置随机种子以确保结果可复现
    torch.manual_seed(42)
    np.random.seed(42)

    datasets_to_process = [args.dataset] if args.dataset else ['ORSSD', 'EORSSD', 'ORS_4199']

    for current_dataset in datasets_to_process:
        print(f"\n{'='*20} Processing Dataset: {current_dataset} ({args.labeled_percent}) {'='*20}")
        
        model_path = './data/{}_{}_{}/best.pth'.format(args.model_name, current_dataset, args.labeled_percent)
        pseudo_path = './data/{}_{}_{}/pseudo_labels'.format(args.model_name, current_dataset, args.labeled_percent)
        labeled_list = '/home/liuyanfeng/RSI_SOD/{} dataset/train{}.txt'.format(current_dataset, args.labeled_percent)
        
        # 1. 加载配置和模型
        with open(args.config, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        
        from models import make
        model = make(config['model']).to(device)
        
        # 加载官方 Prompt 相关权重
        print(f"===> [{current_dataset}] Loading official Prompt weights from sam3.pt...")
        sam3_official = torch.load('/home/liuyanfeng/RSI_SOD/SAM3-Adapter/sam3.pt', map_location='cpu')
        official_map = {
            'pe_layer.positional_encoding_gaussian_matrix': 'tracker.sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix',
            'no_mask_embed.weight': 'tracker.sam_prompt_encoder.no_mask_embed.weight',
            'not_a_point_embed.weight': 'tracker.sam_prompt_encoder.not_a_point_embed.weight'
        }
        for i in range(4):
            official_map[f'point_embeddings.{i}.weight'] = f'tracker.sam_prompt_encoder.point_embeddings.{i}.weight'
        
        current_sd = model.state_dict()
        for target_key, official_key in official_map.items():
            if official_key in sam3_official:
                current_sd[target_key].copy_(sam3_official[official_key])
        model.load_state_dict(current_sd)

        # 加载业务模型权重
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        state_dict = {k: v for k, v in state_dict.items() if 'point_embeddings' not in k and 'pe_layer' not in k and 'no_mask_embed' not in k}
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded business weights from {model_path}")

        # 2. 加载未标注数据集
        unlabeled_dataset = SemiDataset(dataset_name=current_dataset, mode='train_u', size=args.img_size, labeled_percent=args.labeled_percent)
        # 动态绑定 pseudo_path 供 filter_reliable_samples 使用
        unlabeled_dataset.pseudo_path = pseudo_path
        dataloader = DataLoader(unlabeled_dataset, batch_size=1, shuffle=False)

        # 3. 执行筛选并合并
        filter_reliable_samples(
            model_name = args.model_name,
            model=model,
            dataloader=dataloader,
            device=device,
            output_txt='extended_train{}.txt'.format(args.labeled_percent),
            labeled_list_path=labeled_list
        )
