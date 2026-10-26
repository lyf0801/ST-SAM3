import argparse
import os
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
from dataset.semi import SemiDataset

try:
    from model.semseg.dpt import DPT
except ImportError:
    print("Error: model.semseg.dpt not found. Please ensure your path is correct.")
    exit(1)

def get_parser():
    parser = argparse.ArgumentParser(description='SemiEarth Testing')
    parser.add_argument('--config', type=str, required=True, help='config file path')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to best.pth')
    parser.add_argument('--save-path', type=str, default=None, help='path to save visual results')
    return parser

def intersectionAndUnion(output, target, K, ignore_index=255):
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size).copy()

    output[np.where(target == ignore_index)[0]] = 255
    intersection = output[np.where(output == target)[0]]

    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))

    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

def test(args):

    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    val_dataset = SemiDataset(
        name=cfg['dataset'],
        root=cfg['data_root'],
        mode='val',
        size=cfg['crop_size']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    print(f"Dataset: {cfg['dataset']}, Val images: {len(val_dataset)}")

    raw_backbone = cfg['backbone']

    features = 128
    out_channels = [96, 192, 384, 768]
    encoder_size = raw_backbone

    if 'small' in raw_backbone:
        encoder_size = 'small'
        features = 64
        out_channels = [48, 96, 192, 384]
        print("Detected Small backbone: reducing head channels (features=64).")

    elif 'base' in raw_backbone:
        encoder_size = 'base'
        features = 128
        out_channels = [96, 192, 384, 768]

    elif 'large' in raw_backbone:
        encoder_size = 'large'
        features = 256
        out_channels = [256, 512, 1024, 1024]

    elif 'giant' in raw_backbone:
        encoder_size = 'giant'

    print(f"Initializing DPT -> Size: {encoder_size}, Features: {features}, OutCh: {out_channels}")

    n_classes = cfg.get('nclass', cfg.get('num_classes'))
    model = DPT(encoder_size, n_classes, features=features, out_channels=out_channels)

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    try:
        model.load_state_dict(new_state_dict, strict=True)
        print("Successfully loaded model weights (strict=True).")
    except Exception as e:
        print(f"Strict load failed: {e}")
        print("Trying strict=False...")
        model.load_state_dict(new_state_dict, strict=False)

    model.cuda()
    model.eval()

    intersection_meter = 0
    union_meter = 0

    print("Start evaluating...")
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    with torch.no_grad():
        for i, (img, mask, filename) in enumerate(tqdm(val_loader)):
            img = img.cuda()
            mask = mask.cuda()

            h, w = img.shape[2], img.shape[3]
            patch_size = 14

            new_h = int(math.ceil(h / patch_size) * patch_size)
            new_w = int(math.ceil(w / patch_size) * patch_size)

            if new_h != h or new_w != w:
                img = F.interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=True)

            pred = model(img)

            if isinstance(pred, tuple):
                pred = pred[0]

            target_h, target_w = mask.shape[1], mask.shape[2]

            if pred.shape[2:] != (target_h, target_w):
                pred = F.interpolate(pred, size=(target_h, target_w), mode='bilinear', align_corners=True)

            pred = torch.argmax(pred, dim=1)

            pred_np = pred.cpu().numpy()
            mask_np = mask.cpu().numpy()

            if args.save_path:
                from PIL import Image
                name = filename[0] if isinstance(filename, (list, tuple)) else f"{i}"
                name = os.path.splitext(os.path.basename(name))[0]
                pred_img = Image.fromarray(pred_np[0].astype(np.uint8))
                pred_img.save(os.path.join(args.save_path, name + ".png"))

            intersection, union, target = intersectionAndUnion(
                pred_np, mask_np, n_classes, ignore_index=255
            )

            intersection_meter += intersection
            union_meter += union

    iou_class = intersection_meter / (union_meter + 1e-10)
    mIoU = np.mean(iou_class)

    print("------------------------------------------------")
    for i, iou in enumerate(iou_class):
        print(f"Class {i}: {iou * 100:.2f}")
    print(f"mIoU: {mIoU * 100:.2f}")
    print("------------------------------------------------")

if __name__ == '__main__':
    args = get_parser().parse_args()
    test(args)