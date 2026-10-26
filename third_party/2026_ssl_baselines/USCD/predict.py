import os
import glob
import argparse
import numpy as np
import torch
from PIL import Image

from models import USCDModel
from data.transforms import TestTransform
from utils import load_config, load_checkpoint
from utils.visualization import save_uncertainty_map


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',     required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--img_A',  default=None)
    p.add_argument('--img_B',  default=None)
    p.add_argument('--dir_A',  default=None)
    p.add_argument('--dir_B',  default=None)
    p.add_argument('--output_dir', default='./results')
    p.add_argument('--save_uncertainty', action='store_true')
    p.add_argument('--device', default='auto')
    return p.parse_args()


@torch.no_grad()
def predict_single(model, path_A, path_B, transform, device,
                   out_dir, save_unc=False):
    img_A = Image.open(path_A).convert('RGB')
    img_B = Image.open(path_B).convert('RGB')
    tA, tB, _ = transform(img_A, img_B, None)
    tA, tB = tA.unsqueeze(0).to(device), tB.unsqueeze(0).to(device)

    _, prob, _ = model.forward_student(tA, tB)
    pred = prob.argmax(1)[0]
    unc  = model.compute_uncertainty(prob)[0]

    stem = os.path.splitext(os.path.basename(path_A))[0]
    os.makedirs(out_dir, exist_ok=True)
    Image.fromarray((pred.cpu().numpy() * 255).astype(np.uint8)).save(
        os.path.join(out_dir, f'{stem}_change.png'))

    if save_unc:
        save_uncertainty_map(unc, out_dir, stem)

    print(f'  {stem}: change={pred.float().mean().item()*100:.1f}%')


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu')
              if args.device == 'auto' else torch.device(args.device))

    model = USCDModel(cfg).to(device)
    load_checkpoint(model, args.checkpoint, device=device)
    model.eval()

    dc = cfg['data']
    transform = TestTransform(dc.get('img_size', 256),
                               dc.get('mean', (0.485, 0.456, 0.406)),
                               dc.get('std', (0.229, 0.224, 0.225)))

    if args.img_A and args.img_B:
        predict_single(model, args.img_A, args.img_B, transform,
                       device, args.output_dir, args.save_uncertainty)

    elif args.dir_A and args.dir_B:
        exts = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']
        files = sorted(sum([glob.glob(os.path.join(args.dir_A, e)) for e in exts], []))
        print(f'{len(files)} pairs found')
        for pA in files:
            stem = os.path.splitext(os.path.basename(pA))[0]
            pB = next((os.path.join(args.dir_B, stem + e)
                       for e in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
                       if os.path.exists(os.path.join(args.dir_B, stem + e))), None)
            if pB is None:
                print(f'  skip {stem} (no B match)')
                continue
            predict_single(model, pA, pB, transform,
                           device, args.output_dir, args.save_uncertainty)
    else:
        print('provide --img_A + --img_B  or  --dir_A + --dir_B')

    print(f'\nsaved to {args.output_dir}')


if __name__ == '__main__':
    main()
