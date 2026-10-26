import os
import argparse
import torch
from tqdm import tqdm

from data import build_dataloaders
from models import USCDModel
from utils import (ChangeDetectionMetrics, save_prediction_comparison,
                   load_config, set_seed, load_checkpoint)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',     required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--split',      default='test')
    p.add_argument('--save_vis',   action='store_true')
    p.add_argument('--vis_dir',    default=None)
    p.add_argument('--device',     default='auto')
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, save_vis=False, vis_dir=None):
    model.eval()
    met = ChangeDetectionMetrics()
    for i, batch in enumerate(tqdm(loader, ncols=80)):
        if 'label' not in batch:
            continue
        img_A  = batch['img_A'].to(device)
        img_B  = batch['img_B'].to(device)
        labels = batch['label'].to(device)
        _, prob, _ = model.forward_student(img_A, img_B)
        pred = prob.argmax(1)
        met.update(pred, labels)
        if save_vis and vis_dir:
            unc = model.compute_uncertainty(prob)
            for j in range(img_A.shape[0]):
                fname = batch.get('filename', [f'{i}_{j}'])[j]
                save_prediction_comparison(img_A[j], img_B[j],
                                           labels[j], pred[j], unc[j],
                                           vis_dir, fname)
    return met.summary()


def main():
    args = parse_args()
    set_seed(42)
    cfg = load_config(args.config)
    device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu')
              if args.device == 'auto' else torch.device(args.device))

    _, _, val_loader, test_loader = build_dataloaders(cfg)
    loader = test_loader if args.split == 'test' else val_loader

    model = USCDModel(cfg).to(device)
    load_checkpoint(model, args.checkpoint, device=device)

    vis_dir = args.vis_dir or os.path.join(
        cfg['output']['save_dir'], f'eval_{args.split}')
    result = evaluate(model, loader, device, args.save_vis, vis_dir)

    print('\n' + '=' * 36)
    for k, v in result.items():
        print(f'  {k:<12} {v:.2f}%')
    print('=' * 36)


if __name__ == '__main__':
    main()
