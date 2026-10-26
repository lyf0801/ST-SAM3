import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image


def denormalize(t, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    img = (t.cpu() * s + m).permute(1, 2, 0).numpy()
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def save_prediction_comparison(img_A, img_B, gt, pred, unc, save_path, filename):
    os.makedirs(save_path, exist_ok=True)
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    axes[0].imshow(denormalize(img_A));  axes[0].set_title('T1');    axes[0].axis('off')
    axes[1].imshow(denormalize(img_B));  axes[1].set_title('T2');    axes[1].axis('off')
    axes[2].imshow(gt.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('GT'); axes[2].axis('off')

    gt_b   = gt.cpu().numpy().astype(bool)
    pred_b = pred.cpu().numpy().astype(bool)
    rgb = np.zeros((*pred_b.shape, 3), dtype=np.uint8)
    rgb[ gt_b &  pred_b] = [0, 255, 0]
    rgb[~gt_b &  pred_b] = [255, 0, 0]
    rgb[ gt_b & ~pred_b] = [0, 0, 255]
    axes[3].imshow(rgb)
    axes[3].set_title('Pred (G=TP R=FP B=FN)'); axes[3].axis('off')

    im = axes[4].imshow(unc.cpu().numpy(), cmap='hot', vmin=0, vmax=1)
    axes[4].set_title('Uncertainty'); axes[4].axis('off')
    plt.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{filename}.png'), dpi=150, bbox_inches='tight')
    plt.close()


def save_uncertainty_map(unc, save_path, filename):
    os.makedirs(save_path, exist_ok=True)
    colored = (cm.hot(unc.cpu().numpy())[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(colored).save(
        os.path.join(save_path, f'{filename}_uncertainty.png'))


def plot_training_curves(history, save_path):
    os.makedirs(save_path, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, title, color in [
        (axes[0], 'train_loss', 'Loss',     'steelblue'),
        (axes[1], 'val_f1',    'F1 (%)',    'green'),
        (axes[2], 'val_iou',   'IoU (%)',   'orange'),
    ]:
        if key in history:
            ax.plot(history[key], color=color)
            ax.set_title(title); ax.set_xlabel('Epoch'); ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_curves.png'), dpi=150)
    plt.close()
