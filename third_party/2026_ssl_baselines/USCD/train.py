import os
import argparse
import itertools
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import build_dataloaders
from models import USCDModel
from modules import UAPA, DRCL, UGLR
from utils import (ChangeDetectionMetrics, save_prediction_comparison,
                   plot_training_curves, load_config, set_seed,
                   setup_logger, save_checkpoint, load_checkpoint,
                   get_lr_scheduler)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/custom.yaml')
    p.add_argument('--labeled_ratio', type=float, default=None)
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--device', default='auto')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def get_device(s):
    if s == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(s)


def train_epoch(model, labeled_loader, unlabeled_loader,
                optimizer, uapa, drcl, uglr,
                epoch, total_epochs, cfg, device):
    model.train()
    warmup  = cfg['train'].get('warmup_epochs', 30)
    cw      = cfg['drcl'].get('contrast_loss_weight', 0.1)
    thresh  = cfg['pseudo_label'].get('confidence_threshold', 0.9)

    unlabeled_iter = itertools.cycle(unlabeled_loader)
    total, n = 0.0, 0

    pbar = tqdm(labeled_loader, desc=f'Ep {epoch}/{total_epochs}', ncols=100)
    for batch_L in pbar:
        img_A_L = batch_L['img_A'].to(device)
        img_B_L = batch_L['img_B'].to(device)
        labels  = batch_L['label'].to(device)

        _, prob_L_t, _ = model.forward_teacher(img_A_L, img_B_L)
        unc_L = model.compute_uncertainty(prob_L_t)
        logits_L_s, _, feat_L_s = model.forward_student(img_A_L, img_B_L)
        l_sup, _ = uglr(logits_L_s, labels, unc_L, logits_L_s, labels, unc_L)

        if epoch <= warmup:
            loss = l_sup
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            model.update_teacher()
            total += loss.item(); n += 1
            pbar.set_postfix(loss=f'{loss.item():.4f}', phase='warmup')
            continue

        batch_U = next(unlabeled_iter)
        img_A_U = batch_U['img_A'].to(device)
        img_B_U = batch_U['img_B'].to(device)

        with torch.no_grad():
            _, prob_U_t, _ = model.forward_teacher(img_A_U, img_B_U)
            unc_U = model.compute_uncertainty(prob_U_t)
            pseudo, confident = model.generate_pseudo_labels(prob_U_t, thresh)

        B = min(img_A_U.shape[0], img_A_L.shape[0])
        if B > 1:
            mix_A, mix_B, mix_L = uapa(
                img_A_U[:B], img_B_U[:B], pseudo[:B],
                img_A_U[:B].roll(1, 0), img_B_U[:B].roll(1, 0), pseudo[:B].roll(1, 0),
                unc_U[:B], prob_U_t[:B, 1], epoch, total_epochs)
        else:
            mix_A, mix_B, mix_L = img_A_U[:B], img_B_U[:B], pseudo[:B]

        with torch.no_grad():
            _, prob_aug_t, _ = model.forward_teacher(mix_A, mix_B)

        logits_U_s, _, _ = model.forward_student(mix_A, mix_B)
        unc_mix = model.compute_uncertainty(prob_aug_t)

        _, l_unsup = uglr(logits_L_s, labels, unc_L,
                          logits_U_s, mix_L, unc_mix,
                          confident[:B] if B <= confident.shape[0] else None)

        l_contrast, _, _ = drcl(feat_L_s, labels, prob_L_t,
                                 prob_aug_t[:labels.shape[0]] if prob_aug_t.shape[0] >= labels.shape[0] else prob_L_t,
                                 unc_L)

        loss = l_sup + l_unsup + cw * l_contrast
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.student.parameters(), 1.0)
        optimizer.step()
        model.update_teacher()

        total += loss.item(); n += 1
        pbar.set_postfix(loss=f'{loss.item():.4f}',
                         sup=f'{l_sup.item():.3f}',
                         uns=f'{l_unsup.item():.3f}',
                         ctr=f'{l_contrast.item():.3f}')

    return total / max(n, 1)


@torch.no_grad()
def validate(model, loader, device, cfg, save_vis=False, vis_dir=None):
    model.eval()
    met = ChangeDetectionMetrics()

    for i, batch in enumerate(loader):
        if 'label' not in batch:
            continue
        img_A  = batch['img_A'].to(device)
        img_B  = batch['img_B'].to(device)
        labels = batch['label'].to(device)

        _, prob, _ = model.forward_student(img_A, img_B)
        pred = prob.argmax(1)
        met.update(pred, labels)

        if save_vis and vis_dir and i < 3:
            unc = model.compute_uncertainty(prob)
            for j in range(min(2, img_A.shape[0])):
                fname = batch.get('filename', [f'{i}_{j}'])[j]
                save_prediction_comparison(img_A[j], img_B[j],
                                           labels[j], pred[j], unc[j],
                                           vis_dir, fname)
    return met.summary()


def main():
    args = parse_args()
    set_seed(args.seed)

    cfg = load_config(args.config)
    if args.labeled_ratio is not None:
        cfg['data']['labeled_ratio'] = args.labeled_ratio

    device = get_device(args.device)
    print(f'device: {device}')
    if device.type == 'cuda':
        print(f'gpu: {torch.cuda.get_device_name(0)}')

    save_dir = cfg['output']['save_dir']
    log_dir  = cfg['output']['log_dir']
    os.makedirs(save_dir, exist_ok=True)

    logger = setup_logger(log_dir)
    writer = SummaryWriter(log_dir)
    logger.info(f'config={args.config}  ratio={cfg["data"]["labeled_ratio"]:.0%}')

    labeled_loader, unlabeled_loader, val_loader, test_loader = build_dataloaders(cfg)

    model = USCDModel(cfg).to(device)
    logger.info(f'params: {sum(p.numel() for p in model.student.parameters())/1e6:.1f}M')

    optimizer = torch.optim.SGD(
        model.student.parameters(),
        lr=cfg['train']['lr'],
        momentum=cfg['train'].get('momentum', 0.9),
        weight_decay=cfg['train'].get('weight_decay', 1e-4))
    scheduler = get_lr_scheduler(optimizer, cfg)

    uc = cfg['uapa']
    uapa = UAPA(uc.get('window_size', 16), uc.get('beta', 0.3),
                uc.get('paste_ratio_max', 0.5), uc.get('paste_ratio_min', 0.1))

    dc = cfg['drcl']
    drcl = DRCL(256, dc.get('num_anchors', 32), dc.get('num_samples', 64),
                dc.get('temperature', 0.1), dc.get('memory_bank_size', 512),
                dc.get('global_loss_weight', 0.5)).to(device)

    uglr = UGLR(cfg['uglr'].get('gamma_labeled', 2.0),
                cfg['uglr'].get('gamma_unlabeled', -1.0))

    start_epoch = 1
    best_f1 = 0.0
    history = {'train_loss': [], 'val_f1': [], 'val_iou': []}

    if args.resume:
        start_epoch, saved = load_checkpoint(model, args.resume, optimizer, device)
        start_epoch += 1
        best_f1 = saved.get('F1', 0.0)

    total_epochs  = cfg['train']['epochs']
    eval_interval = cfg['output'].get('eval_interval', 5)
    save_interval = cfg['output'].get('save_interval', 10)
    do_vis        = cfg['output'].get('visualize', True)

    for epoch in range(start_epoch, total_epochs + 1):
        avg_loss = train_epoch(model, labeled_loader, unlabeled_loader,
                               optimizer, uapa, drcl, uglr,
                               epoch, total_epochs, cfg, device)
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(avg_loss)
        writer.add_scalar('loss/train', avg_loss, epoch)
        logger.info(f'ep {epoch:3d} | loss={avg_loss:.4f} | lr={lr:.6f}')

        if epoch % eval_interval == 0 or epoch == total_epochs:
            vis_dir = os.path.join(save_dir, 'vis', f'ep{epoch:03d}') if do_vis else None
            m = validate(model, val_loader, device, cfg, do_vis, vis_dir)
            history['val_f1'].append(m['F1'])
            history['val_iou'].append(m['IoU'])
            writer.add_scalars('metrics', m, epoch)

            is_best = m['F1'] > best_f1
            if is_best: best_f1 = m['F1']
            logger.info(f'  val F1={m["F1"]:.2f}  IoU={m["IoU"]:.2f}'
                        + ('  ★' if is_best else ''))
            save_checkpoint(model, optimizer, epoch, m, save_dir, is_best=is_best)

        if epoch % save_interval == 0:
            save_checkpoint(model, optimizer, epoch, {'F1': best_f1},
                            save_dir, f'ckpt_ep{epoch:03d}.pth')

    best_path = os.path.join(save_dir, 'best_model.pth')
    if os.path.exists(best_path):
        load_checkpoint(model, best_path, device=device)

    m = validate(model, test_loader, device, cfg, do_vis,
                 os.path.join(save_dir, 'test_vis'))
    logger.info(f'test  F1={m["F1"]:.2f}  IoU={m["IoU"]:.2f}  '
                f'P={m["Precision"]:.2f}  R={m["Recall"]:.2f}')

    plot_training_curves(history, log_dir)
    writer.close()


if __name__ == '__main__':
    main()
