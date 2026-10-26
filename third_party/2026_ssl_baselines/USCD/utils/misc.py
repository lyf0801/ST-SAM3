import os
import yaml
import torch
import random
import logging
import numpy as np


def load_config(config_path, base_path='configs/base.yaml'):
    base = {}
    if os.path.exists(base_path):
        with open(base_path) as f:
            base = yaml.safe_load(f) or {}
    with open(config_path) as f:
        custom = yaml.safe_load(f) or {}
    return _deep_merge(base, custom)


def _deep_merge(base, override):
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(log_dir, name='uscd'):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter('[%(asctime)s] %(message)s', '%H:%M:%S')
        for h in [logging.StreamHandler(),
                  logging.FileHandler(os.path.join(log_dir, f'{name}.log'))]:
            h.setFormatter(fmt)
            logger.addHandler(h)
    return logger


def save_checkpoint(model, optimizer, epoch, metrics, save_dir,
                    filename=None, is_best=False):
    os.makedirs(save_dir, exist_ok=True)
    state = {
        'epoch': epoch,
        'student': model.student.state_dict(),
        'teacher': model.teacher.state_dict(),
        'optimizer': optimizer.state_dict(),
        'metrics': metrics,
    }
    path = os.path.join(save_dir, filename or f'ckpt_ep{epoch:03d}.pth')
    torch.save(state, path)
    if is_best:
        torch.save(state, os.path.join(save_dir, 'best_model.pth'))
        print(f'  best saved → F1={metrics.get("F1", 0):.2f}%')


def load_checkpoint(model, path, optimizer=None, device='cpu'):
    ckpt = torch.load(path, map_location=device)
    model.student.load_state_dict(ckpt['student'])
    model.teacher.load_state_dict(ckpt['teacher'])
    if optimizer and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    print(f'  loaded checkpoint epoch {ckpt.get("epoch", "?")}')
    return ckpt.get('epoch', 0), ckpt.get('metrics', {})


def get_lr_scheduler(optimizer, cfg):
    total = cfg['train']['epochs']
    warmup = cfg['train'].get('warmup_epochs', 0)
    lr_min = cfg['train'].get('lr_min', 1e-4)
    lr_max = cfg['train'].get('lr', 0.01)

    def fn(epoch):
        if epoch < warmup:
            return 1.0
        p = (epoch - warmup) / max(total - warmup, 1)
        return max(lr_min / lr_max, 1.0 - p)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)
