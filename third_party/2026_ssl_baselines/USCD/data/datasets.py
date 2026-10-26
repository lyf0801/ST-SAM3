import os
import random
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from .transforms import TrainTransform, TestTransform


class ChangeDetectionDataset(Dataset):
    def __init__(self, data_root, split='train', img_size=256,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                 data_format='folder'):
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.data_format = data_format
        self.transform = (TrainTransform(img_size, mean, std)
                          if split in ('train', 'val')
                          else TestTransform(img_size, mean, std))
        self.samples = self._load_samples()

    def _load_samples(self):
        if self.data_format == 'folder':
            return self._from_folder()
        return self._from_txt()

    def _from_folder(self):
        A_dir = self.data_root / self.split / 'A'
        B_dir = self.data_root / self.split / 'B'
        label_dir = self.data_root / self.split / 'label'
        assert A_dir.exists(), f"Not found: {A_dir}"

        exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
        samples = []
        for fname in sorted(os.listdir(A_dir)):
            if Path(fname).suffix.lower() not in exts:
                continue
            stem = Path(fname).stem
            B_path = self._find(B_dir, stem, exts)
            label_path = self._find(label_dir, stem, exts) if label_dir.exists() else None
            if B_path:
                samples.append((str(A_dir / fname), str(B_path),
                                str(label_path) if label_path else None))
        assert len(samples) > 0, f"No samples in {A_dir}"
        return samples

    def _from_txt(self):
        txt = self.data_root / f'{self.split}.txt'
        assert txt.exists(), f"Not found: {txt}"
        samples = []
        with open(txt) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    A, B, L = parts
                elif len(parts) == 2:
                    A, B, L = *parts, None
                else:
                    continue
                if not os.path.isabs(A):
                    A = str(self.data_root / A)
                    B = str(self.data_root / B)
                    if L: L = str(self.data_root / L)
                samples.append((A, B, L))
        return samples

    def _find(self, directory, stem, extensions):
        for ext in extensions:
            p = directory / (stem + ext)
            if p.exists():
                return p
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        A_path, B_path, label_path = self.samples[idx]
        img_A = Image.open(A_path).convert('RGB')
        img_B = Image.open(B_path).convert('RGB')
        label = Image.open(label_path).convert('L') if label_path else None
        img_A, img_B, label = self.transform(img_A, img_B, label)
        sample = {'img_A': img_A, 'img_B': img_B,
                  'filename': Path(A_path).stem}
        if label is not None:
            sample['label'] = label
        return sample


class SemiSupervisedDataset(Dataset):
    def __init__(self, data_root, labeled_ratio=0.05, mode='labeled',
                 seed=42, img_size=256,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                 data_format='folder'):
        super().__init__()
        assert mode in ('labeled', 'unlabeled')
        self.mode = mode

        full = ChangeDetectionDataset(data_root, 'train', img_size,
                                      mean, std, data_format)
        total = len(full.samples)
        n_labeled = max(1, int(total * labeled_ratio))

        rng = random.Random(seed)
        idx = list(range(total))
        rng.shuffle(idx)
        chosen = idx[:n_labeled] if mode == 'labeled' else idx[n_labeled:]
        self.samples = [full.samples[i] for i in chosen]
        self.transform = TrainTransform(img_size, mean, std)
        print(f'[{mode}] {len(self.samples)} / {total} samples')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        A_path, B_path, label_path = self.samples[idx]
        img_A = Image.open(A_path).convert('RGB')
        img_B = Image.open(B_path).convert('RGB')
        label = (Image.open(label_path).convert('L')
                 if self.mode == 'labeled' and label_path else None)
        img_A, img_B, label = self.transform(img_A, img_B, label)
        sample = {'img_A': img_A, 'img_B': img_B,
                  'filename': Path(A_path).stem}
        if label is not None:
            sample['label'] = label
        return sample


def build_dataloaders(cfg):
    dc = cfg['data']
    kw = dict(
        data_root=dc['data_root'],
        img_size=dc.get('img_size', 256),
        mean=dc.get('mean', (0.485, 0.456, 0.406)),
        std=dc.get('std', (0.229, 0.224, 0.225)),
        data_format=dc.get('data_format', 'folder'),
    )
    bs = cfg['train']['batch_size']
    nw = dc.get('num_workers', 4)
    lr = dc.get('labeled_ratio', 0.05)

    labeled   = SemiSupervisedDataset(**kw, labeled_ratio=lr, mode='labeled')
    unlabeled = SemiSupervisedDataset(**kw, labeled_ratio=lr, mode='unlabeled')
    val       = ChangeDetectionDataset(**kw, split='val')
    test      = ChangeDetectionDataset(**kw, split='test')

    def make_loader(ds, shuffle, drop_last=False):
        return DataLoader(ds, batch_size=bs, shuffle=shuffle,
                         num_workers=nw, pin_memory=True, drop_last=drop_last)

    return (make_loader(labeled, True, True),
            make_loader(unlabeled, True, True),
            make_loader(val, False),
            make_loader(test, False))
