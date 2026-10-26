import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def check_dataset_labels(data_root, split_file, dataset_type='loveda'):
    """检查数据集标签的实际类别"""

    # 读取split文件
    with open(split_file, 'r') as f:
        ids = [line.strip() for line in f.readlines()]

    all_labels = set()
    label_counts = {}

    print(f"检查 {len(ids)} 个样本的标签...")

    for img_id in tqdm(ids):
        # 根据你的数据集结构调整路径
        if dataset_type == 'loveda':
            mask_path = os.path.join(data_root, 'Train/Rural/masks_png', img_id + '.png')
            if not os.path.exists(mask_path):
                mask_path = os.path.join(data_root, 'Train/Urban/masks_png', img_id + '.png')

        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path))
            unique_vals = np.unique(mask)

            for val in unique_vals:
                all_labels.add(int(val))
                label_counts[int(val)] = label_counts.get(int(val), 0) + 1
        else:
            print(f"警告: 找不到 {mask_path}")

    print("\n" + "=" * 50)
    print("数据集标签统计:")
    print("=" * 50)
    print(f"所有唯一标签值: {sorted(all_labels)}")
    print(f"\n标签值分布:")
    for label in sorted(label_counts.keys()):
        print(f"  标签 {label}: 出现在 {label_counts[label]} 个像素中")

    # 检查是否有超出范围的标签
    expected_max = 6  # nclass=7, 所以是0-6
    problematic = [l for l in all_labels if l > expected_max and l != 255]

    if problematic:
        print(f"\n⚠️  发现问题: 以下标签值超出范围 [0, {expected_max}]: {problematic}")
    else:
        print(f"\n✓ 所有标签都在有效范围内 [0, {expected_max}] (255是ignore值)")

    return all_labels, label_counts


# 使用示例
data_root = 'LoveDA'
labeled_path = 'splits/loveda/1_100/labeled.txt'
unlabeled_path = 'splits/loveda/1_100/unlabeled.txt'

print("检查标注数据:")
check_dataset_labels(data_root, labeled_path)

print("\n\n检查未标注数据:")
check_dataset_labels(data_root, unlabeled_path)