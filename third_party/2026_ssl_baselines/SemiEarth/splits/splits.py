import os
import random
import math

random.seed(42)

def split_and_save():
    input_file = 'all.txt'

    if not os.path.exists(input_file):
        print(f"错误：未找到文件 {input_file}，请确保它与脚本在同一目录下。")
        return

    print(f"正在读取 {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        # 过滤空行并去重
        lines = list(set([line.strip() for line in f if line.strip()]))

    total_count = len(lines)
    print(f"共读取到 {total_count} 条唯一数据。")

    # 打乱数据
    lines.sort()  # 先排序保证顺序确定性
    random.shuffle(lines)

    n_train = int(total_count * 0.6)
    n_val = int(total_count * 0.2)

    train_data = lines[:n_train]
    val_data = lines[n_train: n_train + n_val]
    test_data = lines[n_train + n_val:]

    print(f"划分结果: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

    save_file('train.txt', train_data)
    save_file('val.txt', val_data)
    save_file('test.txt', test_data)

    percentages = [0.01, 0.05, 0.10]

    print("\n正在生成半监督划分文件...")
    for p in percentages:
        pct_str = str(int(p * 100))  # '1', '5', '10'

        n_labeled = int(len(train_data) * p)

        labeled_subset = train_data[:n_labeled]
        unlabeled_subset = train_data[n_labeled:]

        label_fname = f'labeled_{pct_str}.txt'
        unlabel_fname = f'unlabeled_{pct_str}.txt'

        save_file(label_fname, labeled_subset)
        save_file(unlabel_fname, unlabeled_subset)
        print(f"  - [{pct_str}%] Labeled: {len(labeled_subset)}, Unlabeled: {len(unlabeled_subset)}")

    print("\n所有文件已生成完毕！")

def save_file(filename, data):

    with open(filename, 'w', encoding='utf-8') as f:
        for line in data:
            f.write(line + '\n')


if __name__ == '__main__':
    split_and_save()