
# ST-SAM3 and DistillMatch

## 视觉基础模型驱动的低标注遥感显著目标检测 (Vision Foundation Model-Guided Learning for Label-Efficient Remote Sensing Salient Object Detection)

刘焰锋<sup>1, 2</sup>，王迪<sup>1, 2</sup>，郭昊南<sup>3, 2</sup>，张敬<sup>1, 2</sup>，张乐飞<sup>1, *</sup>，张良培<sup>3, 4</sup>

<sup>1</sup> 武汉大学计算机学院，武汉 430072；    <sup>2</sup> 北京中关村学院，北京 100094；  
<sup>3</sup> 武汉大学测绘遥感信息工程全国重点实验室，武汉 430079；   <sup>4</sup> 河南省科学院空天信息研究所，郑州 450046

中国图象图形学报. 2026.

论文地址：

DOI：

## Environment / 运行环境

The experiments used an NVIDIA A100 GPU, Python 3.10, PyTorch 2.3.1+cu121 and torchvision 0.18.1. Install the required dependencies, including NumPy, Pillow, PyYAML, tqdm, OpenCV, timm, einops, iopath, huggingface-hub, safetensors, SciPy, matplotlib, ptflops and mmcv. Prepare the official SAM3 checkpoint and the DINOv3/MobileSAM initialization checkpoints before training.

本论文实验使用 NVIDIA A100 GPU，运行环境为 Python 3.10、PyTorch 2.3.1+cu121 和 torchvision 0.18.1。请安装 NumPy、Pillow、PyYAML、tqdm、OpenCV、timm、einops、iopath、huggingface-hub、safetensors、SciPy、matplotlib、ptflops 和 mmcv 等依赖，并在训练前准备官方 SAM3 权重及 DINOv3/MobileSAM 初始化权重。

Update the dataset paths in the stage scripts and dataset loaders, the `sam_checkpoint` path in [configs/cod-sam-vit-l.yaml](configs/cod-sam-vit-l.yaml), and the official Prompt checkpoint paths in stage2/4/6 for your machine. Foundation-model initialization files are placed in [pretrained/](pretrained/) or at the configured checkpoint paths.

请根据机器环境修改各 stage 脚本和数据加载器中的数据集路径、[configs/cod-sam-vit-l.yaml](configs/cod-sam-vit-l.yaml) 的 `sam_checkpoint` 路径，以及 stage2/4/6 中的官方 Prompt 权重路径。基础模型初始化权重放在 [pretrained/](pretrained/) 或配置指定的位置。

## Dataset splits / 数据划分

Prepare each dataset with `images/`, `labels/`, `edges/`, `train.txt` and `test.txt`. [splits/](splits/) contains five initial labeled splits for ORSSD, EORSSD and ORSI-4199 at both 5% and 10% ratios: the unsuffixed files denote v1, used for the main experiments in the paper; `_v2`–`_v5` are used to evaluate stability across random splits. ORSI-4199 uses the identifier `ORS_4199` in code.

每个数据集需要准备 `images/`、`labels/`、`edges/`、`train.txt` 和 `test.txt`。[splits/](splits/) 包含 ORSSD、EORSSD、ORSI-4199 在 5% 和 10% 比例下的五组初始标注划分：无后缀文件对应正文主要实验采用的划分版本 v1，`_v2`–`_v5` 用于随机划分稳定性实验。ORSI-4199 在代码中记为 `ORS_4199`。

The six 75% supervision subsets are provided in `data/Re_Re_Retrain_SAM3_Adapter_DATASET_RATIO/`, including `extended_train5%.txt` or `extended_train10%.txt` and `extended_labeled_mask/`. The masks combine initial ground-truth labels with selected pseudo-labels. Each subset contains 450 masks for ORSSD, 1050 for EORSSD, or 1500 for ORSI-4199. These subsets are for v1; regenerate them when using v2–v5.

六组 75% 监督子集放在 `data/Re_Re_Retrain_SAM3_Adapter_数据集_比例/`，包含 `extended_train5%.txt` 或 `extended_train10%.txt` 及 `extended_labeled_mask/`。掩码由初始真实标注和筛选伪标签共同组成，每组 ORSSD 为 450 张、EORSSD 为 1050 张、ORSI-4199 为 1500 张。这些子集对应 v1，使用 v2–v5 时应重新生成。




## Training / 训练

To train DistillMatch, use the supplied supervision subset and teacher checkpoint and run stage9:

如果想训练 DistillMatch，使用提供的监督子集和教师权重，执行 stage9 即可：

```bash
bash scripts/train_DistillMatch_ORSSD_5%.sh
```

To train ST-SAM3, run stage1–stage7:

如果想训练 ST-SAM3，执行 stage1–stage7：

```bash
bash scripts/train_ST_SAM3_ORSSD_5%.sh
```

Choose either training command as needed. Activate your own Python environment and run from the repository root. For other settings, replace `ORSSD` with `EORSSD` or `ORS_4199`, and `5%` with `10%` in the script name.

可按需选择任一种方法进行训练。请先激活自己配置的 Python 环境，再在仓库根目录执行。其他设置只需将脚本名中的 `ORSSD` 替换为 `EORSSD` 或 `ORS_4199`，将 `5%` 替换为 `10%`。

## Testing / 测试

### Evaluate provided saliency maps / 直接评估显著图

**Download saliency maps: [predicted_saliency_maps/](predicted_saliency_maps/)** — results for all three datasets at both 5% and 10% labeled ratios.

Extract the `predict_smaps_*` folders into `data/output/` and evaluate them with [utils/compute_metrics.py](utils/compute_metrics.py). The metrics are listed below.

**预测显著图下载：[predicted_saliency_maps/](predicted_saliency_maps/)**，包含三个数据集在 **5% 和 10%** 标注比例下的结果。

将其中的 `predict_smaps_*` 文件夹解压到 `data/output/` 后，可使用 [utils/compute_metrics.py](utils/compute_metrics.py) 计算指标。指标如下：

| Method / 方法 | Labeled ratio / 标注比例 | Dataset / 数据集 | MAE ↓ | Max Fβ ↑ | S-measure ↑ |
| --- | --- | --- | ---: | ---: | ---: |
| ST-SAM3 | 5% | ORSSD | 0.0133 | 0.9265 | 0.9340 |
| ST-SAM3 | 5% | EORSSD | 0.0091 | 0.8943 | 0.9254 |
| ST-SAM3 | 5% | ORSI-4199 | 0.0252 | 0.8819 | 0.8889 |
| ST-SAM3 | 10% | ORSSD | 0.0120 | 0.9328 | 0.9379 |
| ST-SAM3 | 10% | EORSSD | 0.0073 | 0.9065 | 0.9348 |
| ST-SAM3 | 10% | ORSI-4199 | 0.0253 | 0.8831 | 0.8906 |
| DistillMatch | 5% | ORSSD | 0.0134 | 0.9233 | 0.9283 |
| DistillMatch | 5% | EORSSD | 0.0093 | 0.8890 | 0.9222 |
| DistillMatch | 5% | ORSI-4199 | 0.0267 | 0.8779 | 0.8857 |
| DistillMatch | 10% | ORSSD | 0.0129 | 0.9211 | 0.9303 |
| DistillMatch | 10% | EORSSD | 0.0078 | 0.8964 | 0.9276 |
| DistillMatch | 10% | ORSI-4199 | 0.0257 | 0.8812 | 0.8883 |

```bash
python utils/compute_metrics.py
```

### Run inference with checkpoints / 使用权重推理

Download the 12 checkpoints from Baidu Netdisk and place the `.pth` files directly in [weights/](weights/), preserving the filenames below.

从百度网盘下载 12 个权重，将 `.pth` 文件直接放入 [weights/](weights/)，保持下表文件名。

**Download / 下载：[Baidu Netdisk / 百度网盘](https://pan.baidu.com/s/1xZlXI3CCeUvb1rrNaQzd3w?pwd=1999)**  
**Extraction code / 提取码：`1999`**

| Dataset / 数据集 | Ratio / 标注比例 | ST-SAM3 | DistillMatch |
| --- | --- | --- | --- |
| ORSSD | 5% | `ST-SAM3_ORSSD_5%.pth` | `DistillMatch_ORSSD_5%.pth` |
| ORSSD | 10% | `ST-SAM3_ORSSD_10%.pth` | `DistillMatch_ORSSD_10%.pth` |
| EORSSD | 5% | `ST-SAM3_EORSSD_5%.pth` | `DistillMatch_EORSSD_5%.pth` |
| EORSSD | 10% | `ST-SAM3_EORSSD_10%.pth` | `DistillMatch_EORSSD_10%.pth` |
| ORSI-4199 | 5% | `ST-SAM3_ORS_4199_5%.pth` | `DistillMatch_ORS_4199_5%.pth` |
| ORSI-4199 | 10% | `ST-SAM3_ORS_4199_10%.pth` | `DistillMatch_ORS_4199_10%.pth` |

Export saliency maps and then compute MAE, maximum F-measure and S-measure. The export scripts default to 5%; to evaluate 10%, update both the checkpoint selection and the output directory/file naming. In [compute_metrics.py](utils/compute_metrics.py), select only methods and ratios whose saliency maps have been exported.

先导出显著图，再计算 MAE、最大 F 分数和 S-measure。导出脚本默认选择 5%；测试 10% 时，请同时修改权重选择和输出目录、文件命名。在 [compute_metrics.py](utils/compute_metrics.py) 中仅选择已经生成显著图的方法及比例。

```bash
python getsmaps_SAM3_Adapter.py
python getsmaps_DistillMatch.py
python utils/compute_metrics.py
```

## Citation / 论文引用

If you use this project in your research, please consider citing our paper. Citation details will be added when available.

如果本项目对您的研究有所帮助，欢迎引用我们的论文。引用信息将在正式公布后补充。

```bibtex
```
