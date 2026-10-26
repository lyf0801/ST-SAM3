
import argparse
import time
import logging

import torch
import yaml

from model.semseg.dpt import DPT
from util.utils import init_log

try:
    from thop import profile as thop_profile, clever_format
    HAS_THOP = True
except ImportError:
    HAS_THOP = False

try:
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    HAS_FVCORE = True
except ImportError:
    HAS_FVCORE = False

parser = argparse.ArgumentParser(description='Model FLOPs & FPS Profiler')
parser.add_argument('--config',      type=str, required=True,  help='训练配置文件路径')
parser.add_argument('--checkpoint',  type=str, default=None,   help='模型权重路径（可选）')
parser.add_argument('--input-size',  type=int, nargs=2,
                    default=[512, 512], metavar=('H', 'W'),    help='单张输入图像尺寸')
parser.add_argument('--batch-size',  type=int, default=1,      help='FPS 测试时的 batch size')
parser.add_argument('--warmup',      type=int, default=50,     help='GPU 预热轮次')
parser.add_argument('--runs',        type=int, default=200,    help='正式计时轮次')
parser.add_argument('--device',      type=str, default='cuda', help='cuda / cpu')


def build_model(cfg: dict) -> torch.nn.Module:
    model_configs = {
        'small': {'encoder_size': 'small', 'features': 64,
                  'out_channels': [48, 96, 192, 384]},
        'base':  {'encoder_size': 'base',  'features': 128,
                  'out_channels': [96, 192, 384, 768]},
        'large': {'encoder_size': 'large', 'features': 256,
                  'out_channels': [256, 512, 1024, 1024]},
        'giant': {'encoder_size': 'giant', 'features': 384,
                  'out_channels': [1536, 1536, 1536, 1536]},
    }
    size_key = cfg['backbone'].split('_')[-1]
    model = DPT(
        **{**model_configs[size_key], 'nclass': cfg['nclass']}
    )
    return model


def load_checkpoint(model: torch.nn.Module, ckpt_path: str,
                    device: torch.device) -> None:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get('model', ckpt)
    new_state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(new_state, strict=False)


def measure_flops(model: torch.nn.Module,
                  dummy: torch.Tensor,
                  logger: logging.Logger) -> None:
    if HAS_THOP:
        import copy
        model_copy = copy.deepcopy(model)
        macs, params = thop_profile(model_copy, inputs=(dummy,), verbose=False)
        del model_copy
        macs_str, params_str = clever_format([macs, params], '%.3f')
        gflops = macs * 2 / 1e9
        logger.info('─' * 55)
        logger.info(f'  [thop]  Params : {params_str}')
        logger.info(f'  [thop]  MACs   : {macs_str}')
        logger.info(f'  [thop]  GFLOPs : {gflops:.2f} G  (MACs × 2)')
    elif HAS_FVCORE:
        flops = FlopCountAnalysis(model, dummy)
        flops.unsupported_ops_settings(
            raise_on_error=False, warn_on_error=False
        )
        gflops = flops.total() / 1e9
        params  = sum(p.numel() for p in model.parameters()) / 1e6
        logger.info('─' * 55)
        logger.info(f'  [fvcore] Params : {params:.2f} M')
        logger.info(f'  [fvcore] GFLOPs : {gflops:.2f} G')
        logger.info('\n' + flop_count_table(flops, max_depth=3))
    else:
        logger.warning(
            '未检测到 thop 或 fvcore，跳过 FLOPs 测量。\n'
            '安装方式: pip install thop  或  pip install fvcore'
        )


@torch.no_grad()
def measure_fps(model: torch.nn.Module,
                dummy: torch.Tensor,
                warmup: int,
                runs: int,
                device: torch.device,
                logger: logging.Logger) -> None:
    use_cuda = device.type == 'cuda'

    logger.info(f'  GPU 预热中 ({warmup} 轮)...')
    for _ in range(warmup):
        _ = model(dummy)
        if use_cuda:
            torch.cuda.synchronize()

    if use_cuda:
        starter = torch.cuda.Event(enable_timing=True)
        ender   = torch.cuda.Event(enable_timing=True)
        starter.record()
        for _ in range(runs):
            _ = model(dummy)
        ender.record()
        torch.cuda.synchronize()
        elapsed_ms = starter.elapsed_time(ender)          # ms
        elapsed_s  = elapsed_ms / 1000.0
    else:
        t0 = time.perf_counter()
        for _ in range(runs):
            _ = model(dummy)
        elapsed_s = time.perf_counter() - t0

    batch_size = dummy.shape[0]
    total_images = runs * batch_size
    fps = total_images / elapsed_s
    latency_ms = elapsed_s * 1000 / runs

    logger.info('─' * 55)
    logger.info(f'  测试轮次     : {runs}')
    logger.info(f'  Batch size   : {batch_size}')
    logger.info(f'  总耗时       : {elapsed_s:.3f} s')
    logger.info(f'  Latency/batch: {latency_ms:.2f} ms')
    logger.info(f'  FPS          : {fps:.1f} images/s')


def main():
    args   = parser.parse_args()
    logger = init_log('profile', logging.INFO)
    logger.propagate = False

    cfg    = yaml.load(open(args.config), Loader=yaml.Loader)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    model = build_model(cfg)

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, device)
        logger.info(f'已加载权重: {args.checkpoint}')
    else:
        logger.info('未提供 checkpoint，使用随机初始化权重')

    model.to(device).eval()

    H, W   = args.input_size
    dummy  = torch.randn(args.batch_size, 3, H, W, device=device)

    logger.info('=' * 55)
    logger.info(f'  Backbone     : {cfg["backbone"]}')
    logger.info(f'  Num classes  : {cfg["nclass"]}')
    logger.info(f'  Input size   : {args.batch_size} × 3 × {H} × {W}')
    logger.info(f'  Device       : {device}')

    dummy_single = torch.randn(1, 3, H, W, device=device)
    logger.info('\n[1/2] FLOPs & Params')
    measure_flops(model, dummy_single, logger)

    logger.info(f'\n[2/2] FPS  (batch={args.batch_size})')
    measure_fps(model, dummy, args.warmup, args.runs, device, logger)

    logger.info('=' * 55)

if __name__ == '__main__':
    main()