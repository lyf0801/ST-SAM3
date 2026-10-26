import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import random
from itertools import cycle
import numpy as np
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter

from options.train_options import TrainOptions
from dataloader import CreateTrainDataLoader
from algorithms import create_algorithm 
from evaluate import evaluate
from utils import Timer, AverageMeter, update_soft_conf_matrix

torch.set_num_threads(4)
torch.cuda.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def main():
    _t = {'iter time' : Timer()}
    _t['iter time'].tic()
    best_acc = 0
    best_fIoU = 0

    # initialize options
    opt = TrainOptions()
    args = opt.initialize()
    if args.percent == 100:        
        args.percent = str(int(args.percent)) if args.percent >= 1 else str(args.percent) 
        args.list_train_l = args.list_train_l.replace('_percent%_labeled', '')
        args.list_train_u = args.list_train_l
    else:
        args.percent = str(int(args.percent)) if args.percent >= 1 else str(args.percent) 
        args.list_train_l = args.list_train_l.replace('percent', args.percent)
        args.list_train_u = args.list_train_u.replace('percent', args.percent)

    args.save_dir = os.path.join(args.save_dir, args.dataset.replace('/', '_'), args.model)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    opt.print_options(args)

    # build dataloader, build algorithm, and load checkpoint
    train_l_loader, _, val_loader = CreateTrainDataLoader(args) 
    algorithm = create_algorithm(args, device)

    start_iter = 0
    save_checkpoint_path = os.path.join(args.save_dir, '%s_p-%s_%s' % \
                           (args.dataset.replace('/', '_'), args.percent, args.algorithm) + '.pth')
    if os.path.exists(save_checkpoint_path) and args.restore:
        print ('loading checkpoint from {}'.format(save_checkpoint_path))
        resume = torch.load(save_checkpoint_path)
        start_iter = resume['curr_iter']
        algorithm.load_state_dict(resume['state_dict'])
        val_acc, val_Recall, val_Precision, val_IoU, val_mIoU = evaluate(
            val_loader, algorithm, device)
        print ('Val:   Acc: {}  mIoU: {}'.format(val_acc, val_IoU))
        train_l_iter = iter(train_l_loader)

    # training loop
    losses_l = AverageMeter()
    for curr_iter in range(start_iter, args.num_iters):
        if curr_iter % len(train_l_loader) == 0:
            train_l_iter = iter(train_l_loader)

        image_l, target_l, _ = next(train_l_iter)       
        image_l = image_l.to(device).detach()
        target_l = target_l.unsqueeze(dim=1).float().to(device).detach()

        loss_dict = algorithm.update(image_l, target_l, curr_iter, args.num_iters)
        losses_l.update(loss_dict['loss_l'], image_l.size(0))

        # print info
        if (curr_iter+1) % args.print_freq==0:
            _t['iter time'].toc(average=False)
            print('[{}][{}-{}][it {}-{}][loss_l {:.4f}][lr {:.4f}][{:.2f}s]'.format(
                    algorithm.name, args.dataset, args.percent, curr_iter+1, args.num_iters, losses_l.avg, 
                    algorithm.optimizer.param_groups[0]['lr']*10000,
                    _t['iter time'].diff) )

        # evaluate model
        if (curr_iter+1) % args.eval_freq==0:
            val_acc, val_Recall, val_Precision, val_IoU, val_mIoU \
                = evaluate(val_loader, algorithm, device)
            print ('Target Val:  Acc:        {}'.format(val_acc))
            print ('Target Val:  Recall:     {}'.format(np.round(val_Recall, 4)))
            print ('Target Val:  Precision:  {}'.format(np.round(val_Precision, 4)))
            print ('Target Val:  IoU:        {}'.format(np.round(val_IoU, 4)))

            # save best model
            val_fIoU = val_IoU[1]
            if best_fIoU < val_fIoU:
                best_acc = val_acc
                best_fIoU = val_fIoU
                print ('taking snapshot ...')
                state = {
                    'curr_iter': curr_iter,
                    'best_acc': best_acc,
                    'best_fIoU': best_fIoU,
                    'state_dict': algorithm.state_dict(),
                }
                torch.save(state, save_checkpoint_path)
            losses_l.reset()
            algorithm.metric.reset()
            _t['iter time'].tic()
    

if __name__ == '__main__':
    main()
