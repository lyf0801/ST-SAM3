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
    best_IoU_fore = 0

    # initialize options
    opt = TrainOptions()
    args = opt.initialize()
    args.percent = str(int(args.percent)) if args.percent >= 1 else str(args.percent) 
    args.list_train_l = args.list_train_l.replace('percent', args.percent)
    args.list_train_u = args.list_train_u.replace('percent', args.percent)
    args.save_dir = os.path.join(args.save_dir, args.dataset.replace('/', '_'), args.model)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    opt.print_options(args)

    # build dataloader, build algorithm, and load checkpoint
    train_l_loader, train_u_loader, val_loader = CreateTrainDataLoader(args) 
    algorithm = create_algorithm(args, device)

    start_iter = 0
    save_checkpoint_path = os.path.join(args.save_dir, '%s_p-%s_%s_%s' % \
                           (args.dataset.replace('/', '_'), args.percent, 
                            args.algorithm, algorithm.lamda) + '.pth')
    if os.path.exists(save_checkpoint_path) and args.restore:
        print ('loading checkpoint from {}'.format(save_checkpoint_path))
        resume = torch.load(save_checkpoint_path)
        start_iter = resume['curr_iter']
        algorithm.load_state_dict(resume['state_dict'])
        val_acc, val_Recall, val_Precision, val_IoU, val_mIoU = evaluate(
            val_loader, algorithm, device)
        print ('Val:   Acc: {}  mIoU: {}'.format(val_acc, val_IoU))
        train_l_iter = iter(train_l_loader)
        train_u_iter = iter(train_u_loader)

    # training loop
    losses_l = AverageMeter()
    losses_u = AverageMeter()
    for curr_iter in range(start_iter, args.num_iters):
        if curr_iter % len(train_l_loader) == 0:
            train_l_iter = iter(train_l_loader)
        if curr_iter % len(train_u_loader) == 0:
            train_u_iter = iter(train_u_loader)

        image_l, target_l, _ = next(train_l_iter)       
        image_l = image_l.to(device).detach()
        target_l = target_l.unsqueeze(dim=1).float().to(device).detach()

        # UniMatch needs two streams of unlabeled data
        if algorithm.name == 'UniMatch':
            image_u, image_us, image_us2, target_u, _ = next(train_u_iter)    
            image_u = image_u.to(device).detach()
            image_us = image_us.to(device).detach()
            image_us2 = image_us2.to(device).detach()
            loss_dict = algorithm.update(image_l, target_l, \
                image_u, image_us, image_us2, target_u, curr_iter, args.num_iters)
        else:            
            image_u, image_us, target_u, _ = next(train_u_iter)    
            image_u = image_u.to(device).detach()
            image_us = image_us.to(device).detach()
            loss_dict = algorithm.update(image_l, target_l, \
                image_u, image_us, target_u, curr_iter, args.num_iters)

        losses_l.update(loss_dict['loss_l'], image_l.size(0))
        losses_u.update(loss_dict['loss_u'], image_l.size(0))

        # print info
        if (curr_iter+1) % args.print_freq==0:
            _t['iter time'].toc(average=False)
            if 'DBMatch' in algorithm.name:
                print('[{}][{}-{}][it {}-{}][loss_l {:.4f}][loss_u {:.4f}][iou_u {:.4f}][lmd {:.2f}][bias {:.4f}][thr {:.2f}][lr {:.4f}][{:.0f}s]'.format(
                        algorithm.name, args.dataset, args.percent, curr_iter+1, args.num_iters, 
                        losses_l.avg, losses_u.avg, algorithm.metric.Intersection_over_Union()[1], 
                        algorithm.lamda, algorithm.bias.item(), algorithm.conf_thr,
                        algorithm.optimizer.param_groups[0]['lr']*10000, _t['iter time'].diff)
                )
            elif 'DebiasPL' in algorithm.name:
                print('[{}][{}-{}][it {}-{}][loss_l {:.4f}][loss_u {:.4f}][iou_u {:.4f}][bias {:.2f}][lr {:.4f}][{:.2f}s]'.format(
                        algorithm.name, args.dataset, args.percent, curr_iter+1, args.num_iters, 
                        losses_l.avg, losses_u.avg, algorithm.metric.Intersection_over_Union()[1], 
                        algorithm.lamda * algorithm.bias.item(), 
                        algorithm.optimizer.param_groups[0]['lr']*10000, _t['iter time'].diff)
                )
            elif 'AdaptMatch' in algorithm.name:
                print('[{}][{}-{}][it {}-{}][loss_l {:.4f}][loss_u {:.4f}][iou_u {:.4f}][thr {:.4f}, {:.4f}][lr {:.4f}][{:.2f}s]'.format(
                        algorithm.name, args.dataset, args.percent, curr_iter+1, args.num_iters, 
                        losses_l.avg, losses_u.avg, algorithm.metric.Intersection_over_Union()[1], 
                        algorithm.thr_fore, 1 - algorithm.thr_back, 
                        algorithm.optimizer.param_groups[0]['lr']*10000, _t['iter time'].diff)
                )
            else:
                print('[{}][{}-{}][it {}-{}][loss_l {:.4f}][loss_u {:.4f}][lr {:.4f}][{:.2f}s]'.format(
                        algorithm.name, args.dataset, args.percent, curr_iter+1, args.num_iters, 
                        losses_l.avg, losses_u.avg, algorithm.optimizer.param_groups[0]['lr']*10000, _t['iter time'].diff) 
                )

        # evaluate and save model
        if (curr_iter+1) % args.eval_freq==0:
            val_acc, val_Recall, val_Precision, val_IoU, val_mIoU \
                = evaluate(val_loader, algorithm, device)
            print ('Target Val:  Acc:        {}'.format(val_acc))
            print ('Target Val:  Recall:     {}'.format(np.round(val_Recall, 4)))
            print ('Target Val:  Precision:  {}'.format(np.round(val_Precision, 4)))
            print ('Target Val:  IoU:        {}'.format(np.round(val_IoU, 4)))

            file_name = algorithm.name + '_' + args.dataset.replace('/', '_') + '_' + str(args.percent) + '.txt'
            file_path = os.path.join('./predicts_other', file_name)
            if 'DBMatch' in algorithm.name: 
                with open(file_path, 'a') as file:
                    file.write(f'{val_Recall[1]} {val_Precision[1]} {val_IoU[1]} {algorithm.bias.item()}\n')
            else:                
                with open(file_path, 'a') as file:
                    file.write(f'{val_Recall[1]} {val_Precision[1]} {val_IoU[1]}\n')

            # save best model
            val_IoU_fore = val_IoU[1]
            if best_IoU_fore < val_IoU_fore:
                best_IoU_fore = val_IoU_fore
                print ('taking snapshot ...')
                state = {
                    'curr_iter': curr_iter,
                    'best_IoU_fore': best_IoU_fore,
                    'state_dict': algorithm.state_dict(),
                }
                torch.save(state, save_checkpoint_path)
            losses_l.reset()
            losses_u.reset()
            algorithm.metric.reset()
            _t['iter time'].tic()
    

if __name__ == '__main__':
    main()
