import os
import os.path as osp

import cv2
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from dataloader import CreateTestDataLoader
from algorithms import create_algorithm 

from options.test_options import TestOptions
from utils import Evaluator, upsample_size, upsample_scale

torch.set_num_threads(4)
torch.cuda.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test(test_loader, algorithm, num_classes):
    num_classes = 2
    if 'DBMatch' in algorithm.name:
        print (algorithm.bias)

    algorithm.eval()
    metric = Evaluator(num_classes)
    with torch.no_grad():
        for i, (image, label, _) in enumerate(test_loader):
            image, label = image.cuda(), label.cuda()  
            feat = algorithm.get_feats(image)
            logit = algorithm.get_logits(feat)
            if 'DBMatch' in algorithm.name:
                logit = algorithm.debias_logits(logit)
            prob = torch.sigmoid(logit)

            prob = prob.reshape(-1).cpu().detach().numpy()            
            logit = logit.reshape(-1).cpu().detach().numpy()
            label = label.reshape(-1).cpu().detach().numpy()
            pred = (prob > 0.5).astype(np.int64)
            metric.add_batch(label, pred)

    test_Precision, test_Recall = metric.Precision_Recall()
    test_IoU = metric.Intersection_over_Union()
    test_F1 = metric.F1_Score()
    return test_Recall[1], test_Precision[1], test_IoU[1], test_F1[1]


if __name__ == '__main__':    
    opt = TestOptions()
    args = opt.initialize()
    args.percent = str(int(args.percent)) if args.percent >= 1 else str(args.percent) 
    args.save_dir = os.path.join(args.save_dir, args.dataset.replace('/', '_'), args.model)
    opt.print_options(args)

    # build loader, build algorithm, and load checkpoint
    test_loader = CreateTestDataLoader(args)    
    algorithm = create_algorithm(args, device)
    save_checkpoint_path = os.path.join(args.save_dir, '%s_p-%s_%s_0.6' % \
                           (args.dataset.replace('/', '_'), args.percent, args.algorithm) + '.pth')
    if os.path.exists(save_checkpoint_path):
        print ('loading checkpoint from {}'.format(save_checkpoint_path))
        resume = torch.load(save_checkpoint_path)
        algorithm.load_state_dict(resume['state_dict'], strict=False)
    else:
        raise FileNotFoundError(f"Error: The file '{save_checkpoint_path}' does not exist.")      

    # test epoch
    test_Rec, test_Prec, test_IoU, test_F1 = test(test_loader, algorithm, args.num_classes)

    print ('Test on {}'.format(args.dataset))
    print (' Rec, Pre, IoU, F1 : & {} & {} & {} & {}'.format(
        np.round(test_Rec*100, 2), np.round(test_Prec*100, 2), 
        np.round(test_IoU*100, 2), np.round(test_F1*100, 2)))

