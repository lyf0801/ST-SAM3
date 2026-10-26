#  Change Guiding Network: Incorporating Change Prior to Guide Change Detection in Remote Sensing Imagery,
#  IEEE J. SEL. TOP. APPL. EARTH OBS. REMOTE SENS., PP. 1–17, 2023, DOI: 10.1109/JSTARS.2023.3310208. C. HAN, C. WU, H. GUO, M. HU, J.Li AND H. CHEN,


import os

import cv2
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import utils.visualization as visual
from utils import data_loader
from tqdm import tqdm
import random
from utils.metrics import Evaluator
from network.CGNet import HCGMNet,CGNet
from network.ResNet50_PSP import ResNet__PSP
import time
from itertools import cycle
from SAM.segment_anything import SamPredictor, sam_model_registry
from SAM.sam_prompt import load_mask_and_safe_centers
start=time.time()


class FeatureMemory:
    def __init__(self, memory_length=10000, feat_dim=512):
        self.memory_length = memory_length
        self.fts_memory = []
        self.fts_memory.append(torch.zeros(0, feat_dim).cuda())  # Memo for fts


    def check_if_full(self):
        full = True
        for item in self.fts_memory:
            if item.size(0) < self.memory_length:
                full = False
        return full

    @torch.no_grad()
    def update(self, fts):
        self.fts_memory[0] = torch.cat((fts, self.fts_memory[0]), dim=0)[:self.memory_length]



def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def multi_scale_similarity(ins_features,memory_features,k=[32,64,128]):
    if ins_features.shape[0]>1000:
       idx = np.random.choice(ins_features.shape[0], 1000, False)
       ins_features=ins_features[idx]

    ins_features = torch.nn.functional.normalize(ins_features, p=2, dim=1).mean(dim=0).unsqueeze(1)  # N C -> C 1
    memory_features = torch.nn.functional.normalize(memory_features, p=2, dim=1).mean(dim=0).unsqueeze(1)   # M C -> C 1
    ins_memo_sim=torch.cosine_similarity(ins_features.unsqueeze(1), memory_features.unsqueeze(0), dim=2) # C C

    k_1, k_2, k_3 = k

    v_1, _ = torch.topk(ins_memo_sim, k=k_1, dim=1) # C k_1
    v_2, _ = torch.topk(ins_memo_sim, k=k_2, dim=1) # C k_2
    v_3, _ = torch.topk(ins_memo_sim, k=k_3, dim=1) # C k_3

    den_1 = torch.sum(v_1, dim=1) / k_1 # C 1
    den_2 = torch.sum(v_2, dim=1) / k_2 # C 1
    den_3 = torch.sum(v_3, dim=1) / k_3 # C 1

    ins_similarity = (den_1 + den_2 + den_3) / 3  # C 1
    return ins_similarity

def semi_train(train_loader,unsupervised_train_loader, val_loader, Eva_train,Eva_val, data_name, save_path, net, criterion, criterion_noreduction,optimizer, num_epoches,fm,criterion_mse, predictor,flag):
    vis = visual.Visualization()
    vis.create_summary(data_name)
    global best_iou
    epoch_loss = 0
    net.train(True)
    length = 0
    st = time.time()

    dataloader = iter(zip(cycle(train_loader), unsupervised_train_loader))
    tbar = tqdm(range(len(unsupervised_train_loader)))

    for batch_idx in enumerate(tbar):
        (A_l, B_l, target_l), (WA_ul, WB_ul, SA_ul, SB_ul, target_ul_gt,imageB,imageA) = next(dataloader)
        WA_ul, WB_ul = WA_ul.cuda(non_blocking=True), WB_ul.cuda(non_blocking=True)
        SA_ul, SB_ul = SA_ul.cuda(non_blocking=True), SB_ul.cuda(non_blocking=True)
        A_l, B_l, target_l = A_l.cuda(non_blocking=True), B_l.cuda(non_blocking=True), target_l.cuda(non_blocking=True)
        optimizer.zero_grad()

        # supervised
        preds_l,sup_d = net(A_l, B_l)
        loss_l = criterion(preds_l, target_l)

        # supervised difference update
        h_d,w_d = sup_d.shape[2:]
        target_l_ = F.interpolate(target_l, size=(h_d,w_d), mode='nearest').squeeze(1)
        sup_d = sup_d.permute(0,2,3,1) # B H W C
        ins_sup = sup_d[target_l_==1] # N,C
        fm.update(ins_sup)


        # un_supervised
        weak_ul,W_d=net(WA_ul,WB_ul)
        strong_ul,S_d=net(SA_ul,SB_ul)

        # target_ul
        target_ul_prob = F.sigmoid(weak_ul)
        target_ul=target_ul_prob.clone().cuda()

        target_ul[target_ul >= 0.5] = 1
        target_ul[target_ul < 0.5] = 0

        final_prob_map=target_ul_prob.clone().cuda()
        # introduce SAM
        for idx,imageB_name in enumerate(imageB):
            if imageB_name != "no_sam":
                image_ori = None
                if(flag==1):
                    # use T2 image
                    image_ori = cv2.imread(imageB_name)
                elif(flag==2):
                    # use T1 image
                    image_ori = cv2.imread(imageA[idx])
                else:
                    image_ori_A = cv2.imread(imageA[idx])
                    image_ori_B = cv2.imread(imageB[idx])

                    arr1 = np.array(image_ori_A, dtype=np.int16)
                    arr2 = np.array(image_ori_B, dtype=np.int16)
                    # use a new image generated from T1 and T2 iamge
                    image_ori = np.abs(arr1 - arr2)
                    image_ori = np.clip(image_ori, 0, 255).astype(np.uint8)
                predictor.set_image(image_ori)
                # pseudo_label
                binary_mask_before_sam = target_ul[idx].clone().detach().cpu().numpy().squeeze() # (256 256)

                safe_centers, bounding_boxes = load_mask_and_safe_centers(binary_mask_before_sam)
                point_coords = safe_centers.astype(np.float32) if len(safe_centers) > 0 else None
                point_labels = np.ones(len(safe_centers), dtype=np.int32) if len(safe_centers) > 0 else None
                prob_map_after_sam = target_ul_prob[idx]
                if bounding_boxes is None or point_coords is None or point_labels is None:
                    final_prob_map[idx] = final_prob_map[idx]
                    continue
                for box, point, label in zip(bounding_boxes, point_coords, point_labels):
                    masks, _, logits = predictor.predict(
                        box = box[None, :],
                        point_coords = point[None, :],
                        point_labels = np.array([label]),
                        multimask_output = False
                    )
                    masks_torch = torch.from_numpy(masks).cuda()
                    logits_torch = torch.from_numpy(logits).cuda()
                    prob_map = torch.sigmoid(logits_torch).cuda()
                    prob_map_after_sam = prob_map_after_sam+(prob_map*0.5-prob_map_after_sam*0.5)*masks_torch
                    final_prob_map[idx] = prob_map_after_sam
            else:
                final_prob_map[idx] = final_prob_map[idx]
        final_target_ul = final_prob_map.clone().cuda()
        final_target_ul[final_target_ul>=0.5] = 1
        final_target_ul[final_target_ul < 0.5] = 0
        mask = torch.zeros_like(final_prob_map)
        mask[(final_prob_map > 0.9) | (final_prob_map < 0.1)] = 1
        mask = mask.float()

        # IA loss
        h_d,w_d = W_d.shape[2:]
        mask_ = F.interpolate(mask, size=(h_d,w_d), mode='nearest').squeeze(1).cuda()
        W_d = W_d.permute(0, 2, 3, 1)
        S_d = S_d.permute(0, 2, 3, 1)
        W_d_selected = torch.zeros(W_d.shape[0],W_d.shape[1],W_d.shape[2],128).cuda()
        S_d_selected = torch.zeros(S_d.shape[0], S_d.shape[1], S_d.shape[2], 128).cuda()
        for i in range(W_d.shape[0]):
            ins_W_d_1 = W_d[i][mask_[i]==1]
            # get memory features
            memory_changed_ins_features = fm.fts_memory[0]
            if memory_changed_ins_features.shape[0] > 1000:
                idx = np.random.choice(memory_changed_ins_features.shape[0], 1000, False)
                memory_changed_ins_features = memory_changed_ins_features[idx]

            if memory_changed_ins_features.shape[0]>0 and ins_W_d_1.shape[0]>0:
              similarity = multi_scale_similarity(ins_W_d_1,memory_changed_ins_features)
            else:
              similarity = torch.randn(W_d.shape[3])


            _,idx = torch.topk(similarity, k=128, dim=0)

            W_d_i=W_d[i,:,:,:].squeeze(dim=0).permute(2,0,1)
            W_d_selected[i,:,:,:] = W_d_i[idx].permute(1,2,0)

            S_d_i = S_d[i, :, :, :].squeeze(dim=0).permute(2, 0, 1)
            S_d_selected[i, :, :, :] = S_d_i[idx].permute(1, 2, 0)

        loss_ia = (criterion_mse(W_d_selected,S_d_selected).mean(dim=3)*mask_).mean()

        loss_ul = (criterion_noreduction(strong_ul, final_target_ul) * mask).mean()
        total_loss = loss_l+0.5*(loss_ul+0.5*loss_ia)
        total_loss.backward()
        optimizer.step()
        epoch_loss += total_loss.item()


        # supervised part metirc
        output = F.sigmoid(preds_l)
        output[output >= 0.5] = 1
        output[output < 0.5] = 0
        pred = output.data.cpu().numpy().astype(int)
        target = target_l.cpu().numpy().astype(int)
        Eva_train.add_batch(target, pred)
        length += 1
    IoU = Eva_train.Intersection_over_Union()[1]
    Pre = Eva_train.Precision()[1]
    Recall = Eva_train.Recall()[1]
    F1 = Eva_train.F1()[1]
    train_loss = epoch_loss / length

    vis.add_scalar(epoch, IoU, 'mIoU')
    vis.add_scalar(epoch, Pre, 'Precision')
    vis.add_scalar(epoch, Recall, 'Recall')
    vis.add_scalar(epoch, F1, 'F1')
    vis.add_scalar(epoch, train_loss, 'train_loss')

    print(
        'Epoch [%d/%d], Loss: %.4f,\n[Training]IoU: %.4f, Precision:%.4f, Recall: %.4f, F1: %.4f' % (
            epoch, num_epoches, \
            train_loss, \
            IoU, Pre, Recall, F1))
    print("Strat validing!")

    net.train(False)
    net.eval()
    for i, (A, B, mask, filename) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            A = A.cuda()
            B = B.cuda()
            Y = mask.cuda()
            preds,_ = net(A, B)
            output = F.sigmoid(preds)
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            pred = output.data.cpu().numpy().astype(int)
            target = Y.cpu().numpy().astype(int)

            Eva_val.add_batch(target, pred)

            length += 1
    IoU = Eva_val.Intersection_over_Union()
    Pre = Eva_val.Precision()
    Recall = Eva_val.Recall()
    F1 = Eva_val.F1()
    print('[Validation] IoU: %.4f, Precision:%.4f, Recall: %.4f, F1: %.4f' % (IoU[1], Pre[1], Recall[1], F1[1]))

    new_iou = IoU[1]
    if new_iou >= best_iou:
        best_iou = new_iou
        best_epoch = epoch
        best_net = net.state_dict()
        print('Best Model Iou :%.4f; F1 :%.4f; Best epoch : %d' % (IoU[1], F1[1], best_epoch))
        torch.save(best_net, save_path + '_best_iou.pth')
    print('Best Model Iou :%.4f; F1 :%.4f' % (best_iou, F1[1]))
    vis.close_summary()

def poly_lr(epoch):
    total_epochs = 50
    power = 0.9
    return (1 - epoch / total_epochs) ** power

if __name__ == '__main__':
    seed_everything(42)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=50, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=4, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
    parser.add_argument('--data_name', type=str, default='WHU',
                        help='the test rgb images root')
    parser.add_argument('--flag', type=int, default=1)
    parser.add_argument('--labeled_ratio', type=int, default=5)
    parser.add_argument('--model_name', type=str, default='resnet_changer',
                        help='the test rgb images root')
    parser.add_argument('--save_path', type=str,
                        default='./output/')
    opt = parser.parse_args()

    # set the device for training
    if opt.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')
    elif opt.gpu_id == '1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('USE GPU 1')
    if opt.gpu_id == '2':
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        print('USE GPU 2')
    if opt.gpu_id == '3':
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        print('USE GPU 3')
    if opt.gpu_id == '4':
        os.environ["CUDA_VISIBLE_DEVICES"] = "4"
        print('USE GPU 4')
    if opt.gpu_id == '5':
        os.environ["CUDA_VISIBLE_DEVICES"] = "5"
        print('USE GPU 5')
    if opt.gpu_id == '6':
        os.environ["CUDA_VISIBLE_DEVICES"] = "6"
        print('USE GPU 6')
    if opt.gpu_id == '7':
        os.environ["CUDA_VISIBLE_DEVICES"] = "7"
        print('USE GPU 7')

    opt.save_path = opt.save_path + opt.data_name + '/' + opt.model_name
    if opt.data_name == 'LEVIR':
        opt.train_root = '/data/chengxi.han/data/LEVIR CD Dataset256/train/'
        opt.val_root = '/data/chengxi.han/data/LEVIR CD Dataset256/val/'
    elif opt.data_name == 'WHU':
        opt.train_root = '/data/chengxi.han/data/Building change detection dataset256/train/'
        opt.val_root = '/data/chengxi.han/data/Building change detection dataset256/val/'
    elif opt.data_name == 'CDD':
        opt.train_root = '/data/chengxi.han/data/CDD_ChangeDetectionDataset/Real/subset/train/'
        opt.val_root = '/data/chengxi.han/data/CDD_ChangeDetectionDataset/Real/subset/val/'
    elif opt.data_name == 'DSIFN':
        opt.train_root = '/data/chengxi.han/data/DSIFN256/train/'
        opt.val_root = '/data/chengxi.han/data/DSIFN256/val/'
    elif opt.data_name == 'SYSU':
        opt.train_root = '/data/chengxi.han/data/SYSU-CD/train/'
        opt.val_root = '/data/chengxi.han/data/SYSU-CD/val/'
    elif opt.data_name == 'S2Looking':
        opt.train_root = '/data/chengxi.han/data/S2Looking256/train/'
        opt.val_root = '/data/chengxi.han/data/S2Looking256/val/'


    # semi-train
    supervised_dataset_txt = opt.labeled_ratio + "_train_supervised" + ".txt"
    unsupervised_dataset_txt = opt.labeled_ratio + "_train_unsupervised" + ".txt"
    train_loader = data_loader.get_loader(opt.train_root, supervised_dataset_txt, opt.batchsize, opt.trainsize, num_workers=2, shuffle=True, pin_memory=True)
    val_loader = data_loader.get_test_loader(opt.val_root, opt.batchsize, opt.trainsize, num_workers=2, shuffle=False, pin_memory=True)
    unsupervised_train_loader=data_loader.get_unsupervised_train_loader(opt.train_root, unsupervised_dataset_txt,opt.batchsize, opt.trainsize, num_workers=2, shuffle=True, pin_memory=True)

    Eva_train = Evaluator(num_class = 2)
    Eva_val = Evaluator(num_class=2)

    if opt.model_name == 'HCGMNet':
        model = HCGMNet().cuda()
    elif opt.model_name == 'CGNet':
        model = CGNet().cuda()
    elif opt.model_name=='ResNet__PSP':
        model=ResNet__PSP().cuda()

    sam = sam_model_registry["vit_h"](checkpoint="pth/sam.pth").cuda()
    predictor = SamPredictor(sam)
    # Freeze backbone network parameters
    # model.freeze_backbone_param()


    criterion = nn.BCEWithLogitsLoss().cuda()
    criterion_noreduction = nn.BCEWithLogitsLoss(reduction='none').cuda()
    criterion_mse=nn.MSELoss(reduction='none').cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=poly_lr)
    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data_name = opt.data_name
    best_iou = 0.0

    print("Start train...")

    # semi-supervised
    for epoch in range(1, opt.epoch):
        fm=FeatureMemory()
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
        Eva_train.reset()
        Eva_val.reset()
        semi_train(train_loader,unsupervised_train_loader, val_loader, Eva_train,Eva_val, data_name, save_path, model, criterion, criterion_noreduction,optimizer, opt.epoch,fm,criterion_mse,predictor,opt.flag)
        lr_scheduler.step()

end=time.time()
print('程序训练train的时间为:',end-start)


