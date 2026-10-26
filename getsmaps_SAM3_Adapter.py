import numpy as np
import torch
import torch.utils.data as Data
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from datasets.ORSI_SOD_dataset import ORSI_SOD_Dataset
from tqdm import tqdm


import models

from evaluator import Eval_thread
from PIL import Image
import time


import torchvision
from ptflops import get_model_complexity_info

def print_model_complexity(model, input_size):
    macs, params = get_model_complexity_info(model, input_size, as_strings=True,
                                              print_per_layer_stat=True, verbose=True)
    print(f"MACs: {macs}, Params: {params}")

def unload(x):
    y = x.squeeze().cpu().data.numpy()
    return y
def convert2img(x):
    return Image.fromarray(x*255).convert('L')
def min_max_normalization(x):
    x_normed = (x - np.min(x)) / (np.max(x)-np.min(x))
    return x_normed
def save_smap(smap, path, negative_threshold=0.25):
    # smap: [1, H, W]
    if torch.max(smap) <= negative_threshold:
        smap[smap<negative_threshold] = 0
        smap = convert2img(unload(smap))
    else:
        smap = convert2img(min_max_normalization(unload(smap)))
    if smap.size != (448, 448):
        smap = smap.resize((448,448),Image.Resampling.BICUBIC)
    smap.save(path)



def getsmaps(dataset_name):
   
    test_set = ORSI_SOD_Dataset(dataset_name = dataset_name, img_size=448,  mode = "test", aug = False)

    test_loader = DataLoader(test_set, batch_size = 1, shuffle=False, num_workers=4)

    #define network and load weights
   
    #net = Net(img_size=(448, 448)).cuda().eval()  #img_size=(448, 448)
    config = {
        "name": "sam", 
        "args": {
            "inp_size": 448,
            "loss": "iou",
            "encoder_mode": {
                "name": "sam",
                "img_size": 448,
                "mlp_ratio": 4,
                "patch_size": 16,
                "qkv_bias": True,
                "use_rel_pos": True,
                "window_size": 14,
                "out_chans": 256,
                "scale_factor": 32,
                "input_type": "fft",
                "freq_nums": 0.25,
                "prompt_type": "highpass",
                "prompt_embed_dim": 256,
                "tuning_stage": "1234",  # 注意：原文是数字但无引号，若表示阶段字符串建议用 str
                "handcrafted_tune": True,
                "embedding_tune": True,
                "adaptor": "adaptor",
                "embed_dim": 1024,
                "depth": 24,
                "num_heads": 16,
                "global_attn_indexes": [5, 11, 17, 23],
            }
        }
    }
    net = models.make(config).cuda().eval()

   

    print_model_complexity(net, (3, 448, 448))
    
    if dataset_name == "ORSSD":

        #net.load_state_dict(torch.load("./data/SAM3_Adapter_ORSSD_5%/best.pth", map_location='cpu'))
        #net.load_state_dict(torch.load("./data/SAM3_Adapter_ORSSD_10%/best.pth", map_location='cpu'))
       
        #net.load_state_dict(torch.load("./data/Retrain_SAM3_Adapter_ORSSD_5%/best.pth", map_location='cpu'))
        #net.load_state_dict(torch.load("./data/Retrain_SAM3_Adapter_ORSSD_10%/best.pth", map_location='cpu'))

        #net.load_state_dict(torch.load("./data/Re_Retrain_SAM3_Adapter_ORSSD_1%/best.pth", map_location='cpu'))
        #net.load_state_dict(torch.load("./data/Re_Retrain_SAM3_Adapter_ORSSD_5%/best.pth", map_location='cpu'))
        #net.load_state_dict(torch.load("./data/Re_Retrain_SAM3_Adapter_ORSSD_10%/best.pth", map_location='cpu'))

        #net.load_state_dict(torch.load("./data/Re_Re_Retrain_SAM3_Adapter_ORSSD_5%/best.pth", map_location='cpu'))
        #net.load_state_dict(torch.load("./data/Re_Re_Retrain_SAM3_Adapter_ORSSD_10%/best.pth", map_location='cpu'))

        net.load_state_dict(torch.load("./weights/ST-SAM3_ORSSD_5%.pth", map_location='cpu'))
        #net.load_state_dict(torch.load("./weights/ST-SAM3_ORSSD_10%.pth", map_location='cpu'))
    elif dataset_name == "EORSSD":
        #net.load_state_dict(torch.load("./data/SAM3_Adapter_EORSSD_5%/best.pth", map_location='cpu'))
        #net.load_state_dict(torch.load("./data/SAM3_Adapter_EORSSD_10%/best.pth", map_location='cpu'))

        #net.load_state_dict(torch.load("./data/Retrain_SAM3_Adapter_EORSSD_5%/best.pth", map_location='cpu'))
        #net.load_state_dict(torch.load("./data/Retrain_SAM3_Adapter_EORSSD_10%/best.pth", map_location='cpu'))

        #net.load_state_dict(torch.load("./data/Re_Retrain_SAM3_Adapter_EORSSD_1%/best.pth", map_location='cpu'))
        #net.load_state_dict(torch.load("./data/Re_Retrain_SAM3_Adapter_EORSSD_5%/best.pth", map_location='cpu'))
        #net.load_state_dict(torch.load("./data/Re_Retrain_SAM3_Adapter_EORSSD_10%/best.pth", map_location='cpu'))

        #net.load_state_dict(torch.load("./data/Re_Re_Retrain_SAM3_Adapter_EORSSD_5%/best.pth", map_location='cpu'))
        #net.load_state_dict(torch.load("./data/Re_Re_Retrain_SAM3_Adapter_EORSSD_10%/best.pth", map_location='cpu'))


        net.load_state_dict(torch.load("./weights/ST-SAM3_EORSSD_5%.pth", map_location='cpu'))
        #net.load_state_dict(torch.load("./weights/ST-SAM3_EORSSD_10%.pth", map_location='cpu'))
    elif dataset_name == "ORS_4199":
        #net.load_state_dict(torch.load("./data/SAM3_Adapter_ORS_4199_5%/best.pth", map_location='cpu'))
        #net.load_state_dict(torch.load("./data/SAM3_Adapter_ORS_4199_10%/best.pth", map_location='cpu'))

        #net.load_state_dict(torch.load("./data/Retrain_SAM3_Adapter_ORS_4199_5%/best.pth", map_location='cpu'))
        #net.load_state_dict(torch.load("./data/Retrain_SAM3_Adapter_ORS_4199_10%/best.pth", map_location='cpu'))

        #net.load_state_dict(torch.load("./data/Re_Retrain_SAM3_Adapter_ORS_4199_1%/best.pth", map_location='cpu'))
        #net.load_state_dict(torch.load("./data/Re_Retrain_SAM3_Adapter_ORS_4199_5%/best.pth", map_location='cpu'))
        #net.load_state_dict(torch.load("./data/Re_Retrain_SAM3_Adapter_ORS_4199_10%/best.pth", map_location='cpu'))
       

        #net.load_state_dict(torch.load("./data/Re_Re_Retrain_SAM3_Adapter_ORS_4199_5%/best.pth", map_location='cpu'))
        #net.load_state_dict(torch.load("./data/Re_Re_Retrain_SAM3_Adapter_ORS_4199_10%/best.pth", map_location='cpu'))


        net.load_state_dict(torch.load("./weights/ST-SAM3_ORS_4199_5%.pth", map_location='cpu'))
        #net.load_state_dict(torch.load("./weights/ST-SAM3_ORS_4199_10%.pth", map_location='cpu'))
    ##save saliency map
    infer_time = 0
    for image, label, _, name in tqdm(test_loader):
    #for data in tqdm(test_loader):
        with torch.no_grad():
            image, label = image.cuda(), label.cuda()
            
            t1 = time.time()
            smap = net(image) 
            if isinstance(smap, list) or isinstance(smap, tuple):
                smap = smap[0]  # 如果输出是一个列表或元组，取第一个元素作为 saliency map
            t2 = time.time()
            infer_time += (t2 - t1)
                
            #if not exist then define
            dirs = "./data/output/predict_smaps" +  "_ST-SAM3_448_5%_" + dataset_name
            if not os.path.exists(dirs):
                os.makedirs(dirs)
            path = os.path.join(dirs, name[0] + "_ST-SAM3_448_5%" + '.png')
            save_smap(smap, path)
            
    print(len(test_loader))
    print(infer_time)
    print(len(test_loader) / infer_time)  # inference speed (without I/O time),

if __name__ == "__main__":

    dataset = ["ORSSD", "EORSSD", "ORS_4199"]# , , 
    for datseti in dataset:
        getsmaps(datseti)

