from pickle import FALSE
import numpy as np
import os
from PIL import Image
from compute_evaluator import Eval_thread
from ORSI_SOD_dataset import ORSI_SOD_Dataset
from torch.utils.data import DataLoader
import os
import time
from torch import nn
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
import cv2
from PIL import Image

if __name__ == '__main__':   
    method_names = ['ST-SAM3_448_5%', 'ST-SAM3_448_10%', 'DistillMatch_448_5%', 'DistillMatch_448_10%']
    print("一共" + str(len(method_names)) + "种对比算法")
    dataset_names = ["ORSSD", "EORSSD", "ORS_4199"]
    for method_name in method_names:
        for dataset_name in dataset_names:

            
            smap_path = "./data/output/predict_smaps_" + method_name + "_" + dataset_name + "/"
          
            test_dataset = ORSI_SOD_Dataset(dataset_name=dataset_name, mode = "test", img_size = 448, aug = False)
            test_loader = DataLoader(test_dataset, batch_size = 1, shuffle=False, num_workers = 4)  

            thread = Eval_thread(smap_path=smap_path, loader = test_loader, method = method_name, dataset = dataset_name, output_dir = "./data/", cuda=True)
            logg, fm = thread.run()
            print(logg)