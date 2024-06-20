'''
Library import
'''
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from PIL import Image
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchmetrics
from torchvision import transforms as transforms
import importlib
from torchmetrics.classification import BinaryAUROC
from torchmetrics.classification import BinaryROC
from torcheval.metrics.functional import binary_f1_score
from tqdm import tqdm
import warnings
warnings.simplefilter('ignore')


'''
余計な出力抑制
'''
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


'''
関数定義
'''

class Dataset(torch.utils.data.Dataset):
    def __init__(self, d):
        self.data = d
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):        
        img, cimg, mask, label, clinical_feature, ccc = self.data[index]
        return img, cimg, mask, label, clinical_feature, ccc
##########################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
K = 5
data_ft = torch.load('Data_tensor/ft')
Dataset_ft    = Dataset(d=data_ft) 

iou_calculation = torchmetrics.classification.BinaryJaccardIndex().to(device='cuda:1') 


import Model.MS2_04011_2 as PlaNet_S_B
importlib.reload(PlaNet_S_B)
net2 = PlaNet_S_B.Net().to(device='cuda:1')  
net2.load_state_dict(torch.load('Result/step0/PlaNet_S_B_'+ str(1) + '.pth'))############B
net2.eval()            



for s in tqdm(range(int(len(data_ft))), desc="Loop level A: each case", position=0, leave=False):
    x, cx, t_seg, t_cls, clinical_feature, ccc= Dataset_ft[s]
    x = x.to(device='cuda:1') 
    t_seg = t_seg.to(device='cuda:1')
    ### TTA block
    TTA_all = []
    TTA_number = 100
    for loop in tqdm(range(TTA_number), desc="Loop level B: each case/Test time augmentation step", position=1, leave=False):
        # Parameter
        seed = random.randint(0, 2**32)
        random.seed(seed)
        angle = random.uniform(-45, 45)
        translate = (random.uniform(0, 1), random.uniform(0, 1))
        scale = random.uniform(0.8, 1.0)                             
        shear = 0
        # Affine transformation
        tmp = transforms.functional.affine(x, angle, translate, scale, shear)
        # Model
        tmp = net2(tmp.unsqueeze(0))  
        tmp = tmp.sigmoid()            
        # Reverse affine transformation
        tmp = transforms.functional.affine(tmp, -angle, (-translate[0], -translate[1]), 1/scale, shear)
        # Threshold & Average
        thresh = 0.5
        tmp = F.threshold(tmp, thresh, 0)
        tmp = F.threshold(-tmp, -thresh, 1)                
        tmp = F.interpolate(tmp, size=None, scale_factor=4, mode='bilinear')  
        if loop == 0:
            TTA_all = tmp
        else:
            TTA_all = torch.cat((TTA_all, tmp), 0)
    pred_seg_2   = torch.mean(input=TTA_all, dim=0).unsqueeze(0)   
    thresh = 0.50 ############  B
    pred_seg_2 = F.threshold(pred_seg_2, thresh, 0)
    pred_seg_2 = F.threshold(-pred_seg_2, -thresh, 1)  
#     pred_seg_2 = -F.max_pool2d(input = -pred_seg_2, kernel_size=5, stride=1, padding=2, dilation=1)
#     pred_seg_2 = F.max_pool2d(input = pred_seg_2, kernel_size=5, stride=1, padding=2, dilation=1)        
    filename = 'Result/step2/PlaNet_S_A_plus_B/'+str(s)
    torch.save(pred_seg_2, filename+"_pB")        


    
    
    