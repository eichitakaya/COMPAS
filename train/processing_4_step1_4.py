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
      
threshs = np.arange(0.05, 1.05, 0.05)
for n3 in range(5):
    data_test = torch.load('Data_tensor/f'+str(n3)+'_test')
    iou_calculation = torchmetrics.classification.BinaryJaccardIndex().to(device='cuda:0') 
    iou_PlaNet_S_A = []
    dice_PlaNet_S_A = []
    iou_PlaNet_S_B = []
    dice_PlaNet_S_B = []
    for n2 in range(2):
        iou_case = []
        dice_case = []
        iou_fold = []
        dice_fold = []
        if n2 == 0:
            pred_tar = torch.load('Result/step1_3/PlaNet_S_A/f'+str(n3)+"_pred_tar")
        elif n2 == 1:
            pred_tar = torch.load('Result/step1_3/PlaNet_S_B/f'+str(n3)+"_pred_tar")
        for n1 in range(20):
            for s in tqdm(range(int(len(data_test))), desc="Loop level A: each case", position=0, leave=False):
                pred_seg = pred_tar[s][0].to(device='cuda:0')
                t_seg    = pred_tar[s][1].to(device='cuda:0')
                thresh = threshs[n1]
                pred_seg = F.threshold(pred_seg, thresh, 0)
                pred_seg = F.threshold(-pred_seg, -thresh, 1)
#                 pred_seg = -F.max_pool2d(input = -pred_seg, kernel_size=5, stride=1, padding=2, dilation=1)
#                 pred_seg = F.max_pool2d(input = pred_seg, kernel_size=5, stride=1, padding=2, dilation=1)
                pred_seg = pred_seg.squeeze()
                t_seg = t_seg.to(torch.float32)

                ##### Criteriaを計算する　#####
                iou = iou_calculation(pred_seg, t_seg)
                iou_case.append(iou.item())

                dice = binary_f1_score(pred_seg.view(-1), t_seg.view(-1))
                dice_case.append(dice.item())

            iou_fold_thresh = torch.mean(input=torch.tensor(iou_case))
            dice_fold_thresh = torch.mean(input=torch.tensor(dice_case))
            iou_fold.append(iou_fold_thresh.detach())
            dice_fold.append(dice_fold_thresh.detach())
        if n2 == 0:
            torch.save(iou_fold, 'Result/step1_4/PlaNet_S_A/f'+str(n3)+"_iou")
            torch.save(dice_fold, 'Result/step1_4/PlaNet_S_A/f'+str(n3)+"_dice")
        if n2 == 1:
            torch.save(iou_fold, 'Result/step1_4/PlaNet_S_B/f'+str(n3)+"_iou")
            torch.save(dice_fold, 'Result/step1_4/PlaNet_S_B/f'+str(n3)+"_dice")
        # Progress report
        print('-------------- Progress report --------------')
        print("Fold:  " + str(n3) + "Model:  " + str(n2) +"  finished")
        print("iou @fold："+str(iou_fold))
        print("dice @fold："+str(dice_fold))
    