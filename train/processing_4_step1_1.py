'''
Library import
'''
import pickle
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

iou_PlaNet_S_A = []
dice_PlaNet_S_A = []
iou_PlaNet_S_B = []
dice_PlaNet_S_B = []

for n3 in range(5):
    data_test = torch.load('Data_tensor/f'+str(n3)+'_test')
    Dataset_test   = Dataset(d = data_test) 
    iou_calculation = torchmetrics.classification.BinaryJaccardIndex().to(device='cuda:0') 
    for n2 in range(2):
        iou_case = []
        dice_case = []
        if n2 == 0:
            import Model.MS2_00000_1 as PlaNet_S_A
            importlib.reload(PlaNet_S_A)
            net = PlaNet_S_A.Net().to(device='cuda:0')  
            net.load_state_dict(torch.load('Result/step0/PlaNet_S_A_'+ str(n3) + '.pth'))   
            net.eval()
        elif n2 == 1:
            import Model.MS2_04011_2 as PlaNet_S_B
            importlib.reload(PlaNet_S_B)
            net = PlaNet_S_B.Net().to(device='cuda:0')  
            net.load_state_dict(torch.load('Result/step0/PlaNet_S_B_'+ str(n3) + '.pth'))  
            net.eval()            

        for s in tqdm(range(int(len(data_test))), desc="Loop level A: each case", position=0, leave=False):
            x, cx, t_seg, t_cls, clinical_feature, ccc= Dataset_test[s]
            x = x.to(device='cuda:0')  
            t_seg = t_seg.to(device='cuda:0')  
            pred_seg = net(x.unsqueeze(0))  
            if n2 == 1:
                pred_seg = F.interpolate(pred_seg, size=None, scale_factor=4, mode='bilinear')
            pred_seg = pred_seg.squeeze()
            t_seg = t_seg.to(torch.float32)

            ##### Criteriaを計算する　#####
            iou = iou_calculation(pred_seg, t_seg)
            iou_case.append(iou.item())

            dice = binary_f1_score(pred_seg.view(-1), t_seg.view(-1))
            dice_case.append(dice.item())

        iou_fold = torch.mean(input=torch.tensor(iou_case))
        dice_fold = torch.mean(input=torch.tensor(dice_case))

        if n2 == 0:
            iou_PlaNet_S_A.append(iou_fold.detach())
            dice_PlaNet_S_A.append(dice_fold.detach())
        if n2 == 1:
            iou_PlaNet_S_B.append(iou_fold.detach())
            dice_PlaNet_S_B.append(dice_fold.detach())
        # Progress report
        print('-------------- Progress report --------------')
        print("Fold:  " + str(n3) + "Model:  " + str(n2) +"  finished")
        print("iou @fold："+str(iou_fold))
        print("dice @fold："+str(dice_fold))
print('-------------- Progress report --------------')
print("Step1-1  finished")
print("iou_PlaNet_S_A： "+str(iou_PlaNet_S_A))
print("iou_PlaNet_S_B： "+str(iou_PlaNet_S_B))
print("dice_PlaNet_S_A： "+str(dice_PlaNet_S_A))
print("dice_PlaNet_S_B： "+str(dice_PlaNet_S_B))

with open('Result/step1_1/iou_PlaNet_S_A.pkl', 'wb') as f:
    pickle.dump(iou_PlaNet_S_A, f)
with open('Result/step1_1/iou_PlaNet_S_B.pkl', 'wb') as f:
    pickle.dump(iou_PlaNet_S_B, f)
    