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
dice_calculation = torchmetrics.classification.BinaryJaccardIndex().to(device='cuda:1') 

iou_cases = []
iou_cases_pos = []
iou_cases_neg = []
dice_cases = []


for s in tqdm(range(int(len(data_ft))), desc="Loop level A: each case", position=0, leave=False):   
    filename_1 = 'Result/step2/PlaNet_S_A_only_withoutTTA/'+str(s)
    pred_seg = torch.load(filename_1+"_p0").to(device='cuda:1') 
    t_seg      = torch.load(filename_1+"_t").to(device='cuda:1') 
    t_cls      = torch.load(filename_1+"_t_cls")

    ##### IoUを計算する　#####
    iou = iou_calculation(pred_seg, t_seg)
    iou_cases.append(iou.item())
    if t_cls == 1:
        iou_cases_pos.append(iou.item())
    elif t_cls == 0:
        iou_cases_neg.append(iou.item())
    
    dice = binary_f1_score(pred_seg.view(-1), t_seg.view(-1))
    dice_cases.append(dice.item())
iou_result_mean = torch.mean(input=torch.tensor(iou_cases))
iou_result_mean_pos = torch.mean(input=torch.tensor(iou_cases_pos))
iou_result_mean_neg = torch.mean(input=torch.tensor(iou_cases_neg))
dice_result_mean = torch.mean(input=torch.tensor(dice_cases))
torch.save(iou_cases, "Result/step3/PlaNet_S_A_only_withoutTTA/iou_cases")
# torch.save(iou_cases_pos, "PlaNet/Step4_result/iou_cases_A_only_pos")
# torch.save(iou_cases_neg, "PlaNet/Step4_result/iou_cases_A_only_neg")
torch.save(iou_result_mean, "Result/step3/PlaNet_S_A_only_withoutTTA/iou_mean")
# torch.save(iou_result_mean_pos, "PlaNet/Step4_result/iou_result_mean_A_only_pos")
# torch.save(iou_result_mean_neg, "PlaNet/Step4_result/iou_result_mean_A_only_neg")
# torch.save(dice_result_mean, "PlaNet/Step4_result/dice_result_mean_A_only")

iou_cases = []
iou_cases_pos = []
iou_cases_neg = []
dice_cases = []

for s in tqdm(range(int(len(data_ft))), desc="Loop level A: each case", position=0, leave=False):   
    filename_1 = 'Result/step2/PlaNet_S_A_only_withoutTTA/'+str(s)
    filename_2 = 'Result/step2/PlaNet_S_A_plus_B/'+str(s)
    pred_seg_1 = torch.load(filename_2+"_pA").to(device='cuda:1') 
    pred_seg_2 = torch.load(filename_2+"_pB").to(device='cuda:1') 
    t_seg      = torch.load(filename_1+"_t").to(device='cuda:1') 
    t_cls      = torch.load(filename_1+"_t_cls")

    pred_seg = torch.cat((pred_seg_1, pred_seg_2), 0)
    pred_seg   = torch.mean(input=pred_seg, dim=0).squeeze() 
    thresh = 0.01 ############################################################
    pred_seg = F.threshold(pred_seg, thresh, 0)
    pred_seg = F.threshold(-pred_seg, -thresh, 1)       
    ##### IoUを計算する　#####
    iou = iou_calculation(pred_seg, t_seg)
    iou_cases.append(iou.item())
    if t_cls == 1:
        iou_cases_pos.append(iou.item())
    elif t_cls == 0:
        iou_cases_neg.append(iou.item())

    dice = binary_f1_score(pred_seg.view(-1), t_seg.view(-1))
    dice_cases.append(dice.item())

    torch.save(pred_seg, filename_2+"_pAplusB")  
iou_result_mean = torch.mean(input=torch.tensor(iou_cases))
iou_result_mean_pos = torch.mean(input=torch.tensor(iou_cases_pos))
iou_result_mean_neg = torch.mean(input=torch.tensor(iou_cases_neg))
dice_result_mean = torch.mean(input=torch.tensor(dice_cases))
torch.save(iou_cases, "Result/step3/PlaNet_S_A_plus_B/iou_cases")
# torch.save(iou_cases_pos, "PlaNet/Step4_result/iou_cases_pos")
# torch.save(iou_cases_neg, "PlaNet/Step4_result/iou_cases_neg")
torch.save(iou_result_mean, "Result/step3/PlaNet_S_A_plus_B/iou_mean")
# torch.save(iou_result_mean_pos, "PlaNet/Step4_result/iou_result_mean_pos")
# torch.save(iou_result_mean_neg, "PlaNet/Step4_result/iou_result_mean_neg")
# torch.save(dice_result_mean, "PlaNet/Step4_result/dice_result_mean")

    
    
    