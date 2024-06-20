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
DS = 'auto_vs2_0'  
DS_fold = "DS_"+ DS +"_ft"
data = torch.load('Datasets_0519/'+DS_fold)
Dataset_final_test    = Dataset(TrainValTest="final_test" , d=data  ) 

import ModelSeries.MS2_01031_2 as PlaNet_S_R
importlib.reload(PlaNet_S_R)
net = PlaNet_S_R.Net().to(device='cuda:0')  
net.load_state_dict(torch.load('PlaNet/Step6_result/PlaNet_S_R_DS_auto_vs2_0_'+ str(3) + '.pth'))   #######################################
net.eval()
auc_calculation = BinaryAUROC()
preds, ts = [], []

for s in tqdm(range(int(len(data[0]))), desc="Loop level A: each case", position=0, leave=False):
    x, cx, t_seg, t_cls, clinical_feature= Dataset_final_test[s]
    cx = cx.to(device='cuda:0')  
    t_cls = t_cls.to(device='cuda:0')  
    pred_cls = net(cx.unsqueeze(0))  
    pred_cls = pred_cls.squeeze(0)
    t_cls = t_cls.unsqueeze(0)
    preds.append(pred_cls.detach())
    ts.append(t_cls.detach()) 

pred = torch.cat(preds)
t = torch.cat(ts).to(torch.int32)

ROC_plotting = BinaryROC()
fpr, tpr, thresholds = ROC_plotting(pred.sigmoid(), t)
AUC = auc_calculation(pred.sigmoid(), torch.tensor(t))
# Progress report
print('-------------- Progress report --------------')
print("AUC @fold："+str(AUC))
torch.save(AUC, "PlaNet/Step8_1_result/AUC")
torch.save([fpr, tpr, thresholds], "PlaNet/Step8_1_result/ROC")
torch.save(pred, "PlaNet/Step8_1_result/pred")
torch.save(t, "PlaNet/Step8_1_result/t")
    