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
import importlib
from torchvision import transforms as transforms
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
def mask_imgs(A, B): #A：img B:mask
    B = B > 0.5
    masked_A = np.multiply(A, B)
    return masked_A

def crop_imgs(A, B): #A：img B:mask
    # maskから重心を計算
    nLabels, labelImages, D, center = cv2.connectedComponentsWithStats(B)
    D = D[1:]
    x = [d[0] for d in D]
    y = [d[1] for d in D]
    w = [d[2] for d in D]
    h = [d[3] for d in D]
    xmax = max([a + b for a, b in zip(x, w)])
    xmin = min(x)
    ymax = max([a + b for a, b in zip(y, h)])
    ymin = min(y)
    w = xmax - xmin
    h = ymax - ymin
    l = max([w,h])
    cropped_A = Image.fromarray(A).crop((xmin+w/2-l/2, ymin+h/2-l/2, xmin+w/2+l/2, ymin+h/2+l/2)).resize((256,256))
    cropped_A = np.array(cropped_A)
    return cropped_A
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
data_ft = torch.load('Data_tensor_check/ft')
Dataset_ft    = Dataset(d=data_ft)


Largest_auc_fold = 1

import Model.MS2_01031_8 as PlaNet_C
importlib.reload(PlaNet_C)
net = PlaNet_C.Net()
net.load_state_dict(torch.load('Result/step5/PlaNet_Ccheck-PT-F1.pth'))   

net.eval()
auc_calculation = BinaryAUROC()
preds, ts = [], []
for s in tqdm(range(int(len(data_ft))), desc="Loop level A: each case", position=0, leave=False):
    x, cx, t_seg, t_cls, clinical_feature, ccc= Dataset_ft[s]
    if s == 0:print(x.max(), cx.max())
    filename = 'Result/step7_2_check-PT-F1-cx/'+str(s)
    torch.save(x, filename+"_r")
    x = x.squeeze(0).detach().cpu().numpy()
    pred_seg = torch.load("Result/step2/PlaNet_S_A_plus_B/"+str(s)+"_pAplusB").detach().cpu().numpy().astype(np.uint8)
    cimgs = mask_imgs(x, pred_seg)
    torch.save(cx, filename+"_cx")
    torch.save(cimgs, filename+"_af_mask")
    torch.save(pred_seg, filename+"_pred_seg")
    cimgs = crop_imgs(cimgs, pred_seg) 
    torch.save(cimgs, filename+"_af_crop")
    cimgs = np.where(cimgs > 0.1, cimgs, torch.tensor(0.0))

    cimgs = torch.from_numpy(cimgs).to(torch.float32)  
    torch.save(t_seg, filename+"_t_seg")
    torch.save(cimgs, filename+"_cimgs")      
#     pred_cls = net(cimgs.unsqueeze(0).unsqueeze(0), clinical_feature.unsqueeze(0))  
    pred_cls = net(cx.unsqueeze(0), clinical_feature.unsqueeze(0))  
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
torch.save(AUC, "Result/step7_2_check-PT-F1-cx/AUC")
torch.save([fpr, tpr, thresholds], "Result/step7_2_check-PT-F1-cx/ROC")
torch.save(pred, "Result/step7_2_check-PT-F1-cx/pred")
torch.save(t, "Result/step7_2_check-PT-F1-cx/t")
    