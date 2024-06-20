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
import Function.F_DatasetMaker_tougou as f
import importlib
importlib.reload(f)
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
    index = B == 0
    index = ~index
    masked_A = A*index
    return masked_A

def crop_imgs_ver2(A, B): #A：img B:mask
    nLabels, labelImages, D, center = cv2.connectedComponentsWithStats(B)
    D_objs = D[1:]
    D_objs2 = np.ndarray([nLabels-1, 4])
    for n in range(nLabels-1):
        D_objs2[n] = [D_objs[n][0], D_objs[n][1], D_objs[n][0]+D_objs[n][2], D_objs[n][1]+D_objs[n][3]]
    w_min = np.min(D_objs2[:,0])
    h_min = np.min(D_objs2[:,1])
    w_max = np.max(D_objs2[:,2])
    h_max = np.max(D_objs2[:,3])
    w_cen, h_cen = (w_max + w_min)/2, (h_max + h_min)/2 
    hen = int(max(w_max - w_min, h_max - h_min)/2)
    cropped_A = Image.fromarray(A).crop((w_cen-hen-1, h_cen-hen-1, w_cen+hen+1, h_cen+hen+1))
    transform  = transforms.Compose([transforms.RandomResizedCrop((256, 256), scale=(1.0, 1.0), ratio=(1.0, 1.0)),])
    cropped_A = transform(cropped_A)
    cropped_A = np.array(cropped_A)
    return cropped_A
class Dataset(torch.utils.data.Dataset):
    def __init__(self, TrainValTest, d):
        self.data = d
        if TrainValTest   == "train":
          self.index1 = 0
        elif TrainValTest == "test":
          self.index1 = 1
        elif TrainValTest == "final_test":
          self.index1 = 0   
        else:
          print('エラーです')
    def __len__(self):
        return len(self.data[self.index1])
    def __getitem__(self, index):        
        img, cimg, mask, label, clinical_feature = self.data[self.index1][index]
        return img, cimg, mask, label, clinical_feature
##########################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
  
K = 5
DS = 'auto_vs2_0' 
DS_fold = "DS_"+ DS +"_ft"
data = torch.load('Datasets_0519/'+DS_fold)
Dataset_final_test    = Dataset(TrainValTest="final_test" , d=data  ) 

import ModelSeries.MS2_01031_2 as PlaNet_S_R
importlib.reload(PlaNet_S_R)
net = PlaNet_S_R.Net()
net.load_state_dict(torch.load('PlaNet/Step6_result/PlaNet_S_R_DS_auto_vs2_0_'+ str(3) + '.pth'))   ####################################### 
net.eval()
auc_calculation = BinaryAUROC()
preds, ts = [], []
for s in tqdm(range(int(len(data[0]))), desc="Loop level A: each case", position=0, leave=False):
    x, cx, t_seg, t_cls, clinical_feature= Dataset_final_test[s]
    x = torch.load("PlaNet/Step4_result/PlaNet_S_A_ONLY_DS_auto_vs2_0_ft_"+str(s)+"_r").squeeze(0).detach().cpu().numpy()
    pred_seg = torch.load("PlaNet/Step4_result/PlaNet_S_A_ONLY_DS_auto_vs2_0_ft_"+str(s)+"_p0").detach().cpu().numpy().astype(np.uint8)
    x = mask_imgs(x, pred_seg)                
    x = crop_imgs_ver2(x, pred_seg) 
    x = torch.from_numpy(x).to(torch.float32)  
    
    pred_cls = net(x.unsqueeze(0).unsqueeze(0))  
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
torch.save(AUC, "PlaNet/Step8_4_result/AUC")
torch.save([fpr, tpr, thresholds], "PlaNet/Step8_4_result/ROC")
torch.save(pred, "PlaNet/Step8_4_result/pred")
torch.save(t, "PlaNet/Step8_4_result/t")
    