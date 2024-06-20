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
batch_size = 8
num_worker = 1
epoch_number = 24

data_train = torch.load('Data_tensor_check/f'+str(0)+'_train')
data_test = torch.load('Data_tensor_check/f'+str(0)+'_test')
data_1fold = data_train + data_test
print(len(data_train), len(data_test), len(data_1fold))

Dataset_1fold   = Dataset(d = data_1fold) 
DataLoader_1fold= DataLoader(Dataset_1fold  , batch_size=batch_size, num_workers=num_worker, pin_memory=True, shuffle=True )

import Model.MS2_01031_8 as PlaNet_C
importlib.reload(PlaNet_C)
net = PlaNet_C.Net().to(device='cuda:1')
trainer = pl.Trainer(max_epochs= epoch_number, log_every_n_steps= 1, devices= [1],accelerator="gpu")
trainer.fit(net, DataLoader_1fold)
torch.save(net.state_dict(), 'Result/step5/PlaNet_Ccheck-PT-F1.pth')                                   

    
