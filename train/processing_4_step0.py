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
epoch_number = 100

for n3 in range(K):
    data_train = torch.load('Data_tensor/f'+str(n3)+'_train')
    Dataset_train   = Dataset(d = data_train) 
    DataLoader_train= DataLoader(Dataset_train  , batch_size=batch_size, num_workers=num_worker, pin_memory=True, shuffle=True )
    
    for n2 in range(2):
        if n2 == 0:
            import Model.MS2_00000_1 as PlaNet_S_A
            importlib.reload(PlaNet_S_A)
            net = PlaNet_S_A.Net().to(device='cuda:0')
            trainer = pl.Trainer(max_epochs= epoch_number, log_every_n_steps= 1, devices= [0],accelerator="gpu")
            trainer.fit(net, DataLoader_train)
            torch.save(net.state_dict(), 'Result/step0/PlaNet_S_A_'+ str(n3) + '.pth')                
        if n2 == 1:
            import Model.MS2_04011_2 as PlaNet_S_B
            importlib.reload(PlaNet_S_B)
            net = PlaNet_S_B.Net().to(device='cuda:0')
            trainer = pl.Trainer(max_epochs= epoch_number, log_every_n_steps= 1, devices= [0],accelerator="gpu")
            trainer.fit(net, DataLoader_train)
            torch.save(net.state_dict(), 'Result/step0/PlaNet_S_B_'+ str(n3) + '.pth')                

    
