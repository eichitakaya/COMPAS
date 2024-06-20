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


##########################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
    
for n3 in range(5):
    data_test = torch.load('Data_tensor/f'+str(n3)+'_test')
    for n2 in range(2):
        pred_tar = []        
        if n2 == 0:
            stem = 'Result/step1_2/PlaNet_S_A/f'+str(n3)+"_"
        elif n2 == 1:
            stem = 'Result/step1_2/PlaNet_S_B/f'+str(n3)+"_"
        for s in tqdm(range(int(len(data_test))), desc="Loop level A: each case", position=0, leave=False):
            pred_seg = torch.load(stem + str(s) + "_p")
            t_seg    = torch.load(stem + str(s) + "_t")
            pred_tar.append([pred_seg, t_seg])
        if n2 == 0:
            torch.save(pred_tar, 'Result/step1_3/PlaNet_S_A/f'+str(n3)+"_pred_tar")
        elif n2 == 1:
            torch.save(pred_tar, 'Result/step1_3/PlaNet_S_B/f'+str(n3)+"_pred_tar")

    