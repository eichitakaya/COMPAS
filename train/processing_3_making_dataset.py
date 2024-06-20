import os
import glob
import cv2
import json
import random
import torch
import torchvision
import numpy as np
import pandas as pd
import pydicom
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import KFold

ft_cases = [7084, 1020, 7139, 7080, 1049,
                    7068, 7018, 1024, 1053, 7039,
                    7023, 7030, 7118, 7114, 1056,
                    7067, 7085, 7125, 7138, 1045,
                    7099, 1041, 7035, 7100, 1011,
                    7150, 7032, 7111, 7040, 7036,
                    1051, 7167, 7121, 7053, 7149,
                    7119, 1009, 7064, 7113, 7101,
                    7010, 1003, 7016]

def list_folders(directory):
    return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
def make_kfold_set(k, ID_N, ID_P):
    kf = KFold(n_splits=k, shuffle=True)
    ID_N_TrTe = []
    ID_P_TrTe = []
    for Tr_index, Te_index in kf.split(ID_N):
        kth_Tr = []
        kth_Te = []
        for i in Tr_index:
            kth_Tr.append(ID_N[i])
        for i in Te_index:
            kth_Te.append(ID_N[i])
        ID_N_TrTe.append((kth_Tr, kth_Te))
    for Tr_index, Te_index in kf.split(ID_P):
        kth_Tr = []
        kth_Te = []
        for i in Tr_index:
            kth_Tr.append(ID_P[i])
        for i in Te_index:
            kth_Te.append(ID_P[i])
        ID_P_TrTe.append((kth_Tr, kth_Te))
    ID_PN_TrTe = []
    for i in range(k):
        Tr = ID_P_TrTe[i][0] + ID_N_TrTe[i][0]
        Te  = ID_P_TrTe[i][1] + ID_N_TrTe[i][1]
        ID_PN_TrTe.append((Tr, Te))
    return ID_PN_TrTe
def mask_imgs(A, B): #A：img B:mask
    B = B > 125
    masked_A = A*B
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

def converter_dataset2data(dataset):
    data = []
    for index in range(len(dataset)):
        img, cimg, mask, label, clinical_feature, ccc = dataset[index]
        data.append([img, cimg, mask, label, clinical_feature, ccc])
    return data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path):
        super().__init__()
        self.paths_list = path
        self.clinical_df = pd.read_csv("Data_clinical/PAS_table.csv")
        self.clinical_df = self.clinical_df.drop("PAS", axis=1)

    def __len__(self):
        return len(self.paths_list)

    def _fetch_clinical_features(self, ID):
        features = torch.Tensor(np.array(self.clinical_df[self.clinical_df["ID"] == ID].iloc[:, 1:8]))
        return features

    def __getitem__(self, index):
        preprocess  = transforms.Compose([
            transforms.RandomResizedCrop((256, 256), scale=(1.0, 1.0), ratio=(1.0, 1.0)),
            transforms.ToTensor()
        ])

        ###slic
        slic = self.paths_list[index]
        ###

        ###img
        path_dicom = slic + '/data_dicom.dcm'
        dicom = pydicom.dcmread(path_dicom)
        img   = dicom.pixel_array
        wc    = dicom.WindowCenter
        ww    = dicom.WindowWidth
        window_max = wc + ww/2                     
        window_min = wc - ww/2

        img   = 255*(img - window_min)/(window_max - window_min) 
        
        img[img > 255] = 255
        img[img < 0] = 0 
        ###
        
        ###mask
        path_mask = slic + '/data_mask.jpg'
        mask = Image.open(path_mask)
        mask = np.array(mask)
        ###
        
        ###cropped_img
        cimg = mask_imgs(img, mask)
        cimg = crop_imgs(cimg, mask)
        
        ###preprocessing
        img  = Image.fromarray(img)
        img  = preprocess(img)
        
        cimg = Image.fromarray(cimg)
        cimg = preprocess(cimg)
        
        mask = Image.fromarray(mask)
        mask = preprocess(mask)
#         mask = torch.tensor(mask.clone().detach(), dtype=torch.int)
        mask = torch.squeeze(mask)
        mask = torch.where(mask > 0.5, torch.tensor(1), torch.tensor(0))
        ###
        

        ###clinical information
        case     = slic[-6:-2]
        if slic[-6] == '7':
            label = 0
        elif slic[-6] == '1':
            label = 1
        label = torch.tensor(label)
        
        
        with open(slic + '/data_CCC_ground_truth.json', 'r', encoding='utf-8') as f:
            ccc = json.load(f)

        clinical_feature = self._fetch_clinical_features(int(case)).flatten()
        return img, cimg, mask, label, clinical_feature, ccc


folders = [int(folder) for folder in list_folders('Data_base')]
train_cases = list(set(folders) - set(ft_cases))

train_pos, train_neg = [], []
for case in train_cases:
    if case < 2000:
        train_pos.append(case)
    elif case>= 7000:
        train_neg.append(case) 
    else: print('error')
        
###checker
print(len(train_pos), len(train_neg), len(ft_cases), len(train_pos)+len(train_neg)+len(ft_cases))
####

train_folds = make_kfold_set(5, train_neg, train_pos)

###checker
p = 0
for n in range(5):
    p = p + (set(train_folds[n][0]) & set(train_folds[n][1]) == set())
    p = p + (set(train_folds[n][0]) & set(ft_cases) == set())
    p = p + (set(train_folds[n][1]) & set(ft_cases) == set())
if p == 15: 
    print('Data processing is successful')
elif p != 15: 
    print('Data is contarminated !!!')
###

for fold in range(5):
    train_slics = ['Data_base/' + str(case) +'/' + str(n) for n in range(5) for case in train_folds[fold][0]]
    test_slics  = ['Data_base/' + str(case) +'/' + str(n) for n in range(5) for case in train_folds[fold][1]]

    ###checker
    p = 0
    p = p + (set(train_slics) & set(test_slics) == set())
    if p == 1: 
        print('Data processing is successful')
    elif p != 1: 
        print('Data is contarminated !!!') 
    ###
    
    train_slics_shu = random.sample(train_slics, len(train_slics))
    test_slics_shu  = random.sample(test_slics , len(test_slics) )
    
    dataset_train = Dataset(train_slics_shu)
    dataset_test  = Dataset(test_slics_shu)
    
    data_train = converter_dataset2data(dataset_train)
    data_test  = converter_dataset2data(dataset_test)
    
    torch.save(data_train, 'Data_tensor_check/f'+str(fold)+'_train')
    torch.save(data_test , 'Data_tensor_check/f'+str(fold)+'_test' )
    
    ###checker
    p = len(dataset_train) + len(dataset_test)
    if p == (218-43)*5: 
        print('Data processing is successful')
    elif p != (218-43)*5: 
        print('Data processing is INCORRECT !!!') 
    ###
    ###Reporter
    print('Data fold:'+str(fold)+' processing is finished')

ft_slics    = ['Data_base/' + str(case) +'/' + str(n) for n in range(5) for case in ft_cases]
dataset_ft  = Dataset(ft_slics)
data_ft     = converter_dataset2data(dataset_ft)
torch.save(data_ft, 'Data_tensor_check/ft')

    
    
    




