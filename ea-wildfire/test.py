from pre_import import *

record = torch.load('EfficientNet-8bands.pkl')
print(record.keys())
efficientnet = efficient_net.EfficientNet_Forestfire(8)
print(efficientnet)
efficientnet.load_state_dict(record['model_state_dict'])


import scipy
mat_file = scipy.io.loadmat('/share/wildfire-2/Wildfire_Detection/East_Asia_revision2/dataset/processed/processed/dataset_AUS_X_tr_CNN_1')
data = mat_file['X_tr_half1']

import AUS_dataset
aus_valset = AUS_dataset.load_info_and_label(delete_col=[])
_, aus_val_dataloader = AUS_dataset.load_AUS_DataLoader(aus_valset, 1)
_, aus_val_dataloader = AUS_dataset.load_AUS_DataLoader(aus_valset, 2)


efficientnet = efficient_net.EfficientNet_Forestfire(9)
print(efficientnet)



from copy import deepcopy
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from scipy import stats
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from copy import deepcopy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import sklearn
import torchvision
import albumentations
import albumentations.pytorch
class ResNet18_Forestfire(nn.Module):
    def __init__(self, input_features, resnet_out=64):
        super(ResNet18_Forestfire, self).__init__()
        self.resnet18 = torchvision.models.resnet18(weights='DEFAULT')
        normalized_weight = self.resnet18.conv1
        self.resnet18.conv1 = nn.Conv2d(input_features, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1,1), bias=False)
        self.resnet18.fc = nn.Linear(512, resnet_out, bias=True)
        # self.metainfo_layers = nn.Sequential(
        #     nn.Linear(4, 16),
        #     nn.ReLU(),
        #     nn.Linear(16, 32),
        #     nn.BatchNorm1d(32),
        #     nn.ReLU(),
        #     nn.Linear(32,16),
        #     nn.BatchNorm1d(16),
        #     nn.ReLU(),
        # )
        self.last_layers = nn.Sequential(
            nn.Linear(resnet_out, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
    def forward(self, meta_info, img):
        resnet_out = self.resnet18(img)
        #metainfo_out = self.metainfo_layers(meta_info)
        #features = torch.cat([resnet_out, metainfo_out], dim=-1)
        out = self.last_layers(resnet_out)
        return out
    

def load_ResNet18(path, band_size, device):
    record = torch.load(path)
    model = ResNet18_Forestfire(band_size)
    model.load_state_dict(record['model_state_dict'])
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(record['optimizer_state_dict'])
    return model, optimizer