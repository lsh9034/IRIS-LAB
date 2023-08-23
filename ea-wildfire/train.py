from copy import deepcopy
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from scipy import stats
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
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

def one_epoch(model, train_dataloader, loss_fn, optimizer, device):
    sum_loss=0
    acc=0
    for batch, (meta_info, data, label) in enumerate(train_dataloader):
        label = label.view((-1,1))
        meta_info = meta_info.to(device)
        data = data.to(device)
        label = label.to(device)
        #print(label, data)
        output = model(meta_info, data)
        #print(output.shape, label.shape)
        loss = loss_fn(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #if(batch%5==0):print(f'{batch} : {loss.item()}')
        sum_loss+=loss.item()
       
        a = torch.round(torch.sigmoid(output))
        #print(a)
        #print(label)
        acc += torch.sum(a==label).detach().cpu().numpy()
    print(acc, len(train_dataloader.dataset))
    return sum_loss/len(train_dataloader), acc/ len(train_dataloader.dataset)

def train_loop(model, epochs, train_dataloader, valid_dataloader, loss_fn, optimizer, device, 
               model_name="model", save_best=False, scheduler=None):
    ma_acc=0
    best_model=None
    for i in range(1,epochs+1):
        train_loss, train_acc = one_epoch(model, train_dataloader,loss_fn, optimizer, device)
        if scheduler!=None:
            scheduler.step()
        val_loss=0
        acc=0
        model.eval()
        with torch.no_grad():
            for meta_info, data, label in valid_dataloader:
                label = label.view((-1,1))
                meta_info = meta_info.to(device)
                data = data.to(device)
                label = label.to(device)
                output = model(meta_info, data)
                a=loss_fn(output,label).item()
                #print(output, label)
                val_loss+=a
                a = torch.round(torch.sigmoid(output))
                acc += torch.sum(a==label).detach().cpu().numpy()
            
            val_loss/=len(valid_dataloader)
            acc/=len(valid_dataloader.dataset)
            print(f'{i} epoch val loss: {val_loss} / train loss: {train_loss} / train acc: {train_acc*100}% / val acc: {acc*100}')
        model.train()
        if save_best and acc>ma_acc:
            best_model = deepcopy(model.state_dict())
            ma_acc=acc
    if save_best: torch.save({
        'model_state_dict': best_model,
        'optimizer_state_dict': optimizer.state_dict(),
        }, './models/'+model_name+'_best.pkl')
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, './models/'+model_name+'.pkl')
    print(ma_acc)

class TrainManager:
    def __init__(self, loss_fn, optimizer, scheduler=None):
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        if scheduler!=None: self.scheduler=scheduler
    
