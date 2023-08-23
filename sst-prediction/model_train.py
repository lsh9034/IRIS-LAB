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
from tqdm import tqdm

def one_epoch(model, train_dataloader, loss_fn, optimizer, device, model_name):
    sum_loss=0
    acc=0
    for batch, (data, label) in tqdm(enumerate(train_dataloader), leave=True):
        data = data.to(device)
        label = label.to(device)
        #print(label, data)
        output = model(data)
        #print(output.shape, label.shape)
        loss = loss_fn(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #if(batch%5==0):print(f'{batch} : {loss.item()}')
        sum_loss+=loss.item()
       
        #print(a)
        #print(label)
    return sum_loss/len(train_dataloader)

def train_loop(model, epochs, train_dataloader, valid_dataloader, loss_fn, optimizer, device, 
               model_name="model", save_best=False, scheduler=None, path='./models/'):
    min_loss=1e10
    best_model=None
    best_optim=None
    for i in range(1,epochs+1):
        if scheduler!=None: print(f'{i} epoch lr: {scheduler.get_last_lr()}')
        train_loss = one_epoch(model, train_dataloader,loss_fn, optimizer, device, model_name)
        if scheduler!=None:
            scheduler.step()
        val_loss=0
        model.eval()
        with torch.no_grad():
            for batch, (data, label) in tqdm(enumerate(valid_dataloader), leave=True):
                data = data.to(device)
                label = label.to(device)
                output = model(data)
                a=loss_fn(output,label).item()
                #print(output, label)
                val_loss+=a
            
            val_loss/=len(valid_dataloader)
            print(f'{i} epoch val loss: {val_loss} / train loss: {train_loss}')
            
        model.train()
        if save_best and val_loss<min_loss:
            best_model = deepcopy(model.state_dict())
            best_optim = deepcopy(optimizer.state_dict())
            min_loss=val_loss
            
    if save_best: torch.save({
        'model_state_dict': best_model,
        'optimizer_state_dict': best_optim,
        }, './models/'+model_name+'_best.pkl')
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, './models/'+model_name+'.pkl')
    print(min_loss)

class TrainManager:
    def __init__(self, loss_fn, optimizer, scheduler=None):
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        if scheduler!=None: self.scheduler=scheduler
    
