from pre_import import *
from openstl.api import BaseExperiment
from openstl.utils import create_parser
from gsim_vp import GSimVP
import gsim_train
import torchvision
import torch
from MovingMNIST import movingMNIST
import model_train
from openstl.models import SimVP_Model
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

hid_T = 128

def load_sst(path):
    record = torch.load(path)
    sst_v2 = GSimVP(in_shape=(T,C,H,W), hid_T=hid_T).to(device)
    sst_v2.load_state_dict(record['model_state_dict'])
    optimizer = torch.optim.Adam(sst_v2.parameters())
    optimizer.load_state_dict(record['optimizer_state_dict'])    
    return sst_v2, optimizer

predict_day = 7
batch_size=100
img_interval=1

train_set = movingMNIST.MovingMNIST(root='.data/mnist', train=True, download=True)

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=100,
                 shuffle=True)


T, C, H, W = 10,1,64,64

sst_v2 = SimVP_Model(in_shape=(T,C,H,W), hid_T=hid_T).to(device)
#sst_v2 = load_sst('./models/sst_v2_2_best.pkl')
optimizer = torch.optim.Adam(sst_v2.parameters())
loss_fn = torch.nn.MSELoss()
scheduler = None #optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
# scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,lr_lambda=lambda epoch: 0.95 ** epoch)
sst_v1_manager = model_train.TrainManager(loss_fn, optimizer, scheduler=scheduler)
print(1)
epoch=20
model_train.train_loop(sst_v2, epoch, train_loader, train_loader, loss_fn, optimizer, 
                       device, f'mnist_simvp_test', save_best=True, scheduler=None,
                       path='./models/')


