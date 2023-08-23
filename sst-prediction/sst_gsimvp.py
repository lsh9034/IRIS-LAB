from pre_import import *
from openstl.api import BaseExperiment
from openstl.utils import create_parser
from gsim_vp import GSimVP
import gsim_train
from Custom_CosineAnnealingWarmRestarts import CosineAnnealingWarmUpRestarts
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

hid_T = 128

def LossPenalty(pred, label):
    n = pred.shape[1]
    device = pred.get_device()
    weight = torch.tensor([1+0.1*i for i in range(0,n)], device='cuda:'+str(device), requires_grad=False)
    img_loss = (pred-label)**2
    for i in range(n):
        img_loss[:,i,:,:,:] = img_loss[:,i,:,:,:]*weight[i]
    mse_loss = torch.mean(img_loss)
    return mse_loss

def load_sst(path):
    record = torch.load(path)
    sst_v2 = GSimVP(in_shape=(T,C,H,W), hid_T=hid_T).to(device)
    sst_v2.load_state_dict(record['model_state_dict'])
    optimizer = torch.optim.Adam(sst_v2.parameters())
    optimizer.load_state_dict(record['optimizer_state_dict'])    
    return sst_v2, optimizer

predict_day = 180
batch_size=1
img_interval=1
sst_train_dataset = dataset_sst.SSTDataset(time_interval=180,img_interval=img_interval, predict_day=predict_day, mode='train')
sst_valid_dataset = dataset_sst.SSTDataset(time_interval=180, img_interval=img_interval, predict_day=predict_day, mode='valid', 
                                           dataset=sst_train_dataset.sst_data)

sst_train_dataloader = DataLoader(sst_train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
sst_valid_dataloader = DataLoader(sst_valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

T, C, H, W = sst_train_dataset[0][0].shape

gsimvp = GSimVP(in_shape=(T,C,H,W), hid_T=hid_T).to(device)
#sst_v2 = load_sst('./models/sst_v2_2_best.pkl')
optimizer = torch.optim.Adam(gsimvp.parameters(), lr=0)
loss_fn = LossPenalty

scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=100, T_mult=1, eta_max=0.01,  T_up=10, gamma=0.5)
#scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-8)
# scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,lr_lambda=lambda epoch: 0.95 ** epoch)
gsimvp_manager = gsim_train.TrainManager(loss_fn, optimizer, scheduler=scheduler)
print(1)
epoch=2000
gsim_train.train_loop(gsimvp, epoch, sst_train_dataloader, sst_valid_dataloader, loss_fn, optimizer, 
                       device, f'gsimvp_{epoch}ep_imginterval_{img_interval}_predict_180', save_best=True, scheduler=gsimvp_manager.scheduler,
                       path='./experiments/LossPenalty_GSimVP/models/')
