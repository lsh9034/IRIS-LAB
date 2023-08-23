from pre_import import *
from openstl.api import BaseExperiment
from openstl.utils import create_parser
from conv_lstm import SST_ConvLSTM
import model_train

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

hid_T = 128

def load_sst(path):
    record = torch.load(path)
    sst_v2 = sst_convlstm().to(device)
    sst_v2.load_state_dict(record['model_state_dict'])
    optimizer = torch.optim.Adam(sst_v2.parameters())
    optimizer.load_state_dict(record['optimizer_state_dict'])    
    return sst_v2, optimizer

predict_day = 7
batch_size=2

sst_train_dataset = dataset_sst.SSTDataset(time_interval=3,img_interval=1,data_label_gap=1, predict_day=predict_day, mode='train')
sst_valid_dataset = dataset_sst.SSTDataset(time_interval=3, img_interval=1, data_label_gap=1, predict_day=predict_day, mode='valid', 
                                           dataset=sst_train_dataset.sst_data)

sst_train_dataloader = DataLoader(sst_train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
sst_valid_dataloader = DataLoader(sst_valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

T, C, H, W = sst_train_dataset[0][0].shape

sst_convlstm = SST_ConvLSTM().to(device)
#sst_v2 = load_sst('./models/sst_v2_2_best.pkl')
optimizer = torch.optim.Adam(sst_convlstm.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-5)
# scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,lr_lambda=lambda epoch: 0.95 ** epoch)
sst_convlstm_manager = model_train.TrainManager(loss_fn, optimizer, scheduler=scheduler)
print(1)
epoch=40
model_train.train_loop(sst_convlstm, epoch, sst_train_dataloader, sst_valid_dataloader, loss_fn, optimizer, 
                       device, f'gsimvp_{epoch}ep_imginterval_1', save_best=True, scheduler=sst_convlstm_manager.scheduler)

