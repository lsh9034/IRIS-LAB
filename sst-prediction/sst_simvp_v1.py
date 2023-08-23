import sys
sys.path.append('./SimVP_v1/')


from pre_import import *
from openstl.api import BaseExperiment
from openstl.utils import create_parser
from SimVP_v1.model import SimVP
import torchsummary
import model_train

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

predict_day = 7
batch_size=256

sst_train_dataset = dataset_sst.SSTDataset(time_interval=3, predict_day=predict_day, mode='train')
sst_valid_dataset = dataset_sst.SSTDataset(time_interval=3, predict_day=predict_day, mode='valid')

sst_train_dataloader = DataLoader(sst_train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
sst_valid_dataloader = DataLoader(sst_valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)


T, C, H, W = sst_train_dataset[0][0].shape
sst_v1 = SimVP((T,C,H,W),hid_T=128, N_T=4).to(device)
torchsummary.summary(sst_v1, input_size=(T,C,H,W))

optimizer = torch.optim.Adam(sst_v1.parameters(), weight_decay=1e-4)
loss_fn = torch.nn.MSELoss()

sst_v1_manager = model_train.TrainManager(loss_fn, optimizer)

model_train.train_loop(sst_v1, 10, sst_train_dataloader, sst_valid_dataloader, loss_fn, optimizer, 
                       device, 'sst_v1_1', save_best=True)

