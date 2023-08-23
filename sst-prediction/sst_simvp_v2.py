from pre_import import *
from openstl.api import BaseExperiment
from openstl.utils import create_parser
from openstl.models import SimVP_Model
import model_train

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'


def load_sst(path):
    record = torch.load(path)
    sst_v2 = SimVP_Model(in_shape=(T,C,H,W), hid_T=64).to(device)
    sst_v2.load_state_dict(record['model_state_dict'])
    optimizer = torch.optim.Adam(sst_v2.parameters())
    optimizer.load_state_dict(record['optimizer_state_dict'])    
    return sst_v2, optimizer
predict_day = 7
batch_size=16

sst_train_dataset = dataset_sst.SSTDataset(time_interval=3,img_interval=2, predict_day=predict_day, mode='train')
sst_valid_dataset = dataset_sst.SSTDataset(time_interval=3, img_interval=2, predict_day=predict_day, mode='valid', 
                                           dataset=sst_train_dataset.sst_data)

sst_train_dataloader = DataLoader(sst_train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
sst_valid_dataloader = DataLoader(sst_valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

T, C, H, W = sst_train_dataset[0][0].shape

sst_v2 = SimVP_Model(in_shape=(T,C,H,W), hid_T=64).to(device)
#sst_v2 = load_sst('./models/sst_v2_2_best.pkl')
optimizer = torch.optim.Adam(sst_v2.parameters(), weight_decay=1e-4, lr=1e-4)
loss_fn = torch.nn.MSELoss()
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
#scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,lr_lambda=lambda epoch: 0.97 ** epoch)
sst_v1_manager = model_train.TrainManager(loss_fn, optimizer, scheduler=scheduler)
print(1)
model_train.train_loop(sst_v2, 400, sst_train_dataloader, sst_valid_dataloader, loss_fn, optimizer, 
                       device, 'sst_v2_400ep_imginterval_2', save_best=True, scheduler=sst_v1_manager.scheduler)

# print(_, C, H, W)
# custom_training_config = {
#     'pre_seq_length': predict_day,
#     'aft_seq_length': predict_day,
#     'total_length': predict_day + predict_day,
#     'batch_size': batch_size,
#     'val_batch_size': batch_size,
#     'epoch': 100,
#     'lr': 1e-3,
#     'metrics': ['mse', 'mae'],

#     'ex_name': 'custom_exp',
#     'dataname': 'custom',
#     'in_shape': [predict_day, C, H, W],
#     'sched': 'cosine'
# }

# custom_model_config = {
#     # For MetaVP models, the most important hyperparameters are:
#     # N_S, N_T, hid_S, hid_T, model_type
#     'method': 'SimVP',
#     # Users can either using a config file or directly set these hyperparameters
#     # 'config_file': 'configs/custom/example_model.py',

#     # Here, we directly set these parameters
#     'model_type': 'gSTA',
#     'N_S': 4,
#     'N_T': 4,
#     'hid_S': 32,
#     'hid_T': 256
# }

# args = create_parser().parse_args([])
# config = args.__dict__

# # update the training config
# config.update(custom_training_config)
# # update the model config
# config.update(custom_model_config)
# print(config)
# exp = BaseExperiment(args, dataloaders=(sst_train_dataloader, sst_valid_dataloader , sst_valid_dataloader))


# print('>'*35 + ' training ' + '<'*35)
# exp.train()

# print('>'*35 + ' testing  ' + '<'*35)
# exp.test()