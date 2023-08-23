from pre_import import *
from gsim_vp import Gradient_img

a = torch.zeros((16,7,2,200,200))
g = Gradient_img()


sst_valid_dataset = dataset_sst.SSTDataset(3, predict_day=7 ,img_interval=7, mode='valid')
data, label = sst_valid_dataset[0]