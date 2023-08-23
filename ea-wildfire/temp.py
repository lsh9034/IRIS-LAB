import mat73
import h5py
import numpy as np
from pre_import import *
# FULL_AUS_PATH = '/share/wildfire-2/Wildfire_Detection/East_Asia_revision2/dataset/processed/raw/'

# data = h5py.File(FULL_AUS_PATH + 'dataset_australia_org_CNN_w15.mat', 'r')
# print(data.keys())
# print(data['X_all'].shape, data['Y_all'].shape, data['info_all'].shape)



# print(1)
full_data = h5py.File('dataset_australia_org_CNN_w15.mat', 'r')
shape = full_data['X_all'].shape
data = np.zeros(shape, dtype=np.float32)
print(2)
full_data['X_all'].read_direct(data)
# print(type(data), data.shape)
# full_data.close()
# import AUS_dataset
# import share
# delete_col = share.delete_col
# aus_valset, aus_val_dataloader = AUS_dataset.load_AUS_FULL_DataLoader(delete_col=delete_col)


ea_trainset, train_dataloader = EA_dataset.EA_Train_DataLoader(delete_col=[])
ea_mean = np.mean(ea_trainset.data, axis=(0,1,2))
ea_std = np.std(ea_trainset.data, axis=(0,1,2))
print(ea_mean)
print(ea_std)

        
def normalize(data):
    for i in range(1, data.shape[1]):
        # mean = np.mean(self.data[:,:,:,i].flatten())
        # std = np.std(self.data[:,:,:,i].flatten())
        # if std==0:std=1
        # self.data[:,:,:,i] = (self.data[:,:,:,i] - mean)/std
        ma = np.nanmax(data[:,i,:,:])
        mi = np.nanmin(data[:,i,:,:])
        data[:,i,:,:] = (data[:,i,:,:] - mi) / (ma - mi)
        
        
        
    
def convert_LC(data):
    lc_data = data[:,0,:,:]
    
    idx = np.where((lc_data<6) | (lc_data==8) | (lc_data==9))
    data[idx[0], 0, idx[1], idx[2]]=2
    del idx
    
    idx = np.where((lc_data==6) | (lc_data==7) | (lc_data==10) | (lc_data==12))
    data[idx[0], 0, idx[1], idx[2]]=1
    del idx
    
    idx = np.where((lc_data==1) | (lc_data>12))
    data[idx[0], 0, idx[1], idx[2]]=0
    del idx
    return



    
def clip_one_col(data, col, ma, mi):
    idx = np.where(data[:,col,:,:]<mi)
    data[idx[0],col,idx[1],idx[2]]=mi
    del idx
    
    idx = np.where(data[:,col,:,:]>ma)
    data[idx[0],col,idx[1],idx[2]]=ma
    del idx
    gc.collect()
    return