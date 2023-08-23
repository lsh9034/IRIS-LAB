from pre_import import *
def plot_band_by_class(data, class_idx):
    plt.figure(figsize=(20,20))
    band_name = ['Land cover', 'Skin temperature', 'Relative humidity', 'Solar zenith angle', 'Satellite zenith angle',
                'Cloud', 'Ch01', 'Ch03', 'BT07', 'BT07-12', 'BT07-14', 'Spatial difference of BT07', 'Spatial difference of BT07-14']
    for i in range(4):
        for j in range(4):
            plt.subplot(4,4, 4*i+j+1)
            plt.title(band_name[4*i+j])
            plt.hist(data[class_idx[0],:,:,4*i+j].flatten(), bins=100, alpha=0.5)
            plt.hist(data[class_idx[1],:,:,4*i+j].flatten(), bins=100, alpha=0.5)
            if 4*i+j == 12:break
    plt.show()
    plt.savefig('./EDA/AUS_histogram_by_EA_Filtered.png')
    plt.close()
    

# ea_trainset, train_dataloader = EA_dataset.EA_Train_DataLoader(delete_col=[])
# class_idx = [np.where(ea_trainset.label.numpy()==0), np.where(ea_trainset.label.numpy()==1)]

aus_valset, aus_dataloader = AUS_dataset.load_AUS_FULL_DataLoader(delete_col=[], norm='Z-Origin',mode='Filter')
class_idx = [np.where(aus_valset.label.numpy()==0), np.where(aus_valset.label.numpy()==1)]

#plot_band_by_class(aus_valset.data, class_idx)
plot_band_by_class(aus_valset.data.permute(0,2,3,1).numpy(), class_idx)