from pre_import import *
import random

ea_valset, train_dataloader = EA_dataset.EA2_Train_DataLoader(delete_col=[], norm='Min-Max')
print(np.min(ea_valset.data[:,:,:,1:]), np.max(ea_valset.data[:,:,:,1:]))

def save_img(img, name, band_name):
    plt.title(band_name)
    plt.imshow(img)
    plt.colorbar()
    plt.savefig(name)
    plt.close()
    
band_idx = {
    'Land cover':0,
    'Skin temperature':1,
    'Relative humidity':2,
    'Solar zenith angle':3,
    'Satellite zenith angle':4,
    'Cloud':5,
    'Ch01':6,
    'Ch03':7,
    'BT07':8,
    'BT07-12':9,
    'BT07-14':10,
    'S-BT07':11,
    'S-BT07-14':12    
}
idx1 = np.where(ea_valset.label==1)
idx1 = idx1[0]
PATH='./EA_train_images/'
sampling_size = 2
rand_idx = [random.randrange(0, len(idx1)) for i in range(sampling_size)]

use_bands = ['Skin temperature', 'Relative humidity', 'Solar zenith angle', 'BT07-14', 'S-BT07', 'S-BT07-14']
for i in rand_idx:
    idx = idx1[i]
    img = ea_valset.data[idx].permute(1,2,0).numpy()
    #img = ea_valset.data[idx]
    
    for band_name in use_bands:
        image = img[:,:,band_idx[band_name]]
        save_img(image, PATH+str(idx)+'_'+band_name+'.png', band_name)
    