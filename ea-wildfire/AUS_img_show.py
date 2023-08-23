from pre_import import *
import random
#aus_valset = AUS_dataset.load_info_and_label(delete_col=[])
#_, aus_val_dataloader = AUS_dataset.load_AUS_DataLoader(aus_valset, 1)

aus_valset, aus_dataloader = AUS_dataset.load_AUS_FULL_DataLoader(delete_col=[], norm='Z')
print(np.max(aus_valset.data.numpy()[:,1:,:,:]), np.min(aus_valset.data.numpy()))
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

idx1 = np.where(aus_valset.label==1)
idx1 = idx1[0]
sampling_size = 3
PATH='./AUS_images/'
rand_idx = [random.randrange(0, len(idx1)) for i in range(sampling_size)]

use_bands = ['Land cover','Skin temperature', 'Relative humidity', 'Solar zenith angle','Ch01','Ch03','BT07', 'BT07-14', 'S-BT07', 'S-BT07-14']
for i in rand_idx:
    idx = idx1[i]
    img = aus_valset.data[idx].permute(1,2,0).numpy()
    #img = aus_valset.data[idx]
    
    for band_name in use_bands:
        image = img[:,:,band_idx[band_name]]
        save_img(image, PATH+str(idx)+'_'+band_name+'.png', band_name)
    
    