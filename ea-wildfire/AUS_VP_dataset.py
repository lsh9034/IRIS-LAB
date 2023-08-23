from pre_import import *

class AUS_VP(Dataset):
    def __init__(self, Y_path, Y_name, delete_col=[], transform=None, norm='Z', mode='Full'):
        
        '''Setting preinformation'''
        self.real_columns = [1,2,3,4,6,7,8,9,10,11,12]
        self.columns_name = ['Land cover', 'Skin temperature', 'Relative humidity', 'Solar zenith angle', 'Satellite zenith angle',
                             'Cloud', 'Ch01', 'Ch03', 'BT07', 'BT07-12', 'BT07-14',
                             'Spatial difference of BT07', 'Spatial difference of BT07-14']
        self.info_columns_name = ['IsVIIRS', 'Site Number', 'Case number', 
                                  'Year', 'Month', 'Day', 'Hour', 'Minute', 'Row', 'Col']
        
        self.EA_means = [0, 288.7456166679585, 37.37278846096888, 66.93127712119069, 64.7637801858399, 0.0, 0.11976039459748777,
                         0.07956394371219357, 289.9017685018098, 42.99139928717568, 4.762841464437443, 0.059951247792144766, 0.05985535932961418]
        
        self.EA_stds = [0, 11.345074383967875, 18.66352533934532, 33.78579003929686, 10.160512089023387, 1, 0.08646598352092276, 
                        0.06163847135085298, 12.092403272325093, 10.231124117777917, 4.811568768671124, 1.7408100784129035, 1.6806614843709322]
        
        self.AUS_means = [3.3775764, 297.4247, 49.818, 87.99783, 34.91156, 0.057588343, -0.10782527, -0.08343556, 296.3449,
                          27.592978, 3.1561363, 0.022168716, 0.036312718]
        
        self.AUS_stds = [1.2916965, 9.193288, 21.75271, 83.5831, 73.40764, 0.23321895, 72.70323, 64.51333, 15.720863, 47.71502,
                         46.323944, 10.58307, 18.452198]
        
        self.delete_col = delete_col
        self.transform = transform
        self.norm=norm
        self.mode=mode
        self.nan_idx = np.load('AUS_nan-idx.npy')
        '''Handling AUS Data'''
        if mode=='Full':
            real_data = self.get_full(delete_col=delete_col)
        elif mode=='Filter':
            real_data = self.get_filtered(delete_col=delete_col)
                    
        self.data = torch.tensor(real_data, dtype=torch.float32)
        del real_data
        gc.collect()
        
        '''Handling label data'''
        mat_file = scipy.io.loadmat(Y_path)
        self.label = torch.tensor(mat_file[Y_name].flatten(), dtype=torch.float32)

        '''filtering label, meta_info'''
        if mode=='Filter':
            self.label = np.delete(self.label, self.nan_idx, axis=0)
         
    def get_nanidx(self, data):
        nan_idx = np.where(np.isnan(data))[0]
        nan_idx = np.array(list(set(nan_idx)), dtype=np.int32)
        nan_idx.sort()
        return nan_idx
    
    def imputation(self, data):
        median = np.nanmedian(data, axis=(0,2,3))
        
        for i in range(0, data.shape[1]):
            idx = np.where(np.isnan(data[:,i,:,:]))
            data[idx[0],i,idx[1],idx[2]]=median[i]
            del idx
            
    
    def convert_LC(self, data):
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
    
    def normalize(self,data):
        if self.norm=='Z' or self.norm=='Min-Max':
            # mean = self.AUS_means
            # #median = np.nanmedian(data, axis=(0,1,2))
            # std = self.AUS_stds
            print('Z and Min-Max norm shouldn''t call normalize function')
            exit(0)

        elif self.norm=='Z-Origin':
            mean = self.EA_means
            std = self.EA_stds 
        
        #Land cover isn't normalized
        start = 1
        if 0 in self.delete_col:
            start=0
            
        for i in range(start, data.shape[1]): #0번은 LC이기 때문에 Normalize하지 않음 만약 delete_col에 0이 있다면 수정해야함.
            if std[i]==0:std[i]=0.1
            data[:,i,:,:] = (data[:,i,:,:]-mean[i])/std[i]
        return
    
    def get_filtered(self, delete_col):
        print('start load')
        if self.norm=='Z':
            data = np.load('AUS_FILTEREDDATA_Z-norm.npy')
        elif self.norm=='Z-Origin':
            data = np.load('AUS_FULLDATA.npy')
        print('end load')
        
        if self.norm=='Z':
            real_data = np.delete(data, delete_col, axis=1)
        if self.norm=='Z-Origin':
            self.normalize(data)
            real_data = np.delete(data, self.nan_idx, axis=0)
            real_data = np.delete(real_data, self.delete_col, axis=1)
        del data
        return real_data
    
    def get_full(self, delete_col):
        print('start load')
        if self.norm=='Z':
            data = np.load('AUS_FULLDATA_Z-norm.npy')
        elif self.norm=='Min-Max':
            data = np.load('AUS_FULLDATA_min-max-norm.npy')
        elif self.norm=='Z-Origin':
            data = np.load('AUS_FULLDATA.npy')
        print('end load')
        
        '''imputation & normalizing & clipping'''
        real_data=None
        if self.norm=='Z' or self.norm=='Min-Max':
            #self.clip()
            real_data = np.delete(data, delete_col, axis=1)
            del data
            self.imputation(real_data)
        elif self.norm=='Z-Origin':
            self.clip_one_col(data, 3, 0, 180)
            self.clip_one_col(data, 4, 0, 90)
            self.normalize(data)
            self.imputation(data)
            real_data = np.delete(data, delete_col, axis=1)
            del data
        
        self.clip(real_data)
        return real_data
    
    def clip(self, data):
        if self.norm=='Min-Max':
            print('Min-Max norm shouldn''t call clip function')
            exit(0)
        elif self.norm=='Z':
            mean = self.AUS_means
            std = self.AUS_stds
        elif self.norm=='Z-Origin':
            mean = self.EA_means
            std = self.EA_stds
                
        start = 1
        if 0 in self.delete_col:
            start=0
            
        mi = -4
        ma = 4
        idx = np.where(data<mi)
        data[idx]=mi
        del idx
        gc.collect()
        
        idx = np.where(data>ma)
        data[idx]=ma
        return
    
    def clip_one_col(self, data, col, ma, mi):
        idx = np.where(data[:,col,:,:]<mi)
        data[idx[0],col,idx[1],idx[2]]=mi
        del idx
        
        idx = np.where(data[:,col,:,:]>ma)
        data[idx[0],col,idx[1],idx[2]]=ma
        del idx
        gc.collect()
        return
    
    def except_site(self, site):
        self.data_info = self.data_info.numpy()
        self.data = self.data.numpy()
        self.label = self.label.numpy()
        idx=self.pre_data_info[:,2]!=site
        self.pre_data_info = self.pre_data_info[idx]
        self.data_info = torch.tensor(self.data_info[idx], dtype=torch.float32)
        self.data = torch.tensor(self.data[idx], dtype=torch.float32)
        self.label = torch.tensor(self.label[idx], dtype=torch.float32)
        
    def save_data(self):
        torch.save()
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data = self.data[index]
        return data[0:5,7,7].flatten(), data[5:,:,:], self.label[index]