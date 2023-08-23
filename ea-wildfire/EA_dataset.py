from pre_import import *
import gc
class EA_forestfire(Dataset):
    def __init__(self, X_info_path, X_info_name, X_path, X_name, Y_path, Y_name, delete_col=[], transform=None, norm='Z'):
        
        '''Setting preinformation'''
        self.real_columns = [1,2,3,4,6,7,8,9,10,11,12]
        self.columns_name = ['Land cover', 'Skin temperature', 'Relative humidity', 'Solar zenith angle', 'Satellite zenith angle',
                             'Cloud', 'Ch01', 'Ch03', 'BT07', 'BT07-12', 'BT07-14',
                             'Spatial difference of BT07', 'Spatial difference of BT07-14']
        self.info_columns_name = ['SetID', 'IsVIIRS', 'Site Number', 'Case number', 
                                  'Year', 'Month', 'Day', 'Hour', 'Minute', 'Row', 'Col']
        self.transform = transform
        self.delete_col = delete_col
        self.norm = norm
        
        '''Handling info data'''
        mat_file = scipy.io.loadmat(X_info_path)
        self.pre_data_info = mat_file[X_info_name]
        filtered_data_info = np.delete(self.pre_data_info, [0,1,2,3,4,9,10], -1) #info 데이터에서 날짜 데이터만 사용
        
        # Convert Site number to One-hot vector & Normalizing date information
        self.data_info=[]
        for i in range(len(filtered_data_info)):
            row = filtered_data_info[i]
            new_row = [0]*(4) #4개의 날짜정보
            new_row[0] = row[0]/12
            new_row[1] = row[1]/31
            new_row[2] = row[2]/23
            new_row[3] = row[3]/59
            self.data_info.append(new_row)
        self.data_info = torch.tensor(np.array(self.data_info, dtype=np.float32), dtype=torch.float32)
        
        '''Handling data'''
        mat_file = scipy.io.loadmat(X_path)
        self.data = mat_file[X_name]
        self.normalize()
        self.clip(self.data)
        self.data = np.delete(self.data, delete_col, -1) #Cloud 컬럼 정보 사용 안함 
        if self.transform==None:
            self.data = torch.tensor(self.data, dtype=torch.float32) 
            self.data = self.data.permute(0,3,1,2).contiguous()
        
        
        '''Handling label data'''
        mat_file = scipy.io.loadmat(Y_path)
        self.label = torch.tensor(mat_file[Y_name].flatten(), dtype=torch.float32)
        
    def normalize(self):
        start = 1
        if 0 in self.delete_col:
            start=0
        means = []
        stds = []
        for i in range(start, self.data.shape[-1]):
            if self.norm=='Z':
                mean = np.mean(self.data[:,:,:,i].flatten())
                std = np.std(self.data[:,:,:,i].flatten())
                if std==0:std=1
                self.data[:,:,:,i] = (self.data[:,:,:,i] - mean)/std
                means.append(mean)
                stds.append(std)
            elif self.norm=='Min-Max':
                ma = np.max(self.data[:,:,:,i])
                mi = np.min(self.data[:,:,:,i])
                self.data[:,:,:,i] = (self.data[:,:,:,i] - mi) / (ma - mi)
                
                
    def clip(self, data):
        start = 1
        if 0 in self.delete_col:
            start=0
            
        mi = -3
        ma = 3
        idx = np.where(data[:,:,:, start:]<mi)
        data[idx[0],idx[1],idx[2], start:]=mi
        del idx
        gc.collect()
        
        idx = np.where(data[:,:,:, start:]>ma)
        data[idx[0],idx[1],idx[2], start:]=ma
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
        
            
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        data = self.data[index]
        if self.transform:
            augmented = self.transform(image=self.data[index]) 
            data = augmented['image'].float()
        return self.data_info[index], data, self.label[index]


class EA_forestfire2(Dataset):
    def __init__(self, X_info_path, X_info_name, X_path, X_name, Y_path, Y_name, delete_col=[], transform=None, norm='Z'):
        
        '''Setting preinformation'''
        self.real_columns = [1,2,3,4,6,7,8,9,10,11,12]
        self.columns_name = ['Land cover', 'Skin temperature', 'Relative humidity', 'Solar zenith angle', 'Satellite zenith angle',
                             'Cloud', 'Ch01', 'Ch03', 'BT07', 'BT07-12', 'BT07-14',
                             'Spatial difference of BT07', 'Spatial difference of BT07-14']
        self.info_columns_name = ['SetID', 'IsVIIRS', 'Site Number', 'Case number', 
                                  'Year', 'Month', 'Day', 'Hour', 'Minute', 'Row', 'Col']
        self.transform = transform
        self.norm = norm
        self.delete_col = delete_col
        
        '''Handling info data'''
        mat_file = scipy.io.loadmat(X_info_path)
        self.pre_data_info = mat_file[X_info_name]
        filtered_data_info = np.delete(self.pre_data_info, [0,1,2,3,4,9,10], -1) #info 데이터에서 날짜 데이터만 사용
        
        # Convert Site number to One-hot vector & Normalizing date information
        self.data_info=[]
        for i in range(len(filtered_data_info)):
            row = filtered_data_info[i]
            new_row = [0]*(4) #4개의 날짜정보
            new_row[0] = row[0]/12
            new_row[1] = row[1]/31
            new_row[2] = row[2]/23
            new_row[3] = row[3]/59
            self.data_info.append(new_row)
        self.data_info = torch.tensor(np.array(self.data_info, dtype=np.float32), dtype=torch.float32)
        
        '''Handling data'''
        mat_file = scipy.io.loadmat(X_path)
        self.data = mat_file[X_name]
        self.normalize()
        #self.clip(self.data)
        self.data = np.delete(self.data, delete_col, -1) #Cloud 컬럼 정보 사용 안함 
        if self.transform==None:
            self.data = torch.tensor(self.data, dtype=torch.float32) 
            self.data = self.data.permute(0,3,1,2).contiguous()
        
        
        '''Handling label data'''
        mat_file = scipy.io.loadmat(Y_path)
        self.label = torch.tensor(mat_file[Y_name].flatten(), dtype=torch.float32)
        
    def normalize(self):
        for i in self.real_columns:
            if self.norm=='Z':
                mean = np.mean(self.data[:,:,:,i].flatten())
                std = np.std(self.data[:,:,:,i].flatten())
                if std==0:std=1
                self.data[:,:,:,i] = (self.data[:,:,:,i] - mean)/std
            elif self.norm=='Min-Max':
                ma = np.max(self.data[:,:,:,i])
                mi = np.min(self.data[:,:,:,i])
                self.data[:,:,:,i] = (self.data[:,:,:,i] - mi) / (ma - mi)
                
    def clip(self, data):
        start = 1
        if 0 in self.delete_col:
            start=0
            
        mi = -3
        ma = 3
        idx = np.where(data[:,:,:, start:]<mi)
        data[idx[0],idx[1],idx[2], start:]=mi
        del idx
        gc.collect()
        
        idx = np.where(data[:,:,:, start:]>ma)
        data[idx[0],idx[1],idx[2], start:]=ma
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
        
            
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        data = self.data[index]
        if self.transform:
            augmented = self.transform(image=self.data[index]) 
            data = augmented['image'].float()
        return data[0:5,7,7].flatten(), data[5:,:,:], self.label[index]
    
    

def make_weights_for_balanced_classes(dataset, n_classes):                        
    count = [0]*n_classes
    label = dataset.label.numpy()
    for i in range(n_classes):
        count[i]=sum(label==i)
    ma = max(count)
    
    weight_per_class = [0.] * n_classes        
    for i in range(n_classes):
        weight_per_class[i] = ma/count[i]
    
    weight = [0]*len(label)
    for i in range(len(label)):
        weight[i]=weight_per_class[int(label[i])]
    return torch.tensor(weight, dtype=torch.float32) 


def EA_Train_DataLoader(delete_col=[0,5,6,7,8], norm='Z'):
    transform = albumentations.Compose([
        albumentations.HorizontalFlip(p=0.5),
        albumentations.RandomRotate90(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.pytorch.transforms.ToTensorV2()
    ])

    trainset = EA_forestfire('/share/wildfire-2/shlee/EA/info_tr_CNN_w15.mat',
                             'info_tr',
                             '/share/wildfire-2/shlee/EA/X_tr_CNN_w15.mat',
                             'X_tr',
                             '/share/wildfire-2/shlee/EA/Y_tr_CNN_w15.mat',
                             'Y_tr',
                             delete_col=delete_col,
                            transform=transform,
                            norm=norm)
    
    weights = make_weights_for_balanced_classes(trainset, 2)                                                                
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights), replacement=True) 

    train_dataloader = DataLoader(trainset, batch_size=512, num_workers=2, sampler=sampler)
    #train_dataloader = DataLoader(trainset, batch_size=512, num_workers=2, shuffle=True)
    print('Complete Loading Train Set')
    return trainset, train_dataloader


def EA_Valid_DataLoader(delete_col=[0,5,6,7,8], norm='Z'):
    valset = EA_forestfire('/share/wildfire-2/shlee/EA/info_va_CNN_w15.mat',
                            'info_va',
                           '/share/wildfire-2/shlee/EA/X_va_CNN_w15.mat',
                            'X_va',
                            '/share/wildfire-2/shlee/EA/Y_va_CNN_w15.mat',
                           'Y_va',
                            delete_col=delete_col,
                            norm=norm)
    
    val_dataloader = DataLoader(valset, batch_size=512, shuffle=False, num_workers=2)
    print('Complete Loading Valid Set')
    return valset, val_dataloader

def EA_Test_DataLoader(delete_col=[0,5,6,7,8], norm='Z'):
    testset = EA_forestfire('/share/wildfire-2/shlee/EA/info_te_CNN_w15.mat',
                            'info_te',
                           '/share/wildfire-2/shlee/EA/X_te_CNN_w15.mat',
                            'X_te',
                            '/share/wildfire-2/shlee/EA/Y_te_CNN_w15.mat',
                           'Y_te',
                            delete_col=delete_col,
                            norm=norm)
    test_dataloader = DataLoader(testset, batch_size=512, shuffle=False, num_workers=2)
    print('Complete Loading Test Set')
    return testset, test_dataloader





def EA2_Train_DataLoader(delete_col=[0,5,6,7,8], norm='Z'):
    transform = albumentations.Compose([
        albumentations.HorizontalFlip(p=0.5),
        albumentations.RandomRotate90(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.pytorch.transforms.ToTensorV2()
    ])

    trainset = EA_forestfire2('/share/wildfire-2/shlee/EA/info_tr_CNN_w15.mat',
                             'info_tr',
                             '/share/wildfire-2/shlee/EA/X_tr_CNN_w15.mat',
                             'X_tr',
                             '/share/wildfire-2/shlee/EA/Y_tr_CNN_w15.mat',
                             'Y_tr',
                             delete_col=delete_col,
                             transform=transform,
                             norm=norm)
    
    weights = make_weights_for_balanced_classes(trainset, 2)                                                                
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights), replacement=True) 

    train_dataloader = DataLoader(trainset, batch_size=512, num_workers=2, sampler=sampler)
    #train_dataloader = DataLoader(trainset, batch_size=512, num_workers=2, shuffle=True)
    print('Complete Loading Train Set')
    return trainset, train_dataloader


def EA2_Valid_DataLoader(delete_col=[0,5,6,7,8], norm='Z'):
    valset = EA_forestfire2('/share/wildfire-2/shlee/EA/info_va_CNN_w15.mat',
                            'info_va',
                           '/share/wildfire-2/shlee/EA/X_va_CNN_w15.mat',
                            'X_va',
                            '/share/wildfire-2/shlee/EA/Y_va_CNN_w15.mat',
                           'Y_va',
                            delete_col=delete_col,
                            norm=norm)
    
    val_dataloader = DataLoader(valset, batch_size=512, shuffle=False, num_workers=2)
    print('Complete Loading Valid Set')
    return valset, val_dataloader

def EA2_Test_DataLoader(delete_col=[0,5,6,7,8], norm='Z'):
    testset = EA_forestfire2('/share/wildfire-2/shlee/EA/info_te_CNN_w15.mat',
                            'info_te',
                           '/share/wildfire-2/shlee/EA/X_te_CNN_w15.mat',
                            'X_te',
                            '/share/wildfire-2/shlee/EA/Y_te_CNN_w15.mat',
                           'Y_te',
                            delete_col=delete_col,
                            norm=norm)
    test_dataloader = DataLoader(testset, batch_size=512, shuffle=False, num_workers=2)
    print('Complete Loading Test Set')
    return testset, test_dataloader