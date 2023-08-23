from pre_import import *
import matplotlib as mpl
class SSTDataset(Dataset):
    def __init__(self, time_interval=3, predict_day=7, img_interval=7, data_label_gap=None, mode='train', dataset=None):
        '''Setting parameters'''        
        self.time_interval=time_interval
        self.predict_day = predict_day
        self.img_interval=img_interval
        if data_label_gap==None:
            self.data_label_gap=predict_day
        else:
            self.data_label_gap = data_label_gap
            
        self.mean=None
        self.std=None
        self.n_days = [365, 366, 365, 365, 365, 366, 365, 365, 365, 366, 365, 365, 365,
                366, 365, 365, 365, 366, 365, 365, 365, 366, 365, 365, 365, 366, 365]
        '''SST Data Loading'''
        if type(dataset)==type(None):
            print('loading npy file')
            sst_data = np.load('/home/shlee/preprocessed_CCI_1995_2021_200_200.npy')
            print('done')
            '''Data Normalizing'''
            print('normalizing')
            sst_data = self.min_max_normalize(sst_data)
            self.sst_data = sst_data
            print('done')
        else:
            sst_data = dataset
            self.sst_data = sst_data
        '''Construct Input, Target Data'''
        data = []
        label = []
        start=0
        end=0
        if mode=='train':
            end=len(sst_data)-sum(self.n_days[-4:]) # cut out last four years
            
        elif mode=='valid':
            start = sum(self.n_days[:-4])
            end = sum(self.n_days[:-2])
            
        elif mode=='test':
            start = sum(self.n_days[:-2])
            end = len(self.sst_data)
        
        print('start making')
        print(start, end)
        s = set()
        #This can occur missing of last few data from end
        for i in range(start, end-(self.data_label_gap*img_interval+predict_day*img_interval)+1, time_interval):
            input_data = sst_data[i:i+predict_day*img_interval:img_interval]
            target_data = sst_data[i+self.data_label_gap*img_interval : i+self.data_label_gap*img_interval+predict_day*img_interval:img_interval]
            data.append(input_data) 
            label.append(target_data)
            s.add(input_data.shape)
            # print(i,i+predict_day*img_interval,img_interval)
            # print(i+self.data_label_gap*img_interval, i+self.data_label_gap*img_interval+predict_day*img_interval, img_interval)
            # break
        print('end')
        #Convert data type
        print('converting')
        print(len(data))
        print(s)
        data = np.expand_dims(np.asarray(data, dtype=np.float32), axis=2)
        label = np.expand_dims(np.asarray(label, dtype=np.float32), axis=2)
        self.data = torch.tensor(data, dtype=torch.float32)
        self.label = torch.tensor(label, dtype=torch.float32)
        print('done')
    
    def min_max_normalize(self, data):
        ma = np.max(data)
        mi = np.min(data)
        data[:,:,:] = (data-mi) / (ma-mi)
        print(ma, mi)
        return data
        
    def __len__(self):
        return len(self.label)
        
    def __getitem__(self, index):
        return self.data[index], self.label[index]


if __name__=='__main__':
    sst_dataset = SSTDataset()
    print(f'Shape of data:{sst_dataset.data.shape}, label:{sst_dataset.label.shape}')
    print(f'Type of data:{type(sst_dataset.data)}, label:{type(sst_dataset.label)}')
    
    
    mpl.rcParams['figure.figsize'] = [25, 15]
    data, label = sst_dataset[0]
    print(label.shape)
    for i in range(sst_dataset.predict_day):
        plt.subplot(1, sst_dataset.predict_day, i+1)
        plt.imshow(data[i][0])
    plt.show()
    
    for i in range(sst_dataset.predict_day):
        plt.subplot(1, sst_dataset.predict_day, i+1)
        plt.imshow(label[i][0])
    plt.show()
