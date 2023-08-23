import numpy as np
from openstl.utils import show_video_line
from pre_import import *
from gsim_vp import GSimVP
import dataset_sst
import matplotlib as mpl
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from MovingMNIST import movingMNIST
def plot_imgs(pred, label):
    n = len(pred)
    for i in range(n):
        ax = plt.subplot(2, n, i+1)
        plt.imshow(label[i][0])
        ax.axis('off')
        ax = plt.subplot(2, n, n+i+1)
        plt.imshow(pred[i][0])
        ax.axis('off')
    
    plt.show()
    plt.close()
    
predict_mode = 'once'
root_path = './experiments/LossPenalty_GSimVP/'
file_path = f'{root_path}/validate/{predict_mode}/'
def f(imgs, name, mi=0, ma=0, cmap='jet'):
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    #plt.rcParams["figure.autolayout"] = True

    fig = plt.figure()
    print(mi, ma)
    ax = fig.add_subplot(111)
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', '5%', '5%')
    tx = ax.set_title('Frame 0')

    def animate(i):
        cax.cla()
        data = np.random.rand(5, 5)
        if mi!=0 or ma!=0:
            im = ax.imshow(imgs[i][0], cmap=cmap, vmin=mi, vmax=ma)
        else:
            im = ax.imshow(imgs[i][0], cmap=cmap)
        fig.colorbar(im, cax=cax)
        tx.set_text('Frame {0}'.format(i))
    ani = animation.FuncAnimation(fig, animate, frames=range(len(imgs)))
    ani.save(f'{file_path+name}.gif', writer='imagemagick')

def predict_seq(batch_data, T):
    total_pred = []
    for i in range(T):    
        pred = gsimvp(batch_data)[0]
        batch_data = torch.cat((batch_data[:,1:,:,:,:], pred[:,:1,:,:,:]), dim=1)
        total_pred.append(pred[0,:1,:,:,:])
    total_pred = torch.cat(total_pred,dim=0).detach().numpy()
    return total_pred

def predict_seq2(batch_data, T):
    total_pred = []
    for i in range(T):    
        pred = gsimvp(batch_data)[0]
        batch_data = torch.cat((batch_data[:,1:,:,:,:], pred[:,-1:,:,:,:]), dim=1)
        total_pred.append(pred[0,-1:,:,:,:])
    total_pred = torch.cat(total_pred,dim=0).detach().numpy()
    return total_pred

set_name='train'
sst_valid_dataset = dataset_sst.SSTDataset(90, predict_day=90 ,img_interval=1,mode=set_name)


data, label = sst_valid_dataset[0]
T, C, H, W = label.shape
print(T,C,H,W)
model_name = 'gsimvp_100ep_imginterval_1_predict_365_best.pkl'
record = torch.load(f'{root_path}models/{model_name}')
gsimvp = GSimVP(in_shape=(T,C,H,W), hid_T=128).eval()
gsimvp.load_state_dict(record['model_state_dict'])


example_idx = 0

if predict_mode=='seq':
    set_n = 4
    seq_n = T*set_n
    data = sst_valid_dataset.sst_data[example_idx:example_idx+7]
    label = sst_valid_dataset.sst_data[example_idx+7: example_idx+37]
    data = data.reshape((1,7,1,200,200))
    label = label.reshape((1,30,1,200,200))
    batch_data = torch.tensor(data)
    total_pred = predict_seq(batch_data[:1], T=30)
    label = label.reshape(-1,1, 200,200)
elif predict_mode=='once':
    data, label = sst_valid_dataset[example_idx]
    batch_data = data.reshape((1, *data.shape))
    total_pred = gsimvp(batch_data)[0][0].detach().numpy()
    label = label.numpy()
    
mpl.rcParams['figure.figsize'] = [25, 10]
#plot_imgs(pred, label)
ma = max(np.max(total_pred), np.max(label))
mi = min(np.min(total_pred), np.min(label))
f(label, f'{model_name}_{set_name}_label_{predict_mode}_{example_idx}', mi, ma)
f(total_pred, f'{model_name}_{set_name}_pred_{predict_mode}_{example_idx}', mi, ma)

diff_img = label - total_pred
diff_img = np.abs(diff_img)
f(diff_img, f'{model_name}_{set_name}_diff_{predict_mode}_{example_idx}', np.min(diff_img), np.max(diff_img), 'Purples')