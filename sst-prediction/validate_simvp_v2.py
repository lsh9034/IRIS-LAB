import numpy as np
from openstl.utils import show_video_line
from pre_import import *
from openstl.models import SimVP_Model
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
        plt.imshow(pred[i][0], cmap='Greys')
        ax.axis('off')
    
    plt.show()
    plt.close()



def make_gif(imgs, name):
    fig, ax = plt.subplots()
    
    n = len(pred)
    gif = []
    title = ax.text(0.5,1.1, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=ax.transAxes, ha="center")
    
    def update(idx):
        title.set_text(str(idx))
        im = ax.imshow(imgs[idx][0],cmap='jet', vmin=0, vmax=1)
        cb = fig.colorbar(im)
        plt.show()
    ani = animation.FuncAnimation(fig, func=update, frames=range(n), interval=300)
    ani.save(f'./gifs/{name}.gif', writer='imagemagick')


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
        if mi!=0 or ma!=0:
            im = ax.imshow(imgs[i][0], cmap=cmap, vmin=mi, vmax=ma)
        else:
            im = ax.imshow(imgs[i][0], cmap=cmap)
        fig.colorbar(im, cax=cax)
        tx.set_text('Frame {0}'.format(i))
    ani = animation.FuncAnimation(fig, animate, frames=range(len(imgs)))
    ani.save(f'./gifs/{name}.gif', writer='imagemagick')

def predict_seq(batch_data):
    total_pred = []
    for i in range(T):    
        pred = sst_v2(batch_data)
        batch_data = torch.cat((batch_data[:,1:,:,:,:], pred[:,:1,:,:,:]), dim=1)
        total_pred.append(pred[0,:1,:,:,:])
    total_pred = torch.cat(total_pred,dim=0).detach().numpy()
    return total_pred




#sst_valid_dataset = dataset_sst.SSTDataset(3, predict_day=7 ,img_interval=2, mode='train')
sst_valid_dataset = movingMNIST.MovingMNIST(root='.data/mnist', train=True, download=True)

data, label = sst_valid_dataset[0]
T, C, H, W = data.shape
print(T,C,H,W)
record = torch.load('./models/mnist_simvp_test_best.pkl')
sst_v2 = SimVP_Model(in_shape=(T,C,H,W), hid_T=128).eval()
sst_v2.load_state_dict(record['model_state_dict'])


example_idx = 123
data, label = sst_valid_dataset[example_idx]
batch_data = data.reshape((1, *data.shape))

predict_mode = 'seq'

if predict_mode=='seq':
    total_pred = predict_seq(batch_data)
elif predict_mode=='once':
    total_pred = sst_v2(batch_data)[0].detach().numpy()
    
label = label.numpy()
mpl.rcParams['figure.figsize'] = [25, 10]
#plot_imgs(pred, label)
ma = max(np.max(total_pred), np.max(label))
mi = min(np.min(total_pred), np.min(label))
f(label, f'label_{example_idx}', mi, ma)
f(total_pred, f'pred_once_{example_idx}', mi, ma)

diff_img = label - total_pred
diff_img = np.abs(diff_img)
f(diff_img, f'diff_{example_idx}', np.min(diff_img), np.max(diff_img), 'Purples')