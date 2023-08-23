import numpy as np
import matplotlib.pyplot as plt
data = np.load('/share/ocean-4/SST/prediction_SST/preprocessed_sst/preprocessed_CCI_1995_2021_200_200.npy')
print(data.shape)
plt.imshow(data[4321], cmap ='jet')
plt.colorbar()

print(np.count_nonzero(np.isnan(data)))

mean_img = np.mean(data, axis=(1,2))
plt.plot(mean_img)


import matplotlib as mpl
mpl.rcParams['figure.figsize'] = [10, 5]
n_days = [365, 366, 365, 365, 365, 366, 365, 365, 365, 366, 365, 365, 365,
          366, 365, 365, 365, 366, 365, 365, 365, 366, 365, 365, 365, 366, 365]
print("Plotting temperature flow of each year")
x = np.array(range(0, len(data)))
start=0
for i in range(len(n_days)):
    days = n_days[i]
    plt.plot(x[start:start+days],mean_img[start:start+days])
    start+=days
plt.show()
    
print("Plotting temperature flow of each year with overlapping")
start=0
for i in range(len(n_days)):
    days = n_days[i]
    plt.plot(mean_img[start:start+days], alpha=0.5, label=str(i+1995))
    start+=days
plt.show()

print("Checking Min, Max")
print(f'Min:{np.min(data)}  Max:{np.max(data)}')    


def plot_temperature(data):
    print("Plotting temperature flow of each year")
    n_days = [365, 366, 365, 365, 365, 366, 365, 365, 365, 366, 365, 365, 365,
          366, 365, 365, 365, 366, 365, 365, 365, 366, 365, 365, 365, 366, 365]
    x = np.array(range(0, len(data)))
    start=0
    for i in range(len(n_days)):
        days = n_days[i]
        plt.plot(x[start:start+days],data[start:start+days])
        start+=days
    plt.show()
    return

def plot_temperature_overlap(data):
    print("Plotting temperature flow of each year with overlapping")
    n_days = [365, 366, 365, 365, 365, 366, 365, 365, 365, 366, 365, 365, 365,
          366, 365, 365, 365, 366, 365, 365, 365, 366, 365, 365, 365, 366, 365]
    start=0
    for i in range(len(n_days)):
        days = n_days[i]
        plt.plot(data[start:start+days], alpha=0.5, label=str(i+1995))
        start+=days
    plt.show()

    print("Checking Min, Max")
    print(f'Min:{np.min(data)}  Max:{np.max(data)}')  

