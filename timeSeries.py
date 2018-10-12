"""

This is a very simple dataset for time series modeling. It generates sinusoids with random phases and periods


helpers:

Display nicely the dataset:
--------------------------------------------

t = np.linspace(-1,1,20)
fig = plt.figure()
fig.set_figheight(8)
fig.set_figwidth(8)
i = 1
n = 10
for xx in np.linspace(-1,1,n):
    for yy in np.linspace(-1,1,n):
        X = gen_sinusoid(xx,yy,t)
        ax1 = fig.add_subplot(n,n,i)
        ax1.plot(X)
        ax1.set_xlim([0,20])
        ax1.set_ylim([0,1])
        #ax1.axis('equal')
        ax1.axis('off')
        i+=1 
plt.show() 


Create PyTorch loader
--------------------------------------------
dataset = Sinusoids(num_steps)
loader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

"""

import numpy as np
from torch.utils.data import Dataset

def gen_sinusoid(x,y,t):
    """
    generate a sinusoid for a period x and phase y
    Args:
        x (float): period
        y (float): phase
        t (list): time vector
    Returns:
        sinusoid (np.array): image of t by the function t |----> 0.5 + 0.5 * sin(pi t x + pi y)
    """
    return np.asarray([ 0.5 + 0.499 *np.sin(3.14 *u * x  + 3.14 * y) for u in t]) 

class Sinusoids(Dataset):
    """
    A simple dataset for time-series modeling compatible with PyTorch torch.utils.data.Loader object
    
    Generate sinuoids with random phases and peridos
    """

    def __init__(self,num_steps,n_phases=1,virtual_size=1000,quantization=-1):
        """
        
        Args:
            num_steps (int): number of steps for each item
            virtual_size (int): virtual size for the dataset
            quantization (int): if > 0, quantize the output into the range [0,'quantization']
        """
            
        self.t = np.linspace(-n_phases,n_phases,num_steps)
        self.virtual_size = virtual_size
        self.quantization = quantization

    def __len__(self):
        return self.virtual_size

    def __getitem__(self, idx):
        z_1 = np.random.uniform(-1,1)
        z_2 = np.random.uniform(-1,1)
        data = np.expand_dims( gen_sinusoid(z_1,z_2,self.t) , 1)
        if self.quantization > 0:
            data = (self.quantization*data).astype(int)
        return data
    
    
