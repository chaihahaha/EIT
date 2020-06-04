from time import time
from torch.autograd import Variable
import config as c
from scipy.ndimage import geometric_transform
import numpy as np

import losses
import model
import torch
import matplotlib.pyplot as plt
import glob
circ_x, circ_y = (614, 618)
original_size = 1225
h,w = 256,256
def p2c(out_coords):
    x_idx, y_idx = out_coords[0], out_coords[1]
    dx          = x_idx * original_size / h - circ_x
    dy          = y_idx * original_size / w - circ_y
    r           = np.sqrt(dx**2 + dy**2)
    theta       = (np.arctan2(dy,dx)+np.pi*2)%(np.pi*2)
    r_idx       = r * h /600.0
    theta_idx   = theta * w / (2 * np.pi)
    return (r_idx, theta_idx)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.load("checkpoints/my_inn.ckpt")
model.model.eval()
nograd = torch.no_grad()
nograd.__enter__()
txt = sorted(glob.glob('test_y/' + '*.txt'))

for filename  in txt:
    y = np.array([x.split(' ')[0] for x in open(filename).readlines()])
    y = y.astype(np.float)[np.newaxis,:]
    y = torch.from_numpy(y).float()
    print(y.dtype)
    y = Variable(y).to(c.device)
    out_x = model.model(y).squeeze()
    fig, ax = plt.subplots(1,1)
    oxi = out_x.cpu().numpy()
    oxi = geometric_transform(oxi, p2c)
    ax.imshow(oxi)
    ax.axis('off')
    fig.savefig(f"imgs/{filename}.png", bbox_inches='tight')
    fig.clf()

test_split = 100
dataset_size = 20000
y_data = torch.Tensor(np.load('dataBoundary.npy')[:dataset_size])
y_test = y_data[-test_split:]
with open('filesList.csv','r') as f:
    s = f.read()
y_name = [i.split("\\")[-1] for i in s.split("\n") if i][1:]
y_test_name = y_name[-test_split:]
y = Variable(y_test).to(c.device)
out_x = model.model(y)
for i in range(100):
    fig, ax = plt.subplots(1,1)
    oxi = out_x[i].cpu().numpy()
    oxi = geometric_transform(oxi, p2c)
    ax.imshow(oxi)
    ax.axis('off')
    fig.savefig(f"imgs/{y_test_name[i]}.png", bbox_inches='tight')
    fig.clf()
     
