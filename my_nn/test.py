from time import time
from torch.autograd import Variable
import config as c
from scipy.ndimage import geometric_transform
import numpy as np

import losses
import model
import torch
import matplotlib.pyplot as plt
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
n_samples = 10
model.load("checkpoints/my_inn.ckpt")
model.model.eval()
loader = c.test_loader
nograd = torch.no_grad()
nograd.__enter__()
for x, y in loader:
    break
x, y = x[:n_samples], y[:n_samples]
x, y = Variable(x).to(c.device), Variable(y).to(c.device)

out_x = model.model(y)
for i in range(n_samples):
    fig, ax = plt.subplots(1,2)
    oxi = out_x[i].cpu().numpy()
    xi  = x[i].cpu().numpy()
    oxi = geometric_transform(oxi, p2c)
    xi = geometric_transform(xi, p2c)
    ax[0].imshow(oxi)
    ax[0].set_title("recovered img")
    ax[1].imshow(xi)
    ax[1].set_title("original img")
    fig.savefig(f"imgs/result{i}.png")
    fig.clf()
