from time import time
from torch.autograd import Variable
import config as c

import losses
import model
import torch
import matplotlib.pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_samples = 10
model.load("checkpoints/my_inn.ckpt")
model.model.eval()
loader = c.train_loader
nograd = torch.no_grad()
nograd.__enter__()
for x, y in loader:
    break
x, y = x[:n_samples], y[:n_samples]
x, y = Variable(x).to(c.device), Variable(y).to(c.device)
if c.add_y_noise > 0:
        y += c.add_y_noise * noise_batch(c.ndim_y)[:n_samples]
if c.ndim_pad_x:
    x = torch.cat((x, c.add_pad_noise * noise_batch(c.ndim_pad_x)[:n_samples]), dim=1)
if c.ndim_pad_zy:
    y = torch.cat((c.add_pad_noise * noise_batch(c.ndim_pad_zy)[:n_samples], y), dim=1)
y = torch.cat((noise_batch(c.ndim_z)[:n_samples], y), dim=1)

out_x = model.model(y, rev=True)
for i in range(n_samples):
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(out_x[i].view(c.img_shape).cpu().numpy())
    ax[0].set_title("recovered img")
    ax[1].imshow(x[i].view(c.img_shape).cpu().numpy())
    ax[1].set_title("original img")
    fig.savefig(f"imgs/result{i}.png")
    fig.clf()
