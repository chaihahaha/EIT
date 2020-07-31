from time import time
from torch.autograd import Variable
from scipy.ndimage import geometric_transform
from skimage.metrics import structural_similarity
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

def dice_loss(x1, x2):
    numerator = 2 * np.sum(x1 * x2)
    denominator = np.sum(x1+x2)
    return 1-(numerator + 1)/(denominator + 1)

def total_var(x1, x2):
    return np.sum(np.abs(x1-x2))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.load("checkpoints/my_inn.ckpt")
model.model.eval()
nograd = torch.no_grad()
nograd.__enter__()
#txt = sorted(glob.glob('test_y/' + '*.txt'))
#
#for filename  in txt:
#    y = np.array([x.split(' ')[0] for x in open(filename).readlines()])
#    y = y.astype(np.float)[np.newaxis,:]
#    y = torch.from_numpy(y).float()
#    print(y.dtype)
#    y = Variable(y).to("cuda")
#    out_x = model.model(y).squeeze()
#    fig, ax = plt.subplots(1,1)
#    oxi = out_x.cpu().numpy()
#    oxi = geometric_transform(oxi, p2c)
#    ax.imshow(oxi)
#    ax.axis('off')
#    fig.savefig(f"imgs/{filename}.png", bbox_inches='tight')
#    fig.clf()

y_test = torch.Tensor(np.load('testBoundary.npy'))
x_test = np.load('testImages.npy')
with open('filesList.csv','r') as f:
    s = f.read()
y_test_name = [i.split("\\")[-1] for i in s.split("\n") if i][1:]
y = Variable(y_test).to("cuda")
out_x = model.model(y)
for i in range(len(y_test)):
    fig, ax = plt.subplots(1,1)
    oxi = out_x[i].cpu().numpy()
    #oxi = geometric_transform(oxi, p2c)
    ax.imshow(oxi)
    ax.axis('off')
    fig.savefig(f"imgs/{y_test_name[i]}.png", bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(1,1)
    oxi = x_test[i]
    ax.imshow(oxi)
    ax.axis('off')
    fig.savefig(f"gts/{y_test_name[i]}.png", bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(1,2)
    oxi = out_x[i].cpu().numpy()
    gtxi = x_test[i]
    ssim = structural_similarity(oxi, gtxi, data_range=255)
    dice = dice_loss(oxi/255.0, gtxi/255.0)
    l2   = np.sum((oxi-gtxi)**2)
    ttv  = total_var(oxi, gtxi)
    
    ax[0].imshow(oxi)
    ax[0].set_title("recovered image")
    ax[1].imshow(gtxi)
    ax[1].set_title("ground truth")
    fig.suptitle("SSIM: {:.3e} Dice loss: {:.3e} \nL2 loss: {:.3e} total variation: {:.3e}".format(ssim, dice, l2, ttv))
    fig.savefig(f"cmps/{y_test_name[i]}.png", bbox_inches='tight')
    plt.close()
