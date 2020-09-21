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
    return np.mean(np.abs(x1-x2))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.load("checkpoints/my_inn.ckpt")
n_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
print("# of params:", n_params)
model.model.eval()
nograd = torch.no_grad()
nograd.__enter__()
txt = sorted(glob.glob('test_y/' + '*.txt'))

for filename  in txt:
    y = np.array([x.split(' ')[0] for x in open(filename).readlines()])
    y = y.astype(np.float)[np.newaxis,:]
    y = torch.from_numpy(y).float()
    y = Variable(y).to("cuda")
    out_x = model.model(y).squeeze()
    fig, ax = plt.subplots(1,1)
    oxi = out_x.cpu().numpy()
    ax.imshow(oxi)
    ax.axis('off')
    fig.savefig(f"{filename}.png", bbox_inches='tight')
    plt.close()

y_test = torch.Tensor(np.load('testBoundary.npy'))
x_test = np.load('testImages.npy')
with open('filesList.csv','r') as f:
    s = f.read()
y_test_name = [i.split("\\")[-1] for i in s.split("\n") if i][1:]
y = Variable(y_test).to("cuda")
out_x = model.model(y)
metrics = []
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
    diff_mean = np.mean(oxi-gtxi)
    diff_std  = np.std(oxi-gtxi)
    mse = np.mean((oxi-gtxi)**2)
    mtv  = total_var(oxi, gtxi)
    metrics.append([ssim,dice,diff_mean,diff_std,mse,mtv])
    
    ax[0].imshow(oxi)
    ax[0].set_title("recovered image")
    ax[1].imshow(gtxi)
    ax[1].set_title("ground truth")
    fig.suptitle("SSIM: {:.3f}, Dice loss: {:.3f} \nstandard deviation: {:.2f}, total variation: {:.2f}".format(ssim, dice, diff_std, mtv))
    ax[0].axis('off')
    ax[1].axis('off')
    fig.savefig(f"cmps/{y_test_name[i]}.png")
    plt.close()
metrics = np.array(metrics)
ssim = metrics[:,0]
dice = metrics[:,1]
diff_mean = metrics[:,2]
diff_std = metrics[:,3]
mse = metrics[:,4]
mtv = metrics[:,5]
fig, ax = plt.subplots(3,2,constrained_layout=True)
ax[0,0].hist(ssim,10)
ax[0,0].set_title("SSIM")
ax[0,1].hist(dice,10)
ax[0,1].set_title("Dice loss")
ax[1,0].hist(diff_mean,10)
ax[1,0].set_title("Mean signed deviation")
ax[1,1].hist(diff_std,10)
ax[1,1].set_title("Standard deviation")
ax[2,0].hist(mse,10)
ax[2,0].set_title("Mean square error")
ax[2,1].hist(mtv,10)
ax[2,1].set_title("Total Variance")
fig.savefig("metrics.png")
plt.close()
