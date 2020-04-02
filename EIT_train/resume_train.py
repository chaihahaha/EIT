import numpy as np
from sklearn.decomposition import PCA
from time import time
from gen_model import *
import torch
import torch.optim
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
import os
from train_utils import *
import sys

img = np.load("dataImages.npy")
print("Original img shape:",img.shape)
original_shape = img.shape[1:]

img = img.reshape(900,-1)
print(img.shape)

boundary = np.load("dataBoundary.npy")
print("Original boundary shape:",boundary.shape)


# load pca from file
with open("pca.pkl", "rb") as f:
    pca = pk.load(f)
img=pca.transform(img)

with open("params.pkl","rb") as f:
    params = pk.load(f)
    (norm_factor_img, norm_factor_bdr, original_shape, ndim_x, ndim_y, ndim_z, ndim_tot, ckpt_dir) = params
original_shape = (266,256)
# load model from file
model = Model(ndim_tot * 2, ndim_tot)

checkpoint = torch.load(ckpt_dir + "model"+sys.argv[1]+".torch")
model.load_state_dict(checkpoint['model_state_dict'])
old_epoch = checkpoint['epoch']

img, boundary = torch.from_numpy(img).float()/norm_factor_img, torch.from_numpy(boundary).float()/norm_factor_bdr
device = 'cuda' if torch.cuda.is_available() else 'cpu'
test_split = 100

# Training parameters
n_epochs = 3000
n_its_per_epoch = 2
batch_size = 900 - test_split
save_freq = 100

lr = 1e-5
l2_reg = 2e-5

y_noise_scale = 1e-1
zeros_noise_scale = 5e-2

# relative weighting of losses:
lambd_predict = 3.
lambd_latent = 300.
lambd_rev = 400.

trainable_parameters = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(trainable_parameters, lr=lr, betas=(0.8, 0.9),
                             eps=1e-6, weight_decay=l2_reg)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(img[:test_split], boundary[:test_split]),
    batch_size=batch_size, shuffle=True, drop_last=True)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(img[test_split:], boundary[test_split:]),
    batch_size=batch_size, shuffle=True, drop_last=True)

for param in trainable_parameters:
    param.data = 0.05*torch.randn_like(param)

model.to(device)

ckpt_dir = "./checkpoints/"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
with open("params.pkl","wb") as f:
    params = (norm_factor_img, norm_factor_bdr, original_shape, ndim_x, ndim_y, ndim_z, ndim_tot, ckpt_dir)
    pk.dump(params, f)

try:
    t_start = time()
    for i_epoch in range(old_epoch + 1, old_epoch + n_epochs):
        loss = train(model, optimizer, train_loader, n_epochs, n_its_per_epoch, batch_size, zeros_noise_scale, ndim_x, ndim_y, ndim_z, ndim_tot, device, y_noise_scale, lambd_predict, lambd_latent, lambd_rev, i_epoch)
        print("Epoch:",i_epoch,"Loss:",loss)
        if i_epoch%save_freq == 0:
            model_opt = {'model_state_dict': model.state_dict(),
                         'optimizer_state_dict': optimizer.state_dict(),
                         'epoch': i_epoch}
            torch.save(model_opt, ckpt_dir + "model"+str(i_epoch)+".torch")
except KeyboardInterrupt:
    pass
finally:
    print(f"\n\nTraining took {(time()-t_start)/60:.2f} minutes\n")


