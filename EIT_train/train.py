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
from utils import *

img = np.load("dataImages.npy")
print("Original img shape:",img.shape)
original_shape = img.shape[1:]

img = img.reshape(900,-1)
print(img.shape)

boundary = np.load("dataBoundary.npy")
print("Original boundary shape:",boundary.shape)

pca = PCA()
pca.fit(img)
img=pca.transform(img)
print("PCA img shape:",img.shape)
with open("pca.pkl", "wb") as f:
    pk.dump(pca, f)

norm_factor_img = np.std(img)
norm_factor_bdr = np.std(boundary)
print("norm_factor_img:",norm_factor_img)
print("norm_factor_bdr:",norm_factor_bdr)


img, boundary = torch.from_numpy(img).float()/norm_factor_img, torch.from_numpy(boundary).float()/norm_factor_bdr
device = 'cuda' if torch.cuda.is_available() else 'cpu'
test_split = 100

# set up dim
ndim_x = img.shape[1]
ndim_y = boundary.shape[1]
ndim_z = 1000
ndim_tot = ndim_x + ndim_y + ndim_z
print("dims:",ndim_x,ndim_y,ndim_z,ndim_tot)

# set up model
model = Model(ndim_tot * 2, ndim_tot)

# Training parameters
n_epochs = 3000
n_its_per_epoch = 2
batch_size = 900 - test_split
save_freq = 100

lr = 2e-5
l2_reg = 1e-3

y_noise_scale = 1e-1
zeros_noise_scale = 5e-2

# relative weighting of losses:
lambd_predict = 3.
lambd_latent = 300.
lambd_rev = 400.

pad_x = torch.zeros(batch_size, ndim_tot - ndim_x)
pad_yz = torch.zeros(batch_size, ndim_tot - ndim_y - ndim_z)

trainable_parameters = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(trainable_parameters, lr=lr, betas=(0.8, 0.9),
                             eps=1e-6, weight_decay=l2_reg)



loss_backward = MMD_multiscale
loss_latent = MMD_multiscale
loss_fit = fit

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
    for i_epoch in range(n_epochs):
        print("Epoch:",i_epoch,"Loss:",train(i_epoch))
        if i_epoch%save_freq == 0:
            model_opt = {'model_state_dict': model.state_dict(),
                         'optimizer_state_dict': optimizer.state_dict()
                         'epoch': i_epoch}
            torch.save(model_opt, ckpt_dir + "model"+str(i_epoch)+".torch")
except KeyboardInterrupt:
    pass
finally:
    print(f"\n\nTraining took {(time()-t_start)/60:.2f} minutes\n")


