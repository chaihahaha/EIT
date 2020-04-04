import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle as pk
from gen_model import *
import sys
import os

# load pca from file
with open("pca.pkl", "rb") as f:
    pca = pk.load(f)
with open("params.pkl","rb") as f:
    params = pk.load(f)
    (norm_factor_img, norm_factor_bdr, original_shape, ndim_x, ndim_y, ndim_z, ndim_tot, ckpt_dir) = params
# load model from file
model = Model(ndim_tot)
i_epoch = sys.argv[1]
checkpoint = torch.load(ckpt_dir + "model"+i_epoch+".torch")
model.load_state_dict(checkpoint['model_state_dict'])
img_dir = "./img/"
if not os.path.exists(img_dir):
    os.makedirs(img_dir)

N_samp = 10
img = np.load("dataImages.npy")
boundary = np.load("dataBoundary.npy")
x_samps = img[-N_samp:]
y_samps = torch.from_numpy(boundary[-N_samp:]).float()
y_samps = torch.cat([torch.randn(N_samp, ndim_z),
                     torch.zeros(N_samp, ndim_tot - ndim_y - ndim_z),
                     y_samps], dim=1)
model.to("cuda")
for sample_index in range(N_samp):
    sample_boundary = y_samps[sample_index,:].view(1,-1)
    sample_img = x_samps[sample_index,:,:]
    recovered_img = model(sample_boundary.cuda(),rev=True)[:,:ndim_x].cpu().detach().numpy()*norm_factor_img
    fig,axs = plt.subplots(1,2)
    axs[0].imshow(pca.inverse_transform(recovered_img).reshape(original_shape))
    axs[0].set_title('recovered img')
    axs[1].imshow(sample_img)
    axs[1].set_title('original img')
    fig.savefig(img_dir + "Result"+str(sample_index)+".png")
    fig.clf()
