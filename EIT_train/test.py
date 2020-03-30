import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle as pk
from gen_model import *

# load pca from file
with open("pca.pkl", "rb") as f:
    pca = pk.load(f)

# load model from file
ndim_tot = 2156
model = Model(ndim_tot * 2, ndim_tot)
i_epoch = 1000
model.load_state_dict(torch.load("./checkpoint/model"+str(i_epoch)+".torch"))

N_samp = 10
img = np.load("dataImages.npy")
boundary = np.load("dataBoundary.npy")
x_samps = img[:N_samp]
y_samps = boundary[:N_samp]
y_samps = torch.cat([torch.randn(N_samp, ndim_z),
                     torch.zeros(N_samp, ndim_tot - ndim_y - ndim_z),
                     y_samps], dim=1)
model.to("cpu")
for sample_index in range(N_samp):
    sample_boundary = y_samps[sample_index,:].view(1,-1)
    sample_img = x_samps[sample_index,:,:]
    recovered_img = model(sample_boundary,rev=True)[:,:ndim_x].detach().numpy()*norm_factor_img
    fig,axs = plt.subplots(1,2)
    axs[0].imshow(pca.inverse_transform(recovered_img).reshape(original_shape))
    axs[0].set_title('recovered img')
    axs[1].imshow(sample_img)
    axs[1].set_title('original img')
    fig.savefig("Result"+str(sample_index)+".png")
    fig.clf()
