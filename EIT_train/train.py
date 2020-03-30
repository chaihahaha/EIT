import numpy as np
from sklearn.decomposition import PCA
from time import time

import torch
import torch.nn as nn
import torch.optim
import numpy as np
import matplotlib.pyplot as plt

from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom

def subnet_fc(c_in, c_out):
    return nn.Sequential(nn.Linear(c_in, hidden),nn.ReLU(),
                         nn.Linear(hidden,  c_out))
def subnet_conv(c_in, c_out):
    return nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=1)

def MMD_multiscale(x, y):
    xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2.*xx
    dyy = ry.t() + ry - 2.*yy
    dxy = rx.t() + ry - 2.*zz

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    for a in [0.05, 0.2, 0.9]:
        XX += a**2 * (a**2 + dxx)**-1
        YY += a**2 * (a**2 + dyy)**-1
        XY += a**2 * (a**2 + dxy)**-1

    return torch.mean(XX + YY - 2.*XY)


def fit(input, target):
    return torch.mean((input - target)**2)

def train(i_epoch=0):
    model.train()

    l_tot = 0
    batch_idx = 0

    t_start = time()

    # If MMD on x-space is present from the start, the model can get stuck.
    # Instead, ramp it up exponetially.
    loss_factor = min(1., 2. * 0.002**(1. - (float(i_epoch) / n_epochs)))

    for x, y in train_loader:
        batch_idx += 1
        if batch_idx > n_its_per_epoch:
            break

        x, y = x.to(device), y.to(device)

        y_clean = y.clone()
        pad_x = zeros_noise_scale * torch.randn(batch_size, ndim_tot -
                                                ndim_x, device=device)
        pad_yz = zeros_noise_scale * torch.randn(batch_size, ndim_tot -
                                                 ndim_y - ndim_z, device=device)

        y += y_noise_scale * torch.randn(batch_size, ndim_y, dtype=torch.float, device=device)

        x, y = (torch.cat((x, pad_x),  dim=1),
                torch.cat((torch.randn(batch_size, ndim_z, device=device), pad_yz, y),
                          dim=1))


        optimizer.zero_grad()

        # Forward step:

        output = model(x)

        # Shorten output, and remove gradients wrt y, for latent loss
        y_short = torch.cat((y[:, :ndim_z], y[:, -ndim_y:]), dim=1) # remove rand padding

        l = lambd_predict * loss_fit(output[:, ndim_z:], y[:, ndim_z:]) # apply loss fit on y

        output_block_grad = torch.cat((output[:, :ndim_z],           # remove rand, same shape as y_short
                                       output[:, -ndim_y:].data), dim=1)

        l += lambd_latent * loss_latent(output_block_grad, y_short) # apply Latent Loss on latent output and output y
        l_tot += l.data.item()

        # l is for z latent loss and y prediction loss
        l.backward()

        # Backward step:
        pad_yz = zeros_noise_scale * torch.randn(batch_size, ndim_tot -
                                                 ndim_y - ndim_z, device=device)
        y = y_clean + y_noise_scale * torch.randn(batch_size, ndim_y, device=device)

        orig_z_perturbed = (output.data[:, :ndim_z] + y_noise_scale *
                            torch.randn(batch_size, ndim_z, device=device))
        y_rev = torch.cat((orig_z_perturbed, pad_yz,
                           y), dim=1)
        y_rev_rand = torch.cat((torch.randn(batch_size, ndim_z, device=device), pad_yz,
                                y), dim=1)

        output_rev = model(y_rev, rev=True)
        output_rev_rand = model(y_rev_rand, rev=True)

        l_rev = (
            lambd_rev
            * loss_factor
            * loss_backward(output_rev_rand[:, :ndim_x],
                            x[:, :ndim_x]) #  apply loss backward on recovered x and real x
        )

        l_rev += lambd_predict * loss_fit(output_rev, x) # apply loss fit on recovered x and real x

        l_tot += l_rev.data.item()
        l_rev.backward()

        for p in model.parameters():
            p.grad.data.clamp_(-15.00, 15.00)

        optimizer.step()

    return l_tot / batch_idx

def predict(x_samps):
    return model(torch.cat((x_samps, torch.zeros(N_samp, ndim_tot - ndim_x)),
                                 dim=1).to(device))[:,-ndim_y:]
def error(y1,y2):
    return torch.mean((y1**2-y2**2)/y2**2)/len(y1)

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

norm_factor_img = np.std(img)
norm_factor_bdr = np.std(boundary)
img, boundary = torch.from_numpy(img).float()/norm_factor_img, torch.from_numpy(boundary).float()/norm_factor_bdr
device = 'cuda' if torch.cuda.is_available() else 'cpu'
test_split = 100

# set up dim
ndim_x = img.shape[1]
ndim_y = boundary.shape[1]
ndim_z = 1000
ndim_tot = ndim_x + ndim_y + ndim_z


# set up network structure
hidden = 2*ndim_tot
nodes = [InputNode(ndim_tot, name='input')]

for k in range(2):
    nodes.append(Node(nodes[-1],
                      GLOWCouplingBlock,
                      {'subnet_constructor':subnet_fc, 'clamp':2.0},
                      name=F'coupling_{k}'))
    nodes.append(Node(nodes[-1],
                      PermuteRandom,
                      {'seed':k},
                      name=F'permute_{k}'))

nodes.append(OutputNode(nodes[-1], name='output'))

model = ReversibleGraphNet(nodes, verbose=False)

# Training parameters
n_epochs = 5000
n_its_per_epoch = 2
batch_size = 900 - test_split
save_freq = 100

lr = 1e-3
l2_reg = 6e-5

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

try:
    t_start = time()
    for i_epoch in range(n_epochs):
        print("Epoch:",i_epoch,"Loss:",train(i_epoch))
        if i_epoch%save_freq == 0:
            torch.save(model.state_dict(), "model.torch")
except KeyboardInterrupt:
    pass
finally:
    print(f"\n\nTraining took {(time()-t_start)/60:.2f} minutes\n")

N_samp = 10

x_samps = img[:N_samp]
y_samps = boundary[:N_samp]
print(x_samps.shape,y_samps.shape)
y_samps += y_noise_scale * torch.randn(N_samp, ndim_y)
y_samps = torch.cat([torch.randn(N_samp, ndim_z),
                     zeros_noise_scale * torch.zeros(N_samp, ndim_tot - ndim_y - ndim_z),
                     y_samps], dim=1)
y_samps = y_samps.to(device)
y_p=predict(x_samps)
print("error of forward prediction:",error(y_p,y_samps[:,-ndim_y:]).item())
for sample_index in range(N_samp):
    sample_boundary = y_samps[sample_index,:].view(1,-1)
    sample_img = x_samps[sample_index,:].view(1,-1).detach().numpy()*norm_factor_img
    recovered_img = model(sample_boundary,rev=True)[:,:ndim_x].cpu().detach().numpy()*norm_factor_img
    fig,axs = plt.subplots(1,2)
    axs[0].imshow(pca.inverse_transform(recovered_img).reshape(original_shape))
    axs[0].set_title('recovered img')
    axs[1].imshow(pca.inverse_transform(sample_img).reshape(original_shape))
    axs[1].set_title('original img')
    fig.savefig("Result.png")
    fig.clf()
