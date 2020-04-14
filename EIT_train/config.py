'''Global configuration for the experiments'''
import torch
import numpy as np
######################
#  General settings  #
######################

# Filename to save the model under
filename_out    = 'checkpoint/my_inn.ckpt'
# Model to load and continue training. Ignored if empty string
filename_in     = ''
# Compute device to perform the training on, 'cuda' or 'cpu'
device          = 'cuda'
# Use interactive visualization of losses and other plots. Requires visdom
interactive_visualization = True
# Run a list of python functions at test time after eacch epoch
# See toy_modes_train.py for reference example
test_time_functions = []

#######################
#  Training schedule  #
#######################

# Initial learning rate
lr_init         = 1.0e-3
#Batch size
batch_size      = 10
# Total number of epochs to train for
n_epochs        = 60
# End the epoch after this many iterations (or when the train loader is exhausted)
n_its_per_epoch = 200
# For the first n epochs, train with a much lower learning rate. This can be
# helpful if the model immediately explodes.
pre_low_lr      = 0
# Decay exponentially each epoch, to final_decay*lr_init at the last epoch.
final_decay     = 0.02
# L2 weight regularization of model parameters
l2_weight_reg   = 1e-5
# Parameters beta1, beta2 of the Adam optimizer
adam_betas = (0.9, 0.95)

#####################
#  Data Loader      #
#####################
x_data = torch.Tensor(np.load('dataImages.npy'))
x_data = x_data.view(len(x_data),-1)
y_data = torch.Tensor(np.load('dataBoundary.npy'))

test_split = 200
x_test = x_data[-test_split:]
y_test = y_data[-test_split:]

x_train = x_data[:-test_split]
y_train = y_data[:-test_split]
test_loader = torch.utils.data.DataLoader(
  torch.utils.data.TensorDataset(x_test, y_test),
  batch_size=batch_size, shuffle=False, drop_last=True)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_train, y_train),
    batch_size=batch_size, shuffle=True, drop_last=True)

#####################
#  Data dimensions  #
#####################
ndim_x      = x_train.shape[1]
ndim_pad_x  = 0

ndim_y      = y_train.shape[1]
ndim_z      = ndim_x - ndim_y
ndim_pad_zy  = 0


assert (ndim_x + ndim_pad_x
        == ndim_y + ndim_z + ndim_pad_zy), "Dimensions don't match up"

############
#  Losses  #
############

train_forward_mmd    = True
train_backward_mmd   = True
train_reconstruction = True
train_max_likelihood = False

lambd_fit_forw       = 1.
lambd_mmd_forw       = 50.
lambd_reconstruct    = 1.
lambd_mmd_back       = 500.
lambd_max_likelihood = 1.

# Both for fitting, and for the reconstruction, perturb y with Gaussian
# noise of this sigma
add_y_noise     = 5e-9
# For reconstruction, perturb z
add_z_noise     = 2e-9
# In all cases, perturb the zero padding
add_pad_noise   = 1e-9

# For noisy forward processes, the sigma on y (assumed equal in all dimensions).
# This is only used if mmd_back_weighted of train_max_likelihoiod are True.
y_uncertainty_sigma = 0.12 * 4

mmd_forw_kernels = [(0.2, 2), (1.5, 2), (3.0, 2)]
mmd_back_kernels = [(0.2, 0.1), (0.2, 0.5), (0.2, 2)]
mmd_back_weighted = False

###########
#  Model  #
###########

# Initialize the model parameters from a normal distribution with this sigma
init_scale = 0.0000010
#
N_blocks   = 4
#
exponent_clamping = 4.0
#
hidden_layer_sizes = 16
#
use_permutation = True
#
verbose_construction = False
