import torch
import numpy as np
import os

######################
#  General settings  #
######################

# Filename to save the model under
ckpt_dir        = 'checkpoints/'
ckpt_name       = 'my_inn.ckpt'
filename_out    = ckpt_dir + ckpt_name
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
# Model to load and continue training. Ignored if empty string
filename_in     = ''#filename_out
# Compute device to perform the training on, 'cuda' or 'cpu'
device          = 'cuda'

test_time_functions = []

#######################
#  Training schedule  #
#######################

# Initial learning rate
lr_init         = 1e-3
#Batch size
batch_size      = 40
# Total number of epochs to train for
n_epochs        = 301
# Saving frequency
save_freq       = 200
# End the epoch after this many iterations (or when the train loader is exhausted)
n_its_per_epoch = 100
# For the first n epochs, train with a much lower learning rate. This can be
# helpful if the model immediately explodes.
pre_low_lr      = 0
# Decay exponentially each epoch, to final_decay*lr_init at the last epoch.
final_decay     = 0.5
# L2 weight regularization of model parameters
l2_weight_reg   = 1e-5
# Parameters beta1, beta2 of the Adam optimizer
adam_betas = (0.9, 0.95)

#####################
#  Data Loader      #
#####################
x_train = torch.Tensor(np.load('trainImages.npy'))
_, x_height, x_width = x_train.shape
print(x_train.shape)
y_train = torch.Tensor(np.load('trainBoundary.npy'))
_, y_dim = y_train.shape

x_test = torch.Tensor(np.load('testImages.npy'))
y_test = torch.Tensor(np.load('testBoundary.npy'))

test_batch_size = batch_size 

test_loader = torch.utils.data.DataLoader(
  torch.utils.data.TensorDataset(x_test, y_test),
  batch_size=test_batch_size, shuffle=False, drop_last=True)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_train, y_train),
    batch_size=batch_size, shuffle=True, drop_last=True)

############
#  Losses  #
############

train_forward_l2    = 0.01
train_forward_mmd    = 0.99

###########
#  Model  #
###########

# Initialize the model parameters from a normal distribution with this sigma
init_scale = 0.0000010
