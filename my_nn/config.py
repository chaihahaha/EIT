import torch
import numpy as np

######################
#  General settings  #
######################

# Filename to save the model under
filename_out    = 'checkpoints/my_inn.ckpt'
# Model to load and continue training. Ignored if empty string
filename_in     = ''
# Compute device to perform the training on, 'cuda' or 'cpu'
device          = 'cuda'

test_time_functions = []

#######################
#  Training schedule  #
#######################

# Initial learning rate
lr_init         = 1e-3
#Batch size
batch_size      = 1
# Total number of epochs to train for
n_epochs        = 10
# Saving frequency
save_freq       = 1000
# End the epoch after this many iterations (or when the train loader is exhausted)
n_its_per_epoch = 100
# For the first n epochs, train with a much lower learning rate. This can be
# helpful if the model immediately explodes.
pre_low_lr      = 0
# Decay exponentially each epoch, to final_decay*lr_init at the last epoch.
final_decay     = 0.01
# L2 weight regularization of model parameters
l2_weight_reg   = 1e-5
# Parameters beta1, beta2 of the Adam optimizer
adam_betas = (0.9, 0.95)

#####################
#  Data Loader      #
#####################
dataset_size = 2000
x_data = torch.Tensor(np.load('dataImages.npy')[:dataset_size])
_, x_height, x_width = x_data.shape
print(x_data.shape)
y_data = torch.Tensor(np.load('dataBoundary.npy')[:dataset_size])
_, y_dim = y_data.shape

test_split = 100
x_test = x_data[-test_split:]
y_test = y_data[-test_split:]

x_train = x_data[:-test_split]
y_train = y_data[:-test_split]
test_batch_size = min(batch_size, test_split)

test_loader = torch.utils.data.DataLoader(
  torch.utils.data.TensorDataset(x_test, y_test),
  batch_size=test_batch_size, shuffle=False, drop_last=True)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_train, y_train),
    batch_size=batch_size, shuffle=True, drop_last=True)

############
#  Losses  #
############

train_forward_l2    = True

###########
#  Model  #
###########

# Initialize the model parameters from a normal distribution with this sigma
init_scale = 0.0000010
