import torch
import numpy as np

######################
#  General settings  #
######################

# Filename to save the model under
filename_out    = 'checkpoints/my_inn.ckpt'
# Model to load and continue training. Ignored if empty string
filename_in     = 'checkpoints/my_inn.ckpt'
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
n_epochs        = 1000
# Saving frequency
save_freq       = 50
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
y_dim = 256
x_height = 256
x_width = 256
###########
#  Losses  #
############

train_forward_l2    = 0.01
train_forward_mmd    = 0.99

###########
#  Model  #
###########

# Initialize the model parameters from a normal distribution with this sigma
init_scale = 0.0000010
