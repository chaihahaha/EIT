import torch
import numpy as np
import os
import random
from PIL import Image
class MyDataSets(torch.utils.data.Dataset):
    def __init__(self, flist):
        self.imgs = np.load(flist+".npy")
        with open(flist, 'r') as fp:
            self.img_path_list = fp.readlines()
    def __getitem__(self, index):
        img_path = self.img_path_list[index].split("\n")[0]
        splited_path = img_path.split("/")
        dir_path = "/".join(splited_path[:-1])
        img_name = splited_path[-1]
        prefix = ".".join(img_name.split(".")[:-1])
        sig, ex, nc, hl = prefix.split("_")
        label_path = "xxxxx.txt"
        # randomly get a label name
        while not os.path.isfile(label_path):
            noise = random.choice(["0","1","5","10","15"]) + "noise"
            label_name = "_".join(["V1",ex, nc, noise, hl]) + ".txt"
            label_path = dir_path + "/" + label_name

        # read label
        with open(label_path, "r") as fp:
            label = [float(x.split()[0]) for x in fp.readlines()]

        ## read img
        #img = Image.open(img_path)
        #left,upper,right,lower = 502, 98, 1686, 1282 # pour Adddata
        #img = img.crop((left,upper,right,lower)).resize((256,256), Image.ANTIALIAS).convert('L')
        img = self.imgs[index]
        img = torch.tensor(img, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        assert img.shape==(256,256)
        assert len(label) == 256
        return img, label
    def __len__(self):
        return len(self.img_path_list)

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
lr_init         = 1e-4
#Batch size
batch_size      = 80
#Test batch size
test_batch_size = 40
# Total number of epochs to train for
n_epochs        = 1001
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

#x_train = torch.Tensor(np.load('trainImages.npy'))
#_, x_height, x_width = x_train.shape
#print(x_train.shape)
#y_train = torch.Tensor(np.load('trainBoundary.npy'))
#_, y_dim = y_train.shape
#
#x_test = torch.Tensor(np.load('testImages.npy'))
#y_test = torch.Tensor(np.load('testBoundary.npy'))
y_dim = 256
x_height, x_width = 256, 256
train_dataset = MyDataSets('trainFilesList.csv')
test_dataset = MyDataSets('testFilesList.csv')

test_loader = torch.utils.data.DataLoader(
  test_dataset,
  batch_size=test_batch_size, shuffle=False, drop_last=True)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size, shuffle=True, drop_last=True)

############
#  Losses  #
############

train_forward_l2    = 1
train_forward_mmd    = 0

###########
#  Model  #
###########

# Initialize the model parameters from a normal distribution with this sigma
init_scale = 1e-4
