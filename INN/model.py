import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable

from FrEIA.framework import *
from FrEIA.modules import *

import config as c

def optim_step():
    #for p in params_trainable:
        #print(torch.mean(torch.abs(p.grad.data)).item())
    optim.step()
    optim.zero_grad()

def scheduler_step():
    #weight_scheduler.step()
    pass

def save(name):
    torch.save({'opt':optim.state_dict(),
                'net':model.state_dict()}, name)

def load(name):
    state_dicts = torch.load(name)
    model.load_state_dict(state_dicts['net'])

nodes = [InputNode(*(c.ndims_x), name='input')]
ndim_x = c.ndim_x
def subnet_fc(c_in, c_out):
    return nn.Sequential(nn.Linear(c_in, 256), nn.ReLU(),
                         nn.Linear(256,  c_out))

def subnet_conv(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, 16,   3, padding=1), nn.ReLU(),
                         nn.Conv2d(16,  c_out, 3, padding=1))

def subnet_conv_1x1(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, 16,   1), nn.ReLU(),
                         nn.Conv2d(16,  c_out, 1))

# Higher resolution convolutional part
#for k in range(2):
#    nodes.append(Node(nodes[-1],
#                         GLOWCouplingBlock,
#                         {'subnet_constructor':subnet_conv, 'clamp':1.2},
#                         name=F'conv_high_res_{k}'))
#    nodes.append(Node(nodes[-1],
#                         PermuteRandom,
#                         {'seed':k},
#                         name=F'permute_high_res_{k}'))
#
#nodes.append(Node(nodes[-1], IRevNetDownsampling, {}))
#
## Lower resolution convolutional part
#for k in range(4):
#    if k%2 == 0:
#        subnet = subnet_conv_1x1
#    else:
#        subnet = subnet_conv
#
#    nodes.append(Node(nodes[-1],
#                         GLOWCouplingBlock,
#                         {'subnet_constructor':subnet, 'clamp':1.2},
#                         name=F'conv_low_res_{k}'))
#    nodes.append(Node(nodes[-1],
#                         PermuteRandom,
#                         {'seed':k},
#                         name=F'permute_low_res_{k}'))

## Make the outputs into a vector, then split off 1/4 of the outputs for the
## fully connected part
#nodes.append(Node(nodes[-1], Flatten, {}, name='flatten'))
#split_node = Node(nodes[-1],
#                     Split1D,
#                     {'split_size_or_sections':(ndim_x // 4, 3 * ndim_x // 4), 'dim':0},
#                     name='split')
#nodes.append(split_node)

# Fully connected part
for k in range(2):
    nodes.append(Node(nodes[-1],
                         GLOWCouplingBlock,
                         {'subnet_constructor':subnet_fc, 'clamp':2.0},
                         name=F'fully_connected_{k}'))
    nodes.append(Node(nodes[-1],
                         PermuteRandom,
                         {'seed':k},
                         name=F'permute_{k}'))

## Concatenate the fully connected part and the skip connection to get a single output
#nodes.append(Node([nodes[-1].out0, split_node.out1],
#                     Concat1d, {'dim':0}, name='concat'))
nodes.append(OutputNode(nodes[-1], name='output'))

model = ReversibleGraphNet(nodes)

model.to(c.device)

params_trainable = list(filter(lambda p: p.requires_grad, model.parameters()))
for p in params_trainable:
    p.data = c.init_scale * torch.randn(p.data.shape).to(c.device)

gamma = (c.final_decay)**(1./c.n_epochs)
optim = torch.optim.Adam(params_trainable, lr=c.lr_init, betas=c.adam_betas, eps=1e-6, weight_decay=c.l2_weight_reg)
weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=gamma)

if c.filename_in:
    load(c.filename_in)
