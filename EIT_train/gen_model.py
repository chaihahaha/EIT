from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom
import torch.nn as nn

def Model(ndim_tot):
    hidden = ndim_tot
    def subnet_fc(c_in, c_out):
        layer_in = nn.Linear(c_in, hidden)
        layer_out = nn.Linear(hidden,  c_out)
        nn.init.xavier_uniform(layer_in.weight)
        nn.init.xavier_uniform(layer_out.weight)
        layer = nn.Sequential(layer_in, nn.ReLU(), layer_out)
        return layer
                             
    def subnet_conv(c_in, c_out):
        return nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=1)

    # set up network structure
    nodes = [InputNode(ndim_tot, name='input')]

    for k in range(5):
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
    return model

