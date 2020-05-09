import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable
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

class MyModel(nn.Module):
    def __init__(self, y_dim, x_height, x_width):
        super(MyModel, self).__init__()

        hidden_channels = 2
        hidden_sizes = 256

        self.one_d = nn.Sequential(
                nn.Linear(y_dim, hidden_sizes),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(hidden_sizes, hidden_channels * x_height * x_width))
        self.view1 = lambda x: x.view(-1, hidden_channels, x_height, x_width)

        self.two_d = nn.Sequential(
                nn.Conv2d(hidden_channels, 1, 3, 1, 1, bias=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(1, 1, 3, 1, 1, bias=True))
        self.view2 = lambda x: x.view(-1, x_height, x_width)

    def forward(self, y):
        y = self.one_d(y)
        out = self.view1(y)
        out = self.two_d(out)
        out = self.view2(out)
        return out



model = MyModel(c.y_dim, c.x_height, c.x_width)
model.to(c.device)
params_trainable = list(filter(lambda p: p.requires_grad, model.parameters()))
for p in params_trainable:
    p.data = c.init_scale * torch.randn(p.data.shape).to(c.device)

gamma = (c.final_decay)**(1./c.n_epochs)
optim = torch.optim.Adam(params_trainable, lr=c.lr_init, betas=c.adam_betas, eps=1e-6, weight_decay=c.l2_weight_reg)
weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=gamma)

if c.filename_in:
    load(c.filename_in)
