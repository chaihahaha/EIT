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

        hidden_channels = 16

        self.one_d = nn.Sequential(
                nn.Linear(y_dim, y_dim//2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(y_dim//2, y_dim//4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(y_dim//4, y_dim//8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(y_dim//8, y_dim//16),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(y_dim//16, hidden_channels * x_height//16 * x_width//16)
                )
        self.view1 = lambda x: x.view(-1, hidden_channels, x_height//16, x_width//16)

        self.two_d = nn.Sequential(
                nn.ConvTranspose2d(hidden_channels, hidden_channels//2, 3, stride=2, padding=1, output_padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(hidden_channels//2, hidden_channels//4, 3, stride=2, padding=1, output_padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(hidden_channels//4, hidden_channels//8, 3, stride=2, padding=1, output_padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(hidden_channels//8, hidden_channels//16, 3, stride=2, padding=1, output_padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(1, 1, 3, 1, 1, bias=True),
                nn.Tanh())
        self.view2 = lambda x: x.view(-1, x_height, x_width)
        self.scaling = lambda x: (x+1) * 255.0/2.0

    def forward(self, y):
        out = self.one_d(y)
        out = self.view1(out)
        out = self.two_d(out)
        out = self.view2(out)
        out = self.scaling(out)
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
