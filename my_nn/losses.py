import torch
import config as c
import numpy as np
def l2_fit(input, target):
    return torch.sum((input - target)**2) / c.batch_size
def MMD_matrix_multiscale(x, y, widths_exponents):
    x = x.view(x.shape[0],-1)
    y = y.view(y.shape[0],-1)
    xx, yy, xy = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = torch.clamp(rx.t() + rx - 2.*xx, 0, np.inf)
    dyy = torch.clamp(ry.t() + ry - 2.*yy, 0, np.inf)
    dxy = torch.clamp(rx.t() + ry - 2.*xy, 0, np.inf)

    XX, YY, XY = (torch.zeros(xx.shape).to(c.device),
                  torch.zeros(xx.shape).to(c.device),
                  torch.zeros(xx.shape).to(c.device))

    for C,a in widths_exponents:
        XX += C**a * ((C + dxx) / a)**-a
        YY += C**a * ((C + dyy) / a)**-a
        XY += C**a * ((C + dxy) / a)**-a

    return XX + YY - 2.*XY
mmd_back_kernels = [(0.2, 0.1), (0.2, 0.5), (0.2, 2)]
def forward_mmd(x0, x1):
    return torch.mean(MMD_matrix_multiscale(x0, x1, mmd_back_kernels))
