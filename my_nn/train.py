from time import time

import torch
import numpy as np
from torch.autograd import Variable
import config as c
import monitoring
import losses
import model

def loss_forward_l2(out, x):
    return losses.l2_fit(out, x)

def train_epoch(i_epoch, test=False):

    if not test:
        model.model.train()
        loader = c.train_loader

    if test:
        model.model.eval()
        loader = c.test_loader
        nograd = torch.no_grad()
        nograd.__enter__()


    batch_idx = 0
    loss_history = []

    for x, y in loader:
        if batch_idx > c.n_its_per_epoch:
            break

        batch_losses = []

        batch_idx += 1
        x, y = Variable(x).to(c.device), Variable(y).to(c.device)

        out = model.model(y)
        if c.train_forward_l2:
            batch_losses.append(loss_forward_l2(out, x))

        l_total = sum(batch_losses)
        loss_history.append([l.item() for l in batch_losses])

        if not test:
            l_total.backward()
            model.optim_step()

    if test:

        if c.test_time_functions:
            out_x = model.model(cated_y, rev=True)
            for f in c.test_time_functions:
                f(out_x, out_y, x, cated_y)

        nograd.__exit__(None, None, None)

    return np.mean(loss_history, axis=0)

def main():
    monitoring.restart()

    try:
        monitoring.print_config()
        t_start = time()
        for i_epoch in range(-c.pre_low_lr, c.n_epochs):

            if i_epoch < 0:
                for param_group in model.optim.param_groups:
                    param_group['lr'] = c.lr_init * 1e-1

            train_losses = train_epoch(i_epoch)
            test_losses  = train_epoch(i_epoch, test=True)

            monitoring.show_loss(np.concatenate([train_losses, test_losses]))
            model.scheduler_step()
            if (i_epoch + 1) % c.save_freq == 0:
                model.save(c.filename_out)
    except:
        model.save(c.filename_out + '_ABORT')
        raise

    finally:
        print("\n\nTraining took %f minutes\n\n" % ((time()-t_start)/60.))
        model.save(c.filename_out)

if __name__ == "__main__":
    main()
