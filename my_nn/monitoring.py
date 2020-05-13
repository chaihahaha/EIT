import numpy as np
import config as c
def print_config():
    config_str = ""
    config_str += "="*80 + "\n"
    config_str += "Config options:\n\n"

    for v in dir(c):
        if v[0]=='_': continue
        s=eval('c.%s'%(v))
        config_str += "  {:25}\t{}\n".format(v,s)

    config_str += "="*80 + "\n"

    print(config_str)
class Visualizer:
    def __init__(self, loss_labels):
            self.n_losses = len(loss_labels)
            self.loss_labels = loss_labels
            self.counter = 0

            self.header = 'Epoch '
            for l in loss_labels:
                self.header += ' %15s' % (l)

    def update_losses(self, losses):
        if self.header:
            print(self.header,flush=True)
            self.header = None

        print('\r', '    '*20, end='', flush=True)
        line = '\r%6.3i' % (self.counter)
        for l in losses:
            line += '  %14.4f' % (l)

        print(line, flush=True)
        self.counter += 1
visualizer = None

def restart():
    global visualizer
    loss_labels = []

    if c.train_forward_l2:
        loss_labels += ['L_l2_fwd']
    if c.train_forward_mmd:
        loss_labels += ['L_mmd_fwd']

    loss_labels += [l + '(test)' for l in loss_labels]

    visualizer = Visualizer(loss_labels)

def show_loss(losses):
    visualizer.update_losses(losses)
def close():
    visualizer.close()
