import numpy as np
from scipy.ndimage import geometric_transform
def c2p(out_coords):
    r_idx, theta_idx = out_coords[0], out_coords[1]
    r     = r_idx * 600.0 / h
    theta = theta_idx * 2 * np.pi / w
    dx    = r * np.cos(theta)
    dy    = r * np.sin(theta)
    x_idx = (circ_x + dx) * h / original_size
    y_idx = (circ_y + dy) * w / original_size
    return (x_idx, y_idx)

imgs = np.load("dataImages.npy")
imgs_polar = np.zeros_like(imgs)
b, h, w = imgs.shape
circ_x, circ_y = (614, 618)
original_size = 1225
for i in range(b):
    imgs_polar[i] = geometric_transform(imgs[i], c2p)
#for i in range(c):
#    for r_idx in range(h):
#        for theta_idx in range(w):
#            r     = r_idx * 600.0 / h
#            theta = theta_idx * 2 * np.pi / w
#            dx    = r * np.cos(theta)
#            dy    = r * np.sin(theta)
#            x_idx = int((circ_x + dx) * h / original_size)
#            y_idx = int((circ_y + dy) * w / original_size)
#            imgs_polar[i, r_idx, theta_idx] = imgs[i, x_idx, y_idx]
np.save("dataImagesPolar.npy", imgs_polar)
