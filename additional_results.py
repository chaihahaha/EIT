import numpy as np
import warnings
warnings.filterwarnings("ignore")

path = './Additional_Results/'

import glob
images = sorted(glob.glob(path + '*.png'))
txt = sorted(glob.glob(path + '*.txt'))

import pandas as pd
df = pd.DataFrame(list(zip(images, txt)),
               columns =['image', 'txt'])
df.to_csv('./filesList.csv')

# Images
from PIL import Image
basewidth = 256
import os

img = Image.open(images[0])
wpercent = (basewidth/float(img.size[0]))
hsize = int((float(img.size[1])*float(wpercent)))
im = []
for filename in images:
    img = Image.open(filename)
    img1 = img.resize((basewidth,hsize), Image.ANTIALIAS).convert('L')
    im.append(np.array(img1))
    img.close()
dataImages=np.array(im)
np.save('./dataImages.npy',dataImages)

# Boundaries
names=[]
for filename in txt:
    names.append([x.split(' ')[0] for x in open(filename).readlines()])
dataBoundary = np.array(names)
dataBoundary=dataBoundary.astype(np.float)

np.save('./dataBoundary.npy',dataBoundary)
