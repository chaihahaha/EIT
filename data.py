import numpy as np
import warnings
warnings.filterwarnings("ignore")

#path = './data/'
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
#left,upper,right,lower = 725, 75, 1750, 1143 # pour data
left,upper,right,lower = 6, 0, 1231, 1225 # pour Adddata
import os

im = []
for filename in images:
    img = Image.open(filename)
    img1 = img.crop((left,upper,right,lower)).resize((256,256), Image.ANTIALIAS).convert('L')
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
