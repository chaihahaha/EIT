import numpy as np
import os
from PIL import Image
def get_img_dataset(flist):
    with open(flist, 'r') as fp:
        img_path_list = fp.readlines()
    left,upper,right,lower = 502, 98, 1686, 1282 # pour Adddata
    
    im = []
    for filename in img_path_list:
        filename = filename.split('\n')[0]
        img = Image.open(filename)
        img1 = img.crop((left,upper,right,lower)).resize((256,256), Image.ANTIALIAS).convert('L')
        im.append(np.array(img1))
        img.close()
    dataImages=np.array(im)
    np.save('./'+flist+'.npy',dataImages)
if __name__=='__main__':
    get_img_dataset("trainFilesList.csv")
    get_img_dataset("testFilesList.csv")
