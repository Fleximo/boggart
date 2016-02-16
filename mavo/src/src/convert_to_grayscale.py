'''
Created on Oct 12, 2015

@author: Bhanu

@copyright: copyright (c), 2015 datamuni.com

'''
from PIL import Image
import numpy as np
import matplotlib.cm as cm
from matplotlib import pylab as plt
import os

fnames = os.listdir(path='../figures/')
for fname in fnames:
    if(fname.endswith('.png')):
        image=Image.open('../figures/%s'%fname).convert("L")
        image.save('../figures/grayed/%s'%fname)
#         arr=np.asarray(image)
#         plt.figimage(arr,cmap=cm.get_cmap("Greys_r"))
#         plt.savefig('../figures/grayed/%s'%fname)
#         plt.close()
#         image.close()