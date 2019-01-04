"""
outputs 10 example greyscale and lab color space images 
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg') # switch backend
import matplotlib.pyplot as plt 


from load_data import load_cifar10
from preprocessing import *
from models import *
import unet
import generator_copy

# DATA - loading and preprocessing 
(x_train, y_train), (x_test, y_test) = load_cifar10()
x_train_lab, y_train_lab = process_lab(x_train, y_train, categories=categories)

grey = np.transpose(x_train_lab[:10,:,:,:], [0,2,3,1])
lab = np.transpose(y_train_lab[:10,:,:,:], [0,2,3,1])


lab = np.concatenate((grey,lab), axis=3)
lab = np.hstack(lab)  
grey = np.hstack(grey)
 
grey = grey[:,:,0]
plt.figure(figsize=(30, 100))
plt.imshow(lab)
plt.savefig("./lab_color.png",dpi = 300)

plt.figure(figsize=(30, 100))
plt.imshow(grey, cmap='gray')
plt.savefig("./grey.png",dpi = 300)