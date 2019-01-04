from __future__ import print_function
import argparse
import os
import math
import numpy as np
import numpy.random as npr
import scipy.misc
import time
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg') # switch backend
import matplotlib.pyplot as plt 


from load_data import load_cifar10
from preprocessing import *
from models import *
import unet
import generator_copy


def plot(input, gtlabel, output, colours, path):
    """
    Generate png plots of input, ground truth, and outputs

    Args:
      input: the greyscale input to the colourization CNN
      gtlabel: the grouth truth categories for each pixel
      output: the predicted categories for each pixel
      colours: numpy array of colour categories and their RGB values
      path: output path
    """
    grey = np.transpose(input[:10,:,:,:], [0,2,3,1])
    gtcolor = get_cat_rgb(gtlabel[:10,0,:,:], colours)
    predcolor = get_cat_rgb(output[:10,0,:,:], colours)

    img = np.vstack([
      np.hstack(np.tile(grey, [1,1,1,3])),
      np.hstack(gtcolor),
      np.hstack(predcolor)])
    scipy.misc.toimage(img, cmin=0, cmax=1).save(path)


experiment = "Unet_256_all"
categories = [cat]
model = "UNet" # "CNN", "DUNet", "UNet"
batch_size = 100
plot_images = True
n_epochs = 50
save_model = True
model_path = os.path.join("./models", experiment)
if not os.path.exists(model_path):
	os.makedirs(model_path)
validation = False # inference
gpu = True
if gpu:
	device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
else:
	device = "cpu"
print(device)
num_filters = 128 
kernel_size = 3
seed = 0
# optimizer
lr = 2.0e-4
beta1 = 0.5
beta2 = 0.999



(x_train, y_train), (x_test, y_test) = load_cifar10()


x_train_lab, y_train_lab = process_lab(x_train, y_train, categories=categories)

grey = np.transpose(x_train_lab[:10,:,:,:], [0,2,3,1])
lab = np.transpose(y_train_lab[:10,:,:,:], [0,2,3,1])

print(grey.shape)
print(lab.shape)
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

#