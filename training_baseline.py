import os
import pdb
import pickle
import argparse

import warnings
warnings.filterwarnings("ignore")

# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# Numpy & Scipy imports
import numpy as np
import scipy
import scipy.misc

# Local imports
# TODO
# dataloader
# TODO
# models
from pix2pix_models import *
from generator import unet


### TODO LIST ###

# 1. generate discriminator (PatchGAN) and generator (mode = 'colorization')
# 2. load training data
# 3.1 define loss functions
# 3.2 optimizer, scheduler (?) 
# 4. training loop
# 4.1 log file: epoch, discriminator loss, generator losses (d and consistency) -> compare pix2pix?
# 4.2 discriminator: forward prop, loss  -> backprop, weight update
# 4.3 generator:  forward prop (! mode = 'colorization'), loss(es) -> backprop, weight update
# # 4.4 log losses, decide when to save models 


# ----- example code to extend or dischard
D = define_D(3, 64, 'n_layers', n_layers_D=3, norm='batch', use_sigmoid=True, init_type='normal', init_gain=0.02, gpu_ids=[0])
print(D)

#------------------TEST FWD PROP DISCRIMINATOR---------------------------------
'''
example = np.ones([1,3,32,32]).astype(float)
print(example.shape)
example = torch.from_numpy(example)
print(example.type())
example = example.type(torch.FloatTensor)
print(example.type())
output = D.forward(example)
print(output)
'''
#-------------------------------------------------------------------------------



# -----  test import and run of generator


# TODO: for baseline training set mode/architecture to colorization for all forward props!

model = unet()
model.cuda()
print(model)

example = np.ones([1,1,32,32]).astype(float)
example = torch.from_numpy(example)
example = example.type(torch.cuda.FloatTensor)

print(example.type())
print(example.size())
pred = model.forward(example, mode = 'colorization')
print(pred.size())