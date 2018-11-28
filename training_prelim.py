# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

# models
from networks import *
from pix2pix_models import *


# to check if network parts can get build etc.


# set up environment 
''' Did not work for whatever reason .... don't want to spend more time on this that's wyh I just copy/pasted the pix2pix models and use their initialization
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using ', device)

#1. load patchDiscriminator
norm_layer = get_norm_layer(norm_type='batch')
D = create_D(device)
# print(D.weight.data)
print(D)
'''

D = define_D(3, 64, 'n_layers', n_layers_D=3, norm='batch', use_sigmoid=True, init_type='normal', init_gain=0.02, gpu_ids=[0])
print(D)

example = np.ones([1,3,32,32]).astype(float)
print(example.shape)
example = torch.from_numpy(example)
print(example.type())
example = example.type(torch.FloatTensor)
print(example.type())
output = D.forward(example)
print(output)
# x = x.type(torch.cuda.FloatTensor)