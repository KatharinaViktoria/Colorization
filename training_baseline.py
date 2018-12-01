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
# 0. setup: GPU
# 1. generate discriminator (PatchGAN) and generator (mode = 'colorization')
# 2. load training data
# 3.1 define loss functions
# 3.2 optimizer (for both D and G), scheduler (?) 
# 4. training loop
# 4.1 log file: epoch, discriminator loss, generator losses (d and consistency) -> compare pix2pix?
# 4.2 discriminator: forward prop, loss  -> backprop, weight update
# 4.3 generator:  forward prop (! mode = 'colorization'), loss(es) -> backprop, weight update
# TODO: for baseline training set mode/architecture to colorization for all forward props!

# # 4.4 log losses, decide when to save models 

#---------------------------------------------------------------------------------

def gt_GAN_loss(batch_size, real, real_label=1.0, fake_label=0.0):
	if real:
		gt_tensor = torch.from_numpy(np.ones([batch_size,1])*real_label)
	else:
		gt_tensor = torch.from_numpy(np.ones([batch_size,1])*fake_label)

	return gt_tensor


#------------------SETUP----------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Discriminator (PatchGAN)
D = define_D(3, 64, 'n_layers', n_layers_D=3, norm='batch', use_sigmoid=True, init_type='normal', init_gain=0.02, gpu_ids=[0])
D = D.to(device)
print(D)

# Generator (U-Net)
G = unet()
G = G.to(device)
print(G)

# ToDO: Dataloader
batch_size = 100


# Loss
# get ground truth tensors/label for GAN loss
label_real = gt_GAN_loss(batch_size, True)
label_fake = gt_GAN_loss(batch_size, False)

criterion_GAN = torch.nn.BCELoss()
criterion_L1 = torch.nn.L1Loss()

# optimizer
lr = 2.0e-4
beta1 = 0.5
beta2 = 0.999
optim_G = optim.Adam(G.parameters(),lr=lr,betas=(beta1,beta2))
optim_D = optim.Adam(D.parameters(),lr=lr,betas=(beta1,beta2))

# log file, model dir
model_dir = os.path.join(os.getcwd(),"baseline")
if not os.path.exists(model_dir):
	os.mkdirs(model_dir)

with open(os.path.join(model_dir,'losses.csv'), 'w') as writefile:

        writer = csv.writer(writefile)
        # TODO: what do we have to keep track of?
        writer.writerow(['epoch', 'batch_num', 'training loss', 'top_5_ right', 'count all val'])
        writefile.flush()
        # command to write a new row in log file
        # writer.writerow([epoch, batch_num, running_loss, top_5_pos, count_all])
        # writefile.flush()





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



# --------------TEST FWD PROP GENERATOR----------------------------------------
'''
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
'''