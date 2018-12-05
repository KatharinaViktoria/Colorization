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
# Models
from pix2pix_models import *
from generator import unet


### TODO LIST ###
# 
# 2. load training data
# extra: after each epoch run inference on a random batch of training data and save images
# grab batch, run inference
# convert predictions from Lab to RGB 
# save as .jpg

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

alpha = 1
gamma = 100 # ref Jay

# optimizer
lr = 2.0e-4
beta1 = 0.5
beta2 = 0.999
G_optimizer = optim.Adam(G.parameters(),lr=lr,betas=(beta1,beta2))
D_optimizer = optim.Adam(D.parameters(),lr=lr,betas=(beta1,beta2))

# log file, model dir
model_dir = os.path.join(os.getcwd(),"baseline")
if not os.path.exists(model_dir):
	os.mkdirs(model_dir)

n_epochs = 100
# ToDO
n_batches = 2 # TODO

with open(os.path.join(model_dir,'losses.csv'), 'w') as writefile:

		writer = csv.writer(writefile)
		writer.writerow(['epoch', 'batch_num', 'D loss', 'G loss'])
		writefile.flush()
		# command to write a new row in log file
		# writer.writerow([epoch, batch_num, running_loss, top_5_pos, count_all])
		# writefile.flush()

		for epoch in range(n_epochs):

			# ToDo: load data
			# get x 
			# get y_col = y_real

			for batch_num in range(n_batches)
				## Train Discriminator
				# Real input
				D_optimizer.zero_grad()
				D_pred_real = D.forward(y_real)
				D_real_loss = criterion_GAN(D_pred_real, label_real)  # ones = true
				D_real_loss.backward()
				D_optimizer.step()

				# Fake input
				D_optimizer.zero_grad()
				y_fake = G.forward(x, mode='colorization')
				D_pred_fake = D.forward(y_fake)
				D_fake_loss = criterion_GAN(D_pred_fake, label_fake)  # zeros = false/fake
				D_fake_loss.backward()
				D_optimizer.step()


				# Train Generator
				G_optimizer.zero_grad()
				y_fake = G.forward(y_real, mode='colorization')
				G_GAN_loss = criterion_GAN(y_fake, label_real)
				G_L1 = criterion_L1(y_fake,y_real)
				G_loss = alpha*G_GAN_loss + gamma*G_L1
				G_loss.backward()
				G_optimizer.step()

				# logger
				if batch%50 == 0:
					writer.writerow([epoch, batch_num, (D_fake_loss+D_real_loss), G_loss])
					writefile.flush()


			## save model
			if epoch>10:
				savefile = "model.%d" % epoch
				torch.save(model.state_dict(),os.path.join(model_dir, savefile))

				## don't use validation for now


			## 
			gc.collect()



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