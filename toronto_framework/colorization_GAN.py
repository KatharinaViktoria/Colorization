"""
Colourization of CIFAR-10 Horses via classification.
"""

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
import torch.optim as optim
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg') # switch backend
import matplotlib.pyplot as plt 


from load_data import load_cifar10
from preprocessing import *
from models import *
import unet
import generator_copy
from pix2pix_models_copy import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
######################################################################
# Torch Helper
######################################################################

def get_torch_vars(xs, ys, gpu=False):
	"""
	Helper function to convert numpy arrays to pytorch tensors.
	If GPU is used, move the tensors to GPU.
	"""

	xs = torch.from_numpy(xs).float()
	ys = torch.from_numpy(ys).float()
	# ys = torch.from_numpy(ys).long()
	if gpu:
		# xs = xs.cuda()
		# ys = ys.cuda()
		xs = torch.tensor(xs, device=device)
		ys = torch.tensor(ys, device = device)
	return Variable(xs), Variable(ys)

def run_validation_step(G, D, criterion_GAN, criterion_L1, x, y_true, batch_size,
						colour, plotpath=None):
	correct = 0.0
	total = 0.0
	losses = []
	for i, (xs, ys) in enumerate(get_batch(x_test_lab, y_test_lab, batch_size)):
		x, y_true = get_torch_vars(xs, ys, gpu)

		# y_fake = G.forward(x)
		y_fake = G.forward(x, mode='colorization')
		D_pred_fake = torch.mean(torch.mean(D.forward(y_fake), 2), 2)
		G_GAN_loss = criterion_GAN(D_pred_fake, label_real)
		G_L1 = criterion_L1(y_fake,y_true)
		G_loss = alpha*G_GAN_loss + gamma*G_L1
		
		losses.append(G_loss.data[0])

	if plotpath: # only plot if a path is provided
		plot_lab(xs, ys, y_fake.detach().cpu().numpy(), plotpath)

	val_loss = np.mean(losses)
	val_acc = 0
	# val_acc = 100 * correct / total
	return val_loss, val_acc

def gt_GAN_loss(batch_size, real, real_label=1.0, fake_label=0.0):
	if real:
		gt_tensor = torch.from_numpy(np.ones([batch_size,1])*real_label).float()
	else:
		gt_tensor = torch.from_numpy(np.ones([batch_size,1])*fake_label).float()
	gt_tensor = torch.tensor(gt_tensor, device=device)
	return gt_tensor

######################################################################
# MAIN
######################################################################

if __name__ == '__main__':

	# Set the maximum number of threads to prevent crash in Teaching Labs
	torch.set_num_threads(5)
	
	# SET ARGUMENTS-------------------------------------------------------
	experiment = "GAN__cutstomUNet_all"
	# model = "UNet" # "CNN", "DUNet", "UNet"
	categories = [airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck]
	batch_size = 10
	plot_images = True
	n_epochs = 50
	save_model = True
	model_path = os.path.join("./models", experiment)
	if not os.path.exists(model_path):
		os.makedirs(model_path)
	validation = False # inference
	num_filters = 128 
	kernel_size = 3
	seed = 0
	lr = 2.0e-4
	beta1 = 0.5
	beta2 = 0.999

	# Create the outputs folder if not created already
	if not os.path.exists(os.path.join("./outputs",experiment)):
		os.makedirs(os.path.join("./outputs",experiment))
	# Numpy random seed
	npr.seed(seed)

	# LOAD THE COLOURS CATEGORIES
	# colours = np.load(args.colours)[0]
	# colours = np.load('colours/colour_kmeans24_cat7.npy')[0]
	# num_colours = np.shape(colours)[0]
	num_colours = 2 # number of output channels

	#----------INITIALIZE NETWORKS, LOSSES, AND DATA-----------------------------
	# Discriminator 
	D = define_D(2, 64, 'n_layers', n_layers_D=3, norm='batch', use_sigmoid=True, init_type='normal', init_gain=0.02, gpu_ids=[0])
	D = D.to(device)
	print(D)

	# Generator
	# G = unet.UNet(n_channels=1, n_classes=2)
	G = generator_copy.unet()
	G = G.to(device)
	print(G)

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


	#  LOAD DATA
	print("Loading data...")
	(x_train, y_train), (x_test, y_test) = load_cifar10()

	# preprocess data to lab color space and gpu tensors
	print("Transforming data...") 
	x_train_lab, y_train_lab = process_lab(x_train, y_train, categories=categories)
	print(x_train_lab.shape)
	print(y_train_lab.shape)
	x_test_lab, y_test_lab = process_lab(x_test, y_test,categories=categories)

	
	print("Beginning training ...")

	
	start = time.time()

	train_losses_D = []
	train_losses_G = []
	valid_losses_G = []
	valid_accs = []
	
	for epoch in range(n_epochs):
		# Train the Model
		D.train()
		G.train()

		losses_D = []
		losses_G = []
		
		for i, (xs, ys) in enumerate(get_batch(x_train_lab,
											   y_train_lab,
											   batch_size)):
			x, y_true = get_torch_vars(xs,ys, True)

			# Train discriminator
			# real input
			D_optimizer.zero_grad()
			D_pred_real = torch.mean(torch.mean(D.forward(y_true), 2),2)
			D_real_loss = criterion_GAN(D_pred_real, label_real)  # ones = true
			D_real_loss.backward()
			D_optimizer.step()

			# fake input
			D_optimizer.zero_grad()
			# y_fake = G.forward(x)
			y_fake = G.forward(x, mode='colorization')
			D_pred_fake = torch.mean(torch.mean(D.forward(y_fake), 2), 2)
			D_fake_loss = criterion_GAN(D_pred_fake, label_fake)  # zeros = false/fake
			D_fake_loss.backward()
			D_optimizer.step()

			D_loss = D_real_loss + D_fake_loss
			losses_D.append(D_loss.data[0])

			# Train generator
			G_optimizer.zero_grad()
			# y_fake = G.forward(x)
			y_fake = G.forward(x, mode='colorization')
			D_pred_fake = torch.mean(torch.mean(D.forward(y_fake), 2), 2)
			G_GAN_loss = criterion_GAN(D_pred_fake, label_real)
			G_L1 = criterion_L1(y_fake,y_true)
			G_loss = alpha*G_GAN_loss + gamma*G_L1
			G_loss.backward()
			G_optimizer.step()
			
			losses_G.append(G_loss.data[0])
	
		
		# plot training images
		if plot_images:
			plot_lab(xs, ys, y_fake.detach().cpu().numpy(),
				 os.path.join('outputs', experiment,'train_%d.png' % epoch))

		
		# plot training images
		avg_loss_G = np.mean(losses_G)
		avg_loss_D = np.mean(losses_D)

		train_losses_D.append(avg_loss_D)
		train_losses_G.append(avg_loss_G)

		time_elapsed = time.time() - start
		print('Epoch [%d/%d], Loss_D: %.4f, Loss_G:, %.4f, Time (s): %d' % (
			epoch+1, n_epochs, avg_loss_D, avg_loss_G,time_elapsed))

		# Evaluate the model
		G.eval()
		D.eval()

		outfile = None

		if plot_images:
			outfile = os.path.join('outputs',experiment,'test_%d.png' % epoch)

		val_loss, val_acc = run_validation_step(G,D,
												criterion_GAN, criterion_L1,
												x_test_lab,
												y_test_lab,
												batch_size,
												colours,
												outfile)

		time_elapsed = time.time() - start
		valid_losses_G.append(val_loss)
		valid_accs.append(val_acc)
		print('Epoch [%d/%d], Val Loss: %.4f, Val Acc: %.1f%%, Time(s): %d' % (
			epoch+1, n_epochs, val_loss, val_acc, time_elapsed))
		
		if save_model:
			if epoch%5 == 0:
				print('Saving model...')
				torch.save(G.state_dict(), os.path.join(model_path,'model'+str(epoch)+'.weights'))
	
	# Plot training curve
	plt.plot(train_losses_G, "ro-", label="Train")
	plt.plot(valid_losses_G, "go-", label="Validation")
	plt.legend()
	plt.title("Loss")
	plt.xlabel("Epochs")
	plt.savefig(os.path.join("outputs",experiment, "training_curve_G.png"))
	plt.close()
	
	plt.plot(train_losses_D, "ro-", label="Train")
	plt.legend()
	plt.title("Loss")
	plt.xlabel("Epochs")
	plt.savefig(os.path.join("outputs",experiment, "training_curve_D.png"))
	plt.close()

	if save_model:
		print('Saving model...')
		torch.save(G.state_dict(), os.path.join(model_path,'model.weights'))
	
	