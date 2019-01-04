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





"""
# TODOs
preprocessing that gives lab (lightness and ab channels) and classes
add shuffling before each epoch starts (we have numpy arrays...so it shouldn't bee too hard - use permutation o.s.)

training scheme: every other batch either colorization G.forward(x, mode="colorization")
or classification G.forward(x, mode="classification")

add classification loss as criterion_Class = categorical crossentropy
backprop

train for 100 epochs

log classification loss on validation and training dataset as well 
plot them -> we'll have 3 training plots: Discriminator, Generator, Classification (part of generator as well but I don't want to get into the business of changing the names)
"""

"""
COLORIZATION OF CIFAR-10 - GAN WITH ADDIDTIONAL CLASSIFICATION LOSS
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)





######################################################################
# Torch Helper
######################################################################

def get_torch_vars(xs, ys_col, ys_class, gpu=False):
	"""
	Helper function to convert numpy arrays to pytorch tensors.
	If GPU is used, move the tensors to GPU.
	"""

	xs = torch.from_numpy(xs).float()
	ys_col = torch.from_numpy(ys_col).float()
	ys_class = torch.from_numpy(ys_class).long()

	if gpu:
		xs = torch.tensor(xs, device=device)
		ys_col = torch.tensor(ys_col, device = device)
		ys_class = torch.tensor(ys_class, device = device)

	return Variable(xs), Variable(ys_col), Variable(ys_class)

def run_validation_step(G, D, criterion_GAN, criterion_L1, x, y_true, y_class, batch_size, plotpath=None):

	# evaluates both colorization and classification for every epoch

	correct = 0.0
	total = 0.0
	losses_col = []
	losses_class = []
	for i, (xs, ys_col, ys_class) in enumerate(get_batch_col_class(x_test_lab, y_test_lab,y_test_class, batch_size)):
		x, y_true, y_class = get_torch_vars(xs, ys_col, ys_class, True)

		# validation colorization
		y_fake = G.forward(x, mode='colorization')
		D_pred_fake = torch.mean(torch.mean(D.forward(y_fake), 2), 2)
		G_GAN_loss = criterion_GAN(D_pred_fake, label_real)
		G_L1 = criterion_L1(y_fake,y_true)
		G_loss = alpha*G_GAN_loss + gamma*G_L1
		losses_col.append(G_loss.data[0])

		# validation classification
		outputs = G.forward(x, mode = 'classification')
		loss_class = criterion_class(outputs,y_class)
		losses_class.append(loss_class.data[0])
		

	if plotpath: # only plot if a path is provided
		plot_lab(xs, ys_col, y_fake.detach().cpu().numpy(), plotpath)

	val_loss_col = np.mean(losses_col)
	val_loss_class = np.mean(losses_class)

	return val_loss_col, val_loss_class

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
	experiment = "proposed_architecture"
	# model = "UNet" # "CNN", "DUNet", "UNet"
	categories = [airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck]
	batch_size = 10
	plot_images = True
	n_epochs = 150
	save_model = True
	model_path = os.path.join("./models", experiment)
	if not os.path.exists(model_path):
		os.makedirs(model_path)
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
	num_colours = 2 # number of output channels

	#----------INITIALIZE NETWORKS, LOSSES, AND DATA-----------------------------
	# Discriminator 
	D = define_D(2, 64, 'n_layers', n_layers_D=3, norm='batch', use_sigmoid=True, init_type='normal', init_gain=0.02, gpu_ids=[0])
	D = D.to(device)
	print(D)

	# Generator
	G = generator_copy.unet()
	G = G.to(device)
	print(G)

	label_real = gt_GAN_loss(batch_size, True)
	label_fake = gt_GAN_loss(batch_size, False)

	criterion_GAN = torch.nn.BCELoss()
	criterion_L1 = torch.nn.L1Loss()
	criterion_class = torch.nn.CrossEntropyLoss()

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
	x_train_lab, y_train_lab, y_train_class = process_lab_class(x_train, y_train)
	print(x_train_lab.shape)
	print(y_train_lab.shape)
	print(y_train_class)
	x_test_lab, y_test_lab, y_test_class = process_lab_class(x_test, y_test)
	
	
	print("Beginning training ...")
	start = time.time()

	train_losses_D = []
	train_losses_G = []
	train_losses_class = []
	valid_losses_col = []
	valid_losses_class = []
	
	for epoch in range(n_epochs):
		# Train the Model
		D.train()
		G.train()

		losses_D = []
		losses_G = []
		losses_class = []
		
		# shuffle xs, ys_col, and ys_class at the beginning of each new epoch
		print("shuffle training data")
		p = npr.permutation(len(y_train_class))
		x_train_lab = x_train_lab[p]
		y_train_lab = y_train_lab[p]
		y_train_class = y_train_class[p]
		
		for i, (xs, ys_col, ys_class) in enumerate(get_batch_col_class(x_train_lab, y_train_lab, y_train_class, batch_size)):

			x, y_true, y_class = get_torch_vars(xs,ys_col,ys_class, True)
			
			# decide if training colorization or classification arm
			if i%2==0:
				classification = False
				colorization = True
			else:
				classification = True
				colorization = False
			"""
			---------------------------------------------------------------------------
			TRAINING
			if colorization: train discriminator and generator (colorization) as if for a usual GAN
				losses: GAN loss: binary crossentropy, colorization loss: L1
			if classification: only train classification arm of generator (encoder + fc layers)
				loss: categorical crossentropy
			---------------------------------------------------------------------------
			"""
			if colorization:
				# Train discriminator
				# real input
				D_optimizer.zero_grad()
				D_pred_real = torch.mean(torch.mean(D.forward(y_true), 2),2)
				D_real_loss = criterion_GAN(D_pred_real, label_real)  # ones = true
				D_real_loss.backward()
				D_optimizer.step()

				# fake input
				D_optimizer.zero_grad()
				y_fake = G.forward(x, mode='colorization')
				D_pred_fake = torch.mean(torch.mean(D.forward(y_fake), 2), 2)
				D_fake_loss = criterion_GAN(D_pred_fake, label_fake)  # zeros = false/fake
				D_fake_loss.backward()
				D_optimizer.step()

				D_loss = D_real_loss + D_fake_loss
				losses_D.append(D_loss.data[0])

				# Train generator
				G_optimizer.zero_grad()
				y_fake = G.forward(x, mode='colorization')
				D_pred_fake = torch.mean(torch.mean(D.forward(y_fake), 2), 2)
				G_GAN_loss = criterion_GAN(D_pred_fake, label_real)
				G_L1 = criterion_L1(y_fake,y_true)
				G_loss = alpha*G_GAN_loss + gamma*G_L1
				G_loss.backward()
				G_optimizer.step()
				
				losses_G.append(G_loss.data[0])

			if classification:
				G_optimizer.zero_grad()
				outputs = G.forward(x, mode = 'classification')
				loss = criterion_class(outputs,y_class)
				loss.backward()
				G_optimizer.step()
				losses_class.append(loss.data[0])
	
		
		
		# plot training images
		
		plot_lab(xs, ys_col, y_fake.detach().cpu().numpy(),
			 os.path.join('outputs', experiment,'train_%d.png' % epoch))

		# update losses for plotting
		avg_loss_G = np.mean(losses_G)
		avg_loss_D = np.mean(losses_D)

		train_losses_D.append(avg_loss_D)
		train_losses_G.append(avg_loss_G)

		avg_loss_class = np.mean(losses_class)
		train_losses_class.append(avg_loss_class)


		time_elapsed = time.time() - start
		print('Epoch [%d/%d], Loss_D: %.4f, Loss_G:, %.4f, Loss_class:, %.4f, Time (s): %d' % (
			epoch+1, n_epochs, avg_loss_D, avg_loss_G,avg_loss_class, time_elapsed))

		# Evaluate the model
		G.eval()
		D.eval()

		outfile = None

		if plot_images:
			outfile = os.path.join('outputs',experiment,'test_%d.png' % epoch)

		val_loss_col, val_loss_class = run_validation_step(G,D,
												criterion_GAN, criterion_L1,
												x_test_lab,
												y_test_lab,
												y_test_class,
												batch_size,
												outfile)

		time_elapsed = time.time() - start
		valid_losses_col.append(val_loss_col)
		valid_losses_class.append(val_loss_class)

		print('Epoch [%d/%d], Val Colorization Loss: %.4f, Val Classification Loss: %.4f, Time(s): %d' % (
			epoch+1, n_epochs, val_loss_col, val_loss_class, time_elapsed))
		
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

	plt.plot(train_losses_class, "ro-", label="Train")
	plt.plot(valid_losses_class, "go-", label="Validation")
	plt.legend()
	plt.title("Loss")
	plt.xlabel("Epochs")
	plt.savefig(os.path.join("outputs",experiment, "training_curve_classification.png"))
	plt.close()

	if save_model:
		print('Saving model...')
		torch.save(G.state_dict(), os.path.join(model_path,'model.weights'))
	
	