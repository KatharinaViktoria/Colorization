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


######################################################################
# Torch Helper
######################################################################

def get_torch_vars(xs, ys, gpu=False):
	"""
	Helper function to convert numpy arrays to pytorch tensors.
	If GPU is used, move the tensors to GPU.

	Args:
	  xs (float numpy tenosor): greyscale input
	  ys (int numpy tenosor): categorical labels 
	  gpu (bool): whether to move pytorch tensor to GPU
	Returns:
	  Variable(xs), Variable(ys)
	"""
	xs = torch.from_numpy(xs).float()
	ys = torch.from_numpy(ys).long()
	# ys = torch.from_numpy(ys).long()
	if gpu:
		# xs = xs.cuda()
		# ys = ys.cuda()
		xs = torch.tensor(xs, device=device)
		ys = torch.tensor(ys, device = device)
	return Variable(xs), Variable(ys)

def run_validation_step(cnn, criterion, x_test_lab, y_test_lab, batch_size):
	correct = 0.0
	total = 0.0
	losses = []
	for i, (xs, ys) in enumerate(get_batch_classification(x_test_lab, y_test_lab, batch_size)):
		images, labels = get_torch_vars(xs, ys, gpu)
		outputs = cnn.forward(images, mode='classification')
		val_loss = criterion(outputs,labels)
		losses.append(val_loss.data[0])

		# _, predicted = torch.max(outputs.data, 1, keepdim=True)
		# total += labels.size(0) * 32 * 32
		# correct += (predicted == labels.data).sum()

	val_loss = np.mean(losses)
	val_acc = 0
	# val_acc = 100 * correct / total
	return val_loss, val_acc


######################################################################
# MAIN
######################################################################

if __name__ == '__main__':
	# Set the maximum number of threads to prevent crash in Teaching Labs
	torch.set_num_threads(5)
	
	# SET ARGUMENTS
	experiment = "Unet_custom_class"
	# categories = [airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck]
	model = "UNet_class" # "CNN", "DUNet", "UNet"
	batch_size = 100
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

	# Create the outputs folder if not created already
	if not os.path.exists(os.path.join("./outputs",experiment)):
		os.makedirs(os.path.join("./outputs",experiment))
	# Numpy random seed
	npr.seed(seed)

	# LOAD THE COLOURS CATEGORIES
	colours = np.load('colours/colour_kmeans24_cat7.npy')[0] # not neede but will break the code if removed....
	num_colours = 2

	# LOAD THE MODEL
	'''
	if model == "CNN":
		cnn = CNN(kernel_size, num_filters, num_colours)
	elif model == "UNet":
		cnn = UNet(kernel_size, num_filters, num_colours)
	else: # model == "DUNet":
		cnn = DilatedUNet(kernel_size, num_filters, num_colours)
	'''
	# cnn = unet.UNet(n_channels=1, n_classes=2)
	cnn = generator_copy.unet()
	print(cnn)

	# LOSS FUNCTION
	criterion = nn.CrossEntropyLoss().to(device)
	optimizer = torch.optim.Adam(cnn.parameters(), lr=lr, betas=(beta1, beta2))

	# DATA
	print("Loading data...")
	(x_train, y_train), (x_test, y_test) = load_cifar10()

	print("Transforming data...")
	x_train, y_train = process_classification(x_train,y_train)
	x_test, y_test = process_classification(x_test, y_test)
	
	print(x_train[1:10,:,:,:])
	print(y_train[1:10])

	# Run validation only
	# if args.valid:
	
	'''
	if validation:
		if not save_model:
			raise ValueError("You need to give trained model to evaluate")

		print("Loading checkpoint...")
		cnn.load_state_dict(torch.load(args.checkpoint, map_location=lambda storage, loc: storage))
		img_path = "outputs/eval_%s.png" % args.model
		val_loss, val_acc = run_validation_step(cnn,
												criterion,
												test_grey,
												test_rgb_cat,
												args.batch_size,
												colours,
												img_path)
		print('Evaluating Model %s: %s' % (args.model, args.checkpoint))
		print('Val Loss: %.4f, Val Acc: %.1f%%' % (val_loss, val_acc))
		print('Sample output available at: %s' % img_path)
		exit(0)
	'''
	
	print("Beginning training ...")

	# if args.gpu: cnn.cuda()
	cnn.to(device)
	start = time.time()

	train_losses = []
	valid_losses = []
	valid_accs = []
	# for epoch in range(args.epochs):
	for epoch in range(n_epochs):
		# Train the Model
		cnn.train() # Change model to 'train' mode
		losses = []

		for i, (xs, ys) in enumerate(get_batch_classification(x_train,
											   y_train,
											   batch_size)):
			# images, labels = get_torch_vars(xs, ys, args.gpu)
			images, labels = get_torch_vars(xs,ys, True)
			# Forward + Backward + Optimize
			optimizer.zero_grad()

			outputs = cnn.forward(images, mode = 'classification')
			loss = criterion(outputs,labels)
			loss.backward()
			optimizer.step()
			losses.append(loss.data[0])
		
		
		avg_loss = np.mean(losses)
		train_losses.append(avg_loss)
		time_elapsed = time.time() - start
		print('Epoch [%d/%d], Loss: %.4f, Time (s): %d' % (
			epoch+1, n_epochs, avg_loss, time_elapsed))

		# Evaluate the model
		cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

		val_loss, val_acc = run_validation_step(cnn,
												criterion,
												x_test,
												y_test,
												batch_size)

		time_elapsed = time.time() - start
		valid_losses.append(val_loss)
		valid_accs.append(val_acc)
		print('Epoch [%d/%d], Val Loss: %.4f, Val Acc: %.1f%%, Time(s): %d' % (
			epoch+1, n_epochs, val_loss, val_acc, time_elapsed))
		
		if save_model:
			if epoch%5 == 0:
				print('Saving model...')
				torch.save(cnn.state_dict(), os.path.join(model_path,'model'+str(epoch)+'.weights'))

	# Plot training curve
	plt.plot(train_losses, "ro-", label="Train")
	plt.plot(valid_losses, "go-", label="Validation")
	plt.legend()
	plt.title("Loss")
	plt.xlabel("Epochs")
	plt.savefig(os.path.join("outputs", experiment, "training_curve.png"))
	plt.close()
	
	# if args.checkpoint:
	if save_model:
		print('Saving model...')
		torch.save(cnn.state_dict(), os.path.join(model_path,'model.weights'))
	