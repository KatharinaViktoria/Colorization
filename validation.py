"""
validation of baseline and our proposed architecture
prepare images for Turing test
"""

import os
import numpy as np
import numpy.random as npr
import scipy.misc
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from skimage import io, color
import scipy.misc

from load_data import load_cifar10
from preprocessing import *
from models import *
import generator

######################################################################
# Torch Helper
######################################################################

def get_torch_vars(xs, ys, gpu=False):

	xs = torch.from_numpy(xs).float()
	ys = torch.from_numpy(ys).float()

	if gpu:
	
		xs = torch.tensor(xs, device=device)
		ys = torch.tensor(ys, device = device)
	return Variable(xs), Variable(ys)


######################################################################

######################################################################

if __name__ == '__main__':

	# SETUP
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print("using ", device)

	categories = [airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck]
	batch_size = 10
	
	save_dir = "./inference_results"
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	

	# DATA

	print("Loading data...")
	(_, _), (x_test, y_test) = load_cifar10()
	x_test, y_test = process_lab(x_test, y_test)

	# shuffle
	print("shuffle training data")
	p = npr.permutation(x_test.shape[0])
	x_test = x_test[p]
	y_test = y_test[p]

	x_test_set = x_test[:100]
	y_test_set = y_test[:100]

	print(x_test.shape)
	print(x_test_set.shape)	


	# IMPORT MODELS
	print("...loading models...")

	# baseline model
	model_baseline = generator.unet()
	model_dir_baseline = "./models/GAN__cutstomUNet_all"
	model_number_baseline = 45 # number of best model
	model_baseline.load_state_dict(torch.load(os.path.join(model_dir_baseline,'model'+str(model_number_baseline)+'.weights')))
	model_baseline = model_baseline.to(device)
	model_baseline.eval()
	
	# combined model (proposed architecture classification and colorization)
	model_combined = generator.unet()
	model_dir_combined = "./models/proposed_architecture"
	model_number_combined = 145 # number of best model
	model_combined.load_state_dict(torch.load(os.path.join(model_dir_combined,'model'+str(model_number_combined)+'.weights')))
	model_combined = model_combined.to(device)
	model_combined.eval()

	
	#	INFERENCE 
	pred_baseline = np.zeros(y_test_set.shape)
	pred_combined = np.zeros(y_test_set.shape)

	for i, (xs, ys) in enumerate(get_batch(x_test_set, y_test_set, batch_size)):

		x, y_true = get_torch_vars(xs,ys, True)
		# baseline model
		y_baseline = model_baseline.forward(x, mode="colorization")
		pred_baseline[(i*10):((i+1)*10),:,:,:] = y_baseline.detach().cpu().numpy()

		# combined model
		y_combined = model_combined.forward(x, mode="colorization")
		pred_combined[(i*10):((i+1)*10),:,:,:] = y_combined.detach().cpu().numpy()

# POSTPROCESSING AND SAVING
	print(x.shape)
	for image_index in range(x_test_set.shape[0]):
		print(image_index)
		grey = np.transpose(x_test_set[image_index,:,:,:], [1,2,0])
		ground_truth = np.transpose(y_test_set[image_index,:,:,:], [1,2,0])
		baseline = np.transpose(pred_baseline[image_index,:,:,:], [1,2,0])
		combined = np.transpose(pred_combined[image_index,:,:,:], [1,2,0])
		
		# convert to right values for conversion to RGB
		grey = grey * 100
		ground_truth = (ground_truth*256)-128
		baseline = (baseline*256)-128
		combined = (combined*256)-128

		# combine lightness and color channels (in lab color space)
		ground_truth = np.concatenate((grey, ground_truth), axis=2)
		baseline = np.concatenate((grey, baseline), axis=2)
		combined = np.concatenate((grey, combined), axis=2)
		
		# convert to RGB
		ground_truth = color.lab2rgb(ground_truth)
		baseline = color.lab2rgb(baseline)
		combined = color.lab2rgb(combined)
		grey =np.hstack(np.tile(grey, [1,1,1,3]))/100
		
		scipy.misc.toimage(grey, cmin=0, cmax=1).save(os.path.join(save_dir, "greyscale_"+str(image_index)+".jpg"))
		scipy.misc.toimage(ground_truth, cmin=0, cmax=1).save(os.path.join(save_dir, "ground_truth_"+str(image_index)+".jpg"))
		scipy.misc.toimage(baseline, cmin=0, cmax=1).save(os.path.join(save_dir, "baseline_"+str(image_index)+".jpg"))
		scipy.misc.toimage(combined, cmin=0, cmax=1).save(os.path.join(save_dir, "combined_"+str(image_index)+".jpg"))