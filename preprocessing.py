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

from skimage import io, color
from load_data import load_cifar10
import random

# CATEGORIES
airplane = 0
automobile = 1
bird =2
cat = 3
deer = 4
dog = 5
frog = 6
horse = 7
ship = 8
truck = 9

######################################################################
# Data related code
######################################################################
def process_lab(xs,ys,max_pixel=256.0, categories=[horse]):
    xs = xs / max_pixel # normalize to 1
    # xs = xs[np.where(ys == horse)[0], :, :, :]
    xs_red = xs[np.where(ys == categories[0])[0], :, :, :]
    print(xs_red.shape)
    if len(categories)>1:
      for cat in range(1,len(categories)):
        xs_cat = xs[np.where(ys == categories[cat])[0], :, :, :]
        # print(xs_cat.shape)
        xs_red = np.concatenate((xs_red,xs_cat), axis=0)
        print(xs_red.shape)
        # print(xs_cat.shape)
        # xs_red.expand(xs_cat)
        
    npr.shuffle(xs_red)
    
    # transfer RGB to lab color space
    xs_l_channel = np.zeros([xs_red.shape[0],1,xs_red.shape[2], xs_red.shape[3]])
    xs_ab_channel = np.zeros([xs_red.shape[0],2,xs_red.shape[2], xs_red.shape[3]])
    for image_iter in range(xs_red.shape[0]):
      image_to_convert = xs_red[image_iter,:,:,:]
      image_to_convert = np.transpose(image_to_convert, [1,2,0])
      image_lab = color.rgb2lab(image_to_convert)
      l_channel = np.transpose(np.expand_dims(image_lab[:,:,0]/100, axis=2),(2,0,1))
      xs_l_channel[image_iter,:,:,:] = l_channel
      ab_channel = np.transpose((image_lab[:,:,1:3]+128)/256, (2,0,1))
      xs_ab_channel[image_iter,:,:,:] = ab_channel

    return xs_l_channel, xs_ab_channel

def process_lab_class(xs,ys,max_pixel=256.0):
    xs = xs / max_pixel # normalize to 1
    
    # transfer RGB to lab color space
    xs_l_channel = np.zeros([xs.shape[0],1,xs.shape[2], xs.shape[3]])
    xs_ab_channel = np.zeros([xs.shape[0],2,xs.shape[2], xs.shape[3]])
    for image_iter in range(xs.shape[0]):
      image_to_convert = xs[image_iter,:,:,:]
      image_to_convert = np.transpose(image_to_convert, [1,2,0])
      image_lab = color.rgb2lab(image_to_convert)
      l_channel = np.transpose(np.expand_dims(image_lab[:,:,0]/100, axis=2),(2,0,1))
      xs_l_channel[image_iter,:,:,:] = l_channel
      ab_channel = np.transpose((image_lab[:,:,1:3]+128)/256, (2,0,1))
      xs_ab_channel[image_iter,:,:,:] = ab_channel

    # shuffling
    ys = ys[:,0]
    p = npr.permutation(len(ys))
    ys = ys[p]
    xs_l_channel = xs_l_channel[p]
    xs_ab_channel = xs_ab_channel[p]
    print(xs_l_channel.shape)
    print(xs_ab_channel.shape)
    print(ys.shape)
    return xs_l_channel, xs_ab_channel, ys

def process_classification(xs,ys,max_pixel=256.0):
    xs = xs / max_pixel # normalize to 1
    
    # transfer RGB to lab color space
    xs_lab = np.zeros(xs.shape)
    for image_iter in range(xs.shape[0]):
      image_to_convert = xs[image_iter,:,:,:]
      image_to_convert = np.transpose(image_to_convert, [1,2,0]) # transpose from Ch,R,C to R,C,Ch
      image_lab = color.rgb2lab(image_to_convert)
      image_lab = np.transpose(image_lab, (2,0,1))# transpose image back to Ch, R, C
      xs_lab[image_iter,:,:,:] = image_lab
    
    ys_one_hot = ys

    # shuffle
    xs_lab = xs_lab[:,1,:,:]
    print(xs_lab.shape)
    xs_lab = np.expand_dims(xs_lab, axis=1)
    print(xs_lab.shape)
    p = npr.permutation(len(ys))
    xs_lab = xs_lab[p]
    ys_one_hot = ys_one_hot[p]
    ys_one_hot = ys_one_hot[:,0]
    
    return xs_lab, ys_one_hot

def process(xs, ys, max_pixel=256.0, categories = [horse]):
    """
    Pre-process CIFAR10 images by taking only the horse category,
    shuffling, and have colour values be bound between 0 and 1

    Args:
      xs: the colour RGB pixel values
      ys: the category labels
      max_pixel: maximum pixel value in the original data
    Returns:
      xs: value normalized and shuffled colour images
      grey: greyscale images, also normalized so values are between 0 and 1
    """
    xs = xs / max_pixel
    # xs = xs[np.where(ys == horse)[0], :, :, :]
    xs = xs[np.where(ys == categories[0])[0], :, :, :]
    npr.shuffle(xs)
    print(xs.shape)
    grey = np.mean(xs, axis=1, keepdims=True)
    return (xs, grey)

def get_batch(x, y, batch_size):
    '''
    Generated that yields batches of data

    Args:
      x: input values
      y: output values
      batch_size: size of each batch
    Yields:
      batch_x: a batch of inputs of size at most batch_size
      batch_y: a batch of outputs of size at most batch_size
    '''
    N = np.shape(x)[0]
    assert N == np.shape(y)[0]
    for i in range(0, N, batch_size):
        batch_x = x[i:i+batch_size, :,:,:]
        batch_y = y[i:i+batch_size, :,:,:]
        yield (batch_x, batch_y)

def get_batch_classification(x, y, batch_size):
    '''
    Generated that yields batches of data

    Args:
      x: input values
      y: output values
      batch_size: size of each batch
    Yields:
      batch_x: a batch of inputs of size at most batch_size
      batch_y: a batch of outputs of size at most batch_size
    '''
    N = np.shape(x)[0]
    assert N == np.shape(y)[0]
    for i in range(0, N, batch_size):
        batch_x = x[i:i+batch_size, :,:,:]
        batch_y = y[i:i+batch_size]
        yield (batch_x, batch_y)
def get_batch_col_class(x, y_col, y_class, batch_size):
    '''
    Yields:
      batch_x: a batch of inputs of size at most batch_size
      batch_y_col: a batch of outputs of size at most batch_size
      batch_y_class: a batch of outputs of size at most batch_size
    '''
    N = np.shape(x)[0]
    assert N == np.shape(y_col)[0]
    assert N == np.shape(y_class)[0]
    for i in range(0, N, batch_size):
        batch_x = x[i:i+batch_size, :,:,:]
        batch_y_col = y_col[i:i+batch_size, :,:,:]
        batch_y_class = y_class[i:i+batch_size]
        yield (batch_x, batch_y_col, batch_y_class)

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

def plot_lab(input, gt, output, path, RGB=True):
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
    # print(grey.dtype)
    gt = np.transpose(gt[:10,:3,:,:], [0,2,3,1])
    # print(gt.dtype)
    predicted = np.transpose(output[:10,:3,:,:], [0,2,3,1])
    # print(predicted.dtype)
    
    # if RGB:
    #   grey = (grey*100)
    #   gt = ((gt*256)-128)
    #   predicted = ((predicted*256)-128)

    # if False:
    if RGB:
      grey = (grey*100)# .astype(int)
      gt = ((gt*256)-128)# /128# .astype(int)
      predicted = ((predicted*256)-128)# /128# .astype(np.float64)
      # print('grey', str(np.amax(grey)), str(np.amin(grey)))
      # print('pred', str(np.amax(predicted)), str(np.amin(predicted)))
      # print('gt', str(np.amax(gt)), str(np.amin(gt)))

    gt = np.concatenate((grey,gt), axis=3)
    gt = np.hstack(gt)
    predicted = np.concatenate((grey,predicted), axis=3)
    predicted = np.hstack(predicted)

    if RGB:
      predicted = color.lab2rgb(predicted)
      gt = color.lab2rgb(gt)
      grey =np.hstack(np.tile(grey, [1,1,1,3]))/100

      # print('grey', str(np.amax(grey)), str(np.amin(grey)))
      # print('pred', str(np.amax(predicted)), str(np.amin(predicted)))
      # print('gt', str(np.amax(gt)), str(np.amin(gt)))
    # if RGB:
    #   predicted = color.lab2rgb(predicted)
    #   gt = color.lab2rgb(gt)
     

    
    img = np.vstack([grey,gt,predicted])
    
    plt.figure(figsize=(30, 100))
    plt.imshow(img)
    plt.savefig(path,dpi = 300)
    plt.close()
    # plt.clf()
    # scipy.misc.toimage(img, cmin=0, cmax=1).save(path)
