import torch
import torch.nn as nn
from torch.nn import init
from networks import *


# to check if network parts can get build etc.

# load patchDiscriminator
norm_layer = networks.get_norm_layer(norm_type='batch')
D = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
