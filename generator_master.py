import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

# go with the classical U-Net architecture as described by Ronneberg et al

def conv_layer(in_channels, out_channels, kernel_size = 4, stride=2, padding=1, batch_norm=True, init_zero_weights=False):
	"""Creates a convolutional layer, with optional batch normalization.
	"""
	layers = []
	conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
	# conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
	if init_zero_weights:
		conv_layer.weight.data = torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.001
	layers.append(conv_layer)

	if batch_norm:
		layers.append(nn.BatchNorm2d(out_channels))
	return nn.Sequential(*layers)


def conv_encoder(in_channels, out_channels):
	return nn.Sequential(
		nn.Conv2d(in_channels, out_channels, 3, padding=1),
		nn.ReLU(inplace=True),
		nn.Conv2d(out_channels, out_channels, 3, padding=1),
		nn.ReLU(inplace=True)
	) 

def conv(in_channels, out_channels, kernel_size=3, batch_norm = True):
	layers = []
	# convolution with kernel size 4, image size is conserved (stride = 1), bias
	layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,kernel_size=kernel_size, stride=1, padding=1, bias=True))
	if batch_norm:
		layers.append(nn.BatchNorm2d(out_channels))
	layers.append(nn.LeakyReLU(0.2, True))
	return nn.Sequential(*layers)


def conv_downsampling(in_channels, out_channels, kernel_size=3, batch_norm=True):
	layers = []
	# convolution with kernel size 4, image size is conserved (stride = 1), bias
	layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,kernel_size=kernel_size, stride=2, padding=1, bias=True))
	if batch_norm:
		layers.append(nn.BatchNorm2d(out_channels))
	layers.append(nn.LeakyReLU(0.2, True))
	return nn.Sequential(*layers)


# upsamling layer (upsamling and reduction of number of channels by factor 2)
def conv_upsampling(in_channels, out_channels=in_channels/2, kernel_size=4, batch_norm=True):
	layers = []
	layers.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=1, bias=True))
	if batch_norm:
		layers.append(nn.BatchNorm2d(out_channels))
	layers.append(nn.LeakyReLU(0.2, True))
	return nn.Sequential(*layers)

def output_conv(in_channels, kernel_size=1, batch_norm=False):
	layers = []
	layers.append(nn.Conv2d(in_channels=in_channels, out_channels=2, kernel_size=kernel_size, stride=1, bias=False))
    if batch_norm:
		layers.append(nn.BatchNorm2d(out_channels))
	layers.append(nn.Tanh())
	return nn.Sequential(*layers)

def fully_connected(max_channels=256, num_classes=10):
    fully_connected_layers = nn.Sequential(
        nn.Linear(4*4*max_channels/2,max_channels*4),
        # nn.BatchNorm1d(max_channels*4),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(max_channels*4,max_channels*4),
        nn.ReLU(True),
        nn.Linear(max_channels*4, num_classes),
        nn.Softmax()
        )
    return fully_connected_layers

# my code
class unet(nn.Module):
    def __init__(self, max_channels = 256, batch_norm=True, classification=True, num_classes=10):
        super(unet,self).__init__()

        self.max_channels=max_channels
        
        #1st block (encoder 1)
        # size of output: o = (i-k) +2p +1 -> 32 - 3 +2 + 1 = 32 need kernel size of 3 to keep image size konstant
        self.conv1 = conv(1,max_channels/8,batch_norm=False) #TODO: why no batch norm for first convolution?
        self.down1 = conv_downsampling(max_channels/8,max_channels/8)

        # 2nd block (encoder 2)
        self.conv2 = conv(max_channels/8,max_channels/4,batch_norm=batch_norm)
        self.down2 = conv_downsampling(max_channels/4,max_channels/4)

        # 3rd block (encoder 3)
        self.conv3 = conv(max_channels/4,max_channels/2,batch_norm=batch_norm)
        self.down3 = conv_downsampling(max_channels/2,max_channels/2)

        # 4rth block (bottleneck) -> attach fully connected layers here
        self.conv4_1 = conv(max_channels/2, max_channels, batch_norm=batch_norm)
        self.conv4_2 = conv(max_channels, max_channels, batch_norm=batch_norm)

        # 5th block
        self.up5 = conv_upsampling(max_channels,max_channels/2,batch_norm=batch_norm)
        self.conv5 = conv(max_channels, max_channels/2, batch_norm=batch_norm)

        # 6th block
        self.up6 = conv_upsampling(max_channels/2, max_channels/4, batch_norm=batch_norm)
        self.conv6 = conv(max_channels/2, max_channels/4, batch_norm=batch_norm)

        # 7th block (output block!!)
        self.up7 = conv_upsampling(max_channels/4, max_channels/8, batch_norm=batch_norm)
        self.conv7 = conv(max_channels/4, max_channels/8, batch_norm=batch_norm)
        self.conv_out = output_conv(max_channels/8, batch_norm=False)

        # TODO: this will not work this way -> have to flatten the features (compare AlexNet or sth similar, how many features...do we need to further downsample first?????)
        if classification:
            self.conv_class = conv(max_channels,max_channels/2, batch_norm=batch_norm)
            self.classification = fully_connected(max_channels=max_channels, num_classes=num_classes)




    def forward(self, x, classification = True):
        horizontal_1 = self.conv1(x)
        out = self.down1(horizontal_1)

        horizontal_2 = self.conv2(out)
        out = self.down2(horizontal_2)

        horizontal_3 = self.conv3(out)
        out = self.down3(horizontal_3)

        features_for_classification = self.conv4_1(out)
        out = self.conv4_2(features_for_classification)

        # apply upsampling layer
        out = self.up5(out)
        # concatenate here and pass the concatenated tensor to next convolution layer
        out = torch.cat([out, horizontal_3], dim=1)
        out = self.conv5(out)

        out = self.up6(out)
        out = torch.cat([out, horizontal_2], dim=1)
        out = self.conv6(out)

        out = self.up7(out)
        out = torch.cat([out, horizontal_1], dim=1)
        out = self.conv7(out)
        col_pred = self.conv_out(out)

        if classification:
            out = self.conv_class(features_for_classification)
            out = out.view(out.size(0), self.max_channels/2 * 4 * 4)
            print('flattened: ', out.size())
            class_pred = self.classification(out)

        if classification:
            return [col_pred, class_pred]
        else:
            return col_pred



	

		
