import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

def conv_layer(in_channels, out_channels, kernel_size=3, bias=False, dilation=1, pad=1):
    """
    Attributes:
    in_channels: Number of channels in the input space
    out_channels: Number of channels produced by the convolution
    kernel_size: Size of the convolving kernel
    stride: Amount of movement between applications of the filter to the input
    dilation: Spacing between kernel elements.
    groups: Number of blocked connections from input channels to output channels.
    bias: Adds a learnable bias to the output.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = kernel_size,
                bias = bias,
                dilation = dilation,
                padding = pad),
        nn.GroupNorm(1, out_channels),
        nn.PReLU(num_parameters=out_channels),
        nn.Dropout(0.1)
        )

def conv_layer_res(in_channels, out_channels, kernel_size=3, bias=False, dilation=1, pad=1):
    """
    Attributes:
    in_channels: Number of channels in the input space
    out_channels: Number of channels produced by the convolution
    kernel_size: Size of the convolving kernel
    stride: Amount of movement between applications of the filter to the input
    dilation: Spacing between kernel elements.
    groups: Number of blocked connections from input channels to output channels.
    bias: Adds a learnable bias to the output.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = kernel_size,
                bias = bias,
                dilation = dilation,
                padding = pad)
        )

# Residual block
class ResidualBlockGan(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlockGan, self).__init__()
        self.conv1 = conv_layer(in_channels, out_channels)
        self.conv2 = conv_layer(out_channels, out_channels)
        self.conv3 = conv_layer_res(out_channels, out_channels)
        self.prelu_res = nn.PReLU(num_parameters=out_channels)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += residual
        out = self.prelu_res(out)
        return out

class WaveDiscriminator(nn.Module):
    def __init__(self, block, params):
        super(WaveDiscriminator, self).__init__()
        self.in_channels = 1 # Similar to a grayscale
        self.out_channels = params.n_channels_out
        self.conv = conv_layer(in_channels = self.in_channels, out_channels = 32)
        self.layer1 = self.make_layer(block, 32, 64)
        self.conv1x1 = conv_layer_res(in_channels = 64, out_channels = self.in_channels, kernel_size= 1, bias = False, dilation=1, pad=0)
        self.fc1 = nn.Linear(2**14, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def make_layer(self, block, out_channels, new_lenght):
        return nn.Sequential(block(out_channels, out_channels),
                            block(out_channels, out_channels),
                            block(out_channels, out_channels),
                            conv_layer(in_channels = out_channels, out_channels = new_lenght)
                            )

    def forward(self, x):
        out = self.conv(x)
        out = self.layer1(out)
        out = self.conv1x1(out)
        out = nn.AdaptiveAvgPool2d(128)(out)
        out = out.view(out.size(0), -1)
        out = self.relu(self.fc1(out))
        out = self.fc2(out) # LS Gan doesn't require sigmoid
        return out