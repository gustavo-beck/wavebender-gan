import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

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
    # GroupNorm (1, output_channel) -> (equivalent with LayerNorm)
    return nn.Sequential(
        nn.Conv1d(in_channels = in_channels,
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
    # GroupNorm (1, output_channel) -> (equivalent with LayerNorm)
    return nn.Sequential(
        nn.Conv1d(in_channels = in_channels,
                  out_channels = out_channels,
                  kernel_size = kernel_size,
                  bias = bias,
                  dilation = dilation,
                  padding = pad),
        nn.GroupNorm(1, out_channels)
        )

def conv_layer_last(in_channels, out_channels, kernel_size=3, bias=False, dilation=1, pad=1):
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
    # GroupNorm (1, output_channel) -> (equivalent with LayerNorm)
    return nn.Sequential(
        nn.Conv1d(in_channels = in_channels,
                  out_channels = out_channels,
                  kernel_size = kernel_size,
                  bias = bias,
                  dilation = dilation,
                  padding = pad)
        )

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
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

class WaveBenderNet(nn.Module):
    def __init__(self, block, params):
        super(WaveBenderNet, self).__init__()
        self.in_channels = params.n_channels_in
        self.out_channels = params.n_channels_out
        self.conv = conv_layer(in_channels = self.in_channels, out_channels = 64)
        self.layer1 = self.make_layer(block, 64, 256)
        self.layer2 = self.make_layer(block, 256, 256)
        self.layer3 = self.make_layer(block, 256, 512)
        self.layer4 = self.make_layer(block, 512, 256)
        self.layer5 = self.make_layer(block, 256, 256)
        self.layer6 = self.make_layer(block, 256, 64)
        self.conv1x1 = conv_layer_res(in_channels = 64, out_channels = self.out_channels, kernel_size= 1, bias = False, dilation=1, pad=0)

    def make_layer(self, block, out_channels, new_lenght):
        return nn.Sequential(block(out_channels, out_channels),
                             block(out_channels, out_channels),
                             block(out_channels, out_channels),
                             conv_layer(in_channels = out_channels, out_channels = new_lenght)
                             )

    def forward(self, x):
        out1 = self.conv(x)
        out = self.layer1(out1)
        out2 = self.layer2(out)
        out = self.layer3(out2)
        out = self.layer4(out)
        out = self.layer5(out)
        out += out2
        out = self.layer6(out)
        out += out1
        out = self.conv1x1(out)
        return out