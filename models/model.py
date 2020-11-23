import torch
import torch.nn as nn
import numpy as np


class ConvBlock(nn.Module):
    """
    Basic convolutionnal block to use for our model.
    Its architecture is : Convolution -> BatchNorm -> Dropout -> Relu -> MaxPool.

    Parameters
    ----------
    in_channels: int
        Number of channels for the input
    in_height: int
        Height of the input to feed in
    in_width: int
        Width of the input to feed in
    out_channels: int
        Number of channels for the output
    kernel_size: int
        Size of the kernel convolution
    stride: int
        Stride of the convolution
    probs_dropout: float in [0,1]
        Probability of dropout
    """
    def __init__(self, in_channels, in_height, in_width, out_channels, kernel_size, stride, probs_dropout):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.in_height = in_height
        self.in_width = in_width
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.probs_dropout = probs_dropout
        self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride),
                                   nn.BatchNorm2d(out_channels),
                                   nn.Dropout(probs_dropout),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size, stride))

        h_out_after_conv, w_out_after_conv = self.calculate_shape_after_conv(in_height, in_width, kernel_size, stride)
        h_out, w_out = self.calculate_shape_after_conv(h_out_after_conv, w_out_after_conv, kernel_size, stride)
        self.h_out = h_out
        self.w_out = w_out

    @staticmethod
    def calculate_shape_after_conv(h_in, w_in, kernel_size, stride):
        """Calculate shape of the output convolution given height and width of the input"""
        h_out = np.floor((h_in - kernel_size) / stride + 1)
        w_out = np.floor((w_in - kernel_size) / stride + 1)
        return int(h_out), int(w_out)

    def forward(self, x):
        """Expect input of size batch_size x in_channels x * x *"""
        out = self.block(x)
        return out


class LinearBlock(nn.Module):
    """
    Basic block for the classification part of the model : linear -> dropout -> relu

    Parameters
    ----------
    in_size: int
        Input size
    out_size: int
        Output size
    probs_dropout: float in [0,1]
        Probabilty of dropout
    """
    def __init__(self, in_size, out_size, probs_dropout):
        super(LinearBlock, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.probs_dropout = probs_dropout
        self.block = nn.Sequential(nn.Linear(in_size, out_size),
                                   nn.Dropout(probs_dropout),
                                   nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class MyCNN(nn.Module):
    """
    CNN model to do binary classification for 2D RGB images

    Parameters
    ----------
    im_size: int
        Height and width of the images
    config_block: dict
        Dictionary with parameters for convolutional blocks
    n_out_channels: list
        List with all number of output channels, for each block
    n_hidden_neurons: list
        List with number of neurons for each layer of the classification part
    """
    def __init__(self,
                 im_size,
                 config_block,
                 n_out_channels,
                 n_hidden_neurons):
        super(MyCNN, self).__init__()
        self.im_size = im_size
        self.config_block = config_block
        self.conv_blocks = nn.ModuleList([])
        self.n_out_channels = n_out_channels
        h_in, w_in = im_size, im_size
        for i in range(len(self.n_out_channels) - 1):
            self.conv_blocks.append(ConvBlock(in_channels=self.n_out_channels[i],
                                              in_height=h_in,
                                              in_width=w_in,
                                              out_channels=self.n_out_channels[i + 1],
                                              **config_block))
            h_in, w_in = self.conv_blocks[i].h_out, self.conv_blocks[i].w_out

        outsize = self.n_out_channels[-1] * self.conv_blocks[-1].w_out * self.conv_blocks[-1].h_out
        n_hidden_neurons.insert(0, outsize)
        self.n_hidden_neurons = n_hidden_neurons

        self.linear_blocks = nn.ModuleList([])
        for i in range(len(self.n_hidden_neurons) - 1):
            self.linear_blocks.append(LinearBlock(self.n_hidden_neurons[i],
                                                  self.n_hidden_neurons[i + 1],
                                                  0.1)
                                      )
        self.classifier = nn.Sequential(*self.linear_blocks,
                                        nn.Linear(self.n_hidden_neurons[-1], 1),
                                        nn.Sigmoid())

    def forward(self, x):
        """Forward pass for x of shape batch_size x 3 x *"""
        for i in range(len(self.n_out_channels) - 1):
            x = self.conv_blocks[i](x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x
