import torch
import torch.nn as nn

from torch.nn import Module, init

from utils.ops import truncated_normal


def down_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, num_convs=2,
               initializers=None, nonlinearity=nn.ReLU,
               down_sample_input=True, down_sampling_op=nn.AvgPool2d(kernel_size=2, stride=2)):
    features = []

    if down_sample_input:
        features.append(down_sampling_op)

    for i in range(num_convs):
        in_ch = in_channels if i == 0 else out_channels
        features.append(Conv2d(in_ch, out_channels, kernel_size, stride, padding, initializers=initializers))
        features.append(nonlinearity(inplace=True))

    return nn.Sequential(*features)


class Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initializers=None, **kwargs):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, padding, **kwargs)

        if initializers is not None:
            if 'w' in initializers:
                initializers['w'](self.weight)
            if 'b' in initializers:
                initializers['b'](self.bias)


class Encoder(Module):

    def __init__(self, num_channels, nonlinearity=nn.ReLU, num_convs=3,
                 initializers={'w': init.kaiming_normal_, 'b': lambda b: truncated_normal(b, std=1e-3)},
                 down_sampling_op=nn.AvgPool2d(kernel_size=2, stride=2)):
        Module.__init__(self)
        self._num_channels = num_channels

        features = []
        for i, (in_channels, out_channels) in enumerate(zip(num_channels[:-1], num_channels[1:])):
            if i == 0:
                down_sample = False
            else:
                down_sample = True

            features.append(down_block(in_channels, out_channels,
                                       nonlinearity=nonlinearity,
                                       initializers=initializers, down_sample_input=down_sample))

        self.features = features

    def forward(self, inputs):
        features = [inputs]
        for feature in self.features:
            features.append(feature(features[-1]))

        return features[1:]

