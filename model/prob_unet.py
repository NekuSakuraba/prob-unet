import torch
import torch.nn as nn

from torch.nn import Module, init
from torch.distributions.multivariate_normal import MultivariateNormal

from utils.ops import *


initializers={'w': init.kaiming_normal_, 'b': lambda b: truncated_normal(b, std=1e-3)}
down_sampling_op = nn.AvgPool2d(kernel_size=2, stride=2)
# up_sampling_op = UpSampling()


def down_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, num_convs=2,
               initializers=None, nonlinearity=nn.ReLU,
               down_sample_input=True, down_sampling_op=down_sampling_op, device='cpu'):
    features = []

    if down_sample_input:
        features.append(down_sampling_op)

    for i in range(num_convs):
        in_ch = in_channels if i == 0 else out_channels
        features.append(Conv2d(in_ch, out_channels, kernel_size, stride, padding, initializers=initializers, device=device))
        features.append(nonlinearity(inplace=True))

    return nn.Sequential(*features)


class Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initializers=None, device='cpu', **kwargs):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, padding, **kwargs)

        if initializers is not None:
            if 'w' in initializers:
                initializers['w'](self.weight)
            if 'b' in initializers:
                initializers['b'](self.bias)
        
        self.to(device=device)


class UpSampling(Module):

    def __init__(self, scale_factor=2, mode='bilinear', align_corners=True):
        Module.__init__(self)

        self.params = dict()
        self.params['mode'] = mode
        self.params['scale_factor'] = scale_factor
        self.params['align_corners'] = align_corners

    def forward(self, input):
        return nn.functional.interpolate(input, **self.params)


class Encoder(Module):

    def __init__(self, num_channels, nonlinearity=nn.ReLU, num_convs=3,
                 initializers=initializers, down_sampling_op=down_sampling_op, device='cpu'):
        Module.__init__(self)
        self._num_channels = num_channels

        features = []
        for i, (in_channels, out_channels) in enumerate(zip(num_channels[:-1], num_channels[1:])):
            if i == 0:
                down_sample = False
            else:
                down_sample = True

            features.append(down_block(in_channels, out_channels,
                                       num_convs=num_convs,
                                       nonlinearity=nonlinearity,
                                       initializers=initializers,
                                       down_sample_input=down_sample,
                                       down_sampling_op=down_sampling_op,
                                       device=device))

        self.features = features

    def forward(self, inputs):
        features = [inputs]
        for feature in self.features:
            features.append(feature(features[-1]))

        return features[1:]


class Decoder(Module):

    def __init__(self, num_channels, num_classes, nonlinearity=nn.ReLU, num_convs=3,
                 initializers=initializers, up_sampling_op=down_sampling_op, device='cpu'):

        Module.__init__(self)

        self.up_sampling_op = up_sampling_op

        features = []
        for in_channels, out_channels in list(zip(num_channels[1:-1], num_channels[2:]))[::-1]:
            feature = []
            for i in range(num_convs):
                in_ch = in_channels + out_channels if i == 0 else in_channels
                feature.append(
                    Conv2d(in_ch, in_channels, kernel_size=3, stride=1, padding=1, initializers=initializers, device=device))
                feature.append(nonlinearity(inplace=True))
            features.append(nn.Sequential(*feature))

        self.features = features

    def forward(self, inputs):
        lower_res_features = inputs[-1]

        for idx, same_res_inputs in enumerate(inputs[:-1][::-1]):
            features = self.up_sampling_op(lower_res_features)
            features = torch.cat([features, same_res_inputs], dim=1)
            lower_res_features = self.features[idx](features)

        return lower_res_features


class UNet(Module):

    def __init__(self, num_channels, num_classes, nonlinearity=nn.ReLU, num_convs=3, initializers=initializers,
                 down_sampling_op=down_sampling_op, up_sampling_op=UpSampling(), device='cpu'):
        Module.__init__(self)

        self.encoder = Encoder(num_channels, nonlinearity, num_convs, initializers, down_sampling_op, device)
        self.decoder = Decoder(num_channels, num_classes, nonlinearity, num_convs, initializers, up_sampling_op, device)

    def forward(self, inputs):
        encoder_features = self.encoder(inputs)
        predicted_logits = self.decoder(encoder_features)
        return predicted_logits


class Prior(Module):

    def __init__(self, latent_dim, num_channels, nonlinearity=nn.ReLU, num_convs=3, initializers=initializers,
                 down_sampling_op=down_sampling_op, device='cpu'):
        Module.__init__(self)

        self.latent_dim = latent_dim

        self._encoder = Encoder(num_channels, nonlinearity, num_convs, initializers, down_sampling_op, device)
        self._conv2d = Conv2d(num_channels[-1], 2 * latent_dim, kernel_size=1, stride=1, padding=0,
                              initializers=initializers, device=device)

    def forward(self, input):
        encoding = self._encoder(input)[-1]
        encoding = mean(encoding, dim=[2, 3], keepdim=True)

        mu_log_sigma = torch.squeeze(self._conv2d(encoding))
        mu = mu_log_sigma[:, :self.latent_dim]
        sigma = torch.exp(mu_log_sigma[:, self.latent_dim:])
        covariance_matrix = create_covariance_matrix(sigma)

        multivariate_normal = MultivariateNormal(mu, covariance_matrix)
        return multivariate_normal


class Posterior(Prior):

    def __init__(self, latent_dim, num_channels, nonlinearity=nn.ReLU, num_convs=3, initializers=initializers,
                 down_sampling_op=down_sampling_op):
        Prior.__init__(self, latent_dim, num_channels, nonlinearity, num_convs, initializers, down_sampling_op)

    def forward(self, input, segment):
        input = torch.cat([input, segment], dim=1)
        return super().forward(input)


class Conv1x1Decoder(Module):

    def __init__(self, in_channels, out_channels, num_classes, num_convs, nonlinearity=nn.ReLU,
                 initializers=initializers, device='cpu'):
        Module.__init__(self)

        features = []
        features.append(
            Conv2d(in_channels, out_channels, kernel_size=1, stride=1, initializers=initializers, device=device))
        features.append(nonlinearity(inplace=True))

        for _ in range(num_convs - 1):
            features.append(
                Conv2d(out_channels, out_channels, kernel_size=1, stride=1, initializers=initializers, device=device))
            features.append(nonlinearity(inplace=True))

        features.append(Conv2d(out_channels, num_classes, kernel_size=1, initializers=initializers, device=device))
        self.net = nn.Sequential(*features)

    def forward(self, features, z):
        pass

