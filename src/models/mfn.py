import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# class SirenLayer(nn.Module):
#     def __init__(self, in_f, out_f, w0=30, is_first=False, is_last=False, last_tanh=False):
#         super().__init__()
#         self.in_f = in_f
#         self.w0 = w0
#         self.linear = nn.Linear(in_f, out_f)
#         self.bias_layer = nn.Linear(1, out_f)  # Add a bias layer
#         self.is_first = is_first
#         self.is_last = is_last
#         self.init_weights()
#         self.last_tanh = last_tanh

#     def init_weights(self):
#         b = 1 / self.in_f if self.is_first else np.sqrt(6 / self.in_f) / self.w0
#         with torch.no_grad():
#             self.linear.weight.uniform_(-b, b)
#             self.bias_layer.weight.data.zero_()
#             self.bias_layer.bias.data.zero_()

#     def modulation(self, coords):
#         # Define modulation function here based on the distance from the center
#         dist_to_center = (coords[...,1]**2 + coords[...,2]**2)
#         modulation_factor = torch.exp(-np.pi* dist_to_center)
#         return modulation_factor

#     def forward(self, x):
#         if self.is_last:
#             x = self.linear(x)
#         elif self.last_tanh:
#             x = self.linear(x)
#             return torch.tanh(x)
#         else:
#             modulation_factor = self.modulation(x)
#             x = self.linear(x)
#             x = torch.sin(self.w0 * x)
#             x = modulation_factor * torch.sin(self.w0 * x)
#         return x

# class SIREN(nn.Module):
#     def __init__(self, params):
#         super(SIREN, self).__init__()

#         num_layers = params['network_depth']
#         hidden_dim = params['network_width']
#         input_dim = params['network_input_size']
#         output_dim = params['network_output_size']
#         last_linear = True
#         if "network_last_linear" in params:
#             last_linear = params["network_last_linear"]
#         last_tanh = False
#         if "last_tanh" in params:
#             last_tanh = params["last_tanh"]

#         layers = [SirenLayer(input_dim, hidden_dim, is_first=True)]
#         for i in range(1, num_layers - 1):
#             layers.append(SirenLayer(hidden_dim, hidden_dim))
#         layers.append(SirenLayer(hidden_dim, output_dim, is_last=last_linear, last_tanh=last_tanh))

#         self.model = nn.Sequential(*layers)

#     def forward(self, x):
#         out = self.model(x)

#         return out


class MFNBase(nn.Module):
    """
    Multiplicative filter network base class.

    Expects the child class to define the 'filters' attribute, which should be 
    a nn.ModuleList of n_layers+1 filters with output equal to hidden_size.
    """

    def __init__(
        self, hidden_size, out_size, n_layers, weight_scale, bias=True, output_act=False
    ):
        super().__init__()

        self.linear = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size, bias) for _ in range(n_layers)]
        )
        self.output_linear = nn.Linear(hidden_size, out_size)
        self.output_act = output_act

        for lin in self.linear:
            lin.weight.data.uniform_(
                -np.sqrt(weight_scale / hidden_size),
                np.sqrt(weight_scale / hidden_size),
            )

        return

    def forward(self, x):
        out = self.filters[0](x)
        for i in range(1, len(self.filters)):
            out = self.filters[i](x) * self.linear[i - 1](out)
        out = self.output_linear(out)

        if self.output_act:
            out = torch.sin(out)

        return out

class FourierLayer(nn.Module):
    """
    Sine filter as used in FourierNet.
    """

    def __init__(self, in_features, out_features, weight_scale):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.linear.weight.data *= weight_scale  # gamma
        self.linear.bias.data.uniform_(-np.pi, np.pi)
        return

    def forward(self, x):
        return torch.sin(self.linear(x))


class FourierNet(MFNBase):
    def __init__(
        self,
        params,
        out_size=1.0,
        input_scale=2.0,
        weight_scale=1.0,
        bias=True,
        output_act=False,
    ):
        n_layers = params['network_depth']
        hidden_size = params['network_width']
        in_size = params['network_input_size']
        out_size = params['network_output_size'] 
        super().__init__(
            hidden_size, out_size, n_layers, weight_scale, bias, output_act
        )
        self.filters = nn.ModuleList(
            [
                FourierLayer(in_size, hidden_size, input_scale / np.sqrt(n_layers + 1))
                for _ in range(n_layers + 1)
            ]
        )

class GaborLayer(nn.Module):
    """
    Gabor-like filter as used in GaborNet.
    """

    def __init__(self, in_features, out_features, weight_scale, alpha=1.0, beta=1.0, with_dist_filtering=False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        if self.with_dist_filtering:
            self.mu = nn.Parameter(2 * torch.rand(out_features, 2) - 1)
        else:
            self.mu = nn.Parameter(2 * torch.rand(out_features, in_features) - 1)
        self.gamma = nn.Parameter(
            torch.distributions.gamma.Gamma(alpha, beta).sample((out_features,))
        )
        self.linear.weight.data *= weight_scale * torch.sqrt(self.gamma[:, None])
        self.linear.bias.data.uniform_(-np.pi, np.pi)
        return

    def forward(self, x, dist_to_center=None):
        if self.with_dist_filtering:
            # Normalize by div by 2
            D = (
                (dist_to_center ** 2).sum(-1)[..., None]
                + (self.mu ** 2).sum(-1)[None, :]
                - 2 * dist_to_center @ self.mu.T
            )

        else:
            D = (
                (x ** 2).sum(-1)[..., None]
                + (self.mu ** 2).sum(-1)[None, :]
                - 2 * x @ self.mu.T
            )
        return torch.sin(self.linear(x)) * torch.exp(-0.5 * D * self.gamma[None, :])

class GaborNet(MFNBase):
    def __init__(
        self,
        params,
        input_scale=2,
        weight_scale=1.0,
        alpha=6.0,
        beta=1.0,
        bias=True,
        output_act=False,
    ):
        n_layers = params['network_depth']
        hidden_size = params['network_width']
        in_size = params['network_input_size']
        out_size = params['network_output_size'] 
        super().__init__(
            hidden_size, out_size, n_layers, weight_scale, bias, output_act
        )
        self.filters = nn.ModuleList(
            [
                GaborLayer(
                    in_size,
                    hidden_size,
                    input_scale / np.sqrt(n_layers + 1),
                    alpha / (n_layers + 1),
                    beta,
                )
                for _ in range(n_layers + 1)
            ]
        )

class KGaborNet(MFNBase):
    def __init__(
        self,
        params,
        input_scale=2,
        weight_scale=1.0,
        alpha=6.0,
        beta=1.0,
        bias=True,
        output_act=False,
    ):
        n_layers = params['network_depth']
        hidden_size = params['network_width']
        in_size = params['network_input_size']
        out_size = params['network_output_size'] 
        super().__init__(
            hidden_size, out_size, n_layers, weight_scale, bias, output_act
        )
        self.filters = nn.ModuleList(
            [
                GaborLayer(
                    in_size,
                    hidden_size,
                    input_scale / np.sqrt(n_layers + 1),
                    alpha / (n_layers + 1),
                    beta,
                )
                for _ in range(n_layers + 1)
            ]
        )

    def forward(self, x, dist_to_center):
        out = self.filters[0](x)
        for i in range(1, len(self.filters)):
            out = self.filters[i](x, dist_to_center) * self.linear[i - 1](out)
        out = self.output_linear(out)

        if self.output_act:
            out = torch.sin(out)

        return out