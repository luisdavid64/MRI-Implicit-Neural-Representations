import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

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
        self.linear2 = nn.Linear(512, out_features)
        # self.linear2.weight.data *= weight_scale  # gamma
        # self.linear2.bias.data.uniform_(-np.pi, np.pi)
        return

    def forward(self, x, dist_to_center):
        if dist_to_center is not None:
            filter = torch.exp(-(self.linear2(dist_to_center))**2)
            return torch.sin(self.linear(x))*filter
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

    def forward(self, x, dist_to_center=None):
        out = self.filters[0](x, dist_to_center)
        for i in range(1, len(self.filters)):
            out = self.filters[i](x, dist_to_center) * self.linear[i - 1](out)
        out = self.output_linear(out)

        if self.output_act:
            out = torch.sin(out)

        return out

class GaborLayer(nn.Module):
    """
    Gabor-like filter as used in GaborNet.
    """

    def __init__(self, in_features, out_features, weight_scale, alpha=1.0, beta=1.0, with_dist_filtering=False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.with_dist_filtering = with_dist_filtering
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

class MultiscaleKFourier(MFNBase):
    def __init__(self,
                 params,
                 weight_scale=1.0,
                 bias=True,
                 output_act=False,
                 centered=True,
                #  output_layers=None,
                 output_layers=[1,3,5,7],
                 reuse_filters=False):

        hidden_layers = params['network_depth']
        hidden_size = params['network_width']
        in_size = params['network_input_size']
        out_size = params['network_output_size'] 
        super().__init__(hidden_size, out_size, hidden_layers,
                         weight_scale, bias, output_act)

        self.hidden_layers = hidden_layers
        self.centered = centered
        self.output_layers = output_layers
        self.reuse_filters = reuse_filters
        self.stop_after = None

        # we need to multiply by this to be able to fit the signal
        self.filters = nn.ModuleList(
            [
                FourierLayer(in_size, hidden_size, weight_scale / np.sqrt(hidden_layers + 1))
                for _ in range(hidden_layers + 1)
            ]
        )
        # linear layers to extract intermediate outputs
        self.output_linear = nn.ModuleList([nn.Linear(hidden_size, out_size) for i in range(len(self.filters))])

        # if outputs layers is None, output at every possible layer
        if self.output_layers is None:
            self.output_layers = np.arange(1, len(self.filters))

        print(self)

    def forward(self, coords):

        outputs = []
        out = self.filters[0](coords)
        for i in range(1, len(self.filters)):
            out = self.filters[i](coords) * self.linear[i - 1](out)

            if i in self.output_layers:
                outputs.append(self.output_linear[i](out))
                if self.stop_after is not None and len(outputs) > self.stop_after:
                    break

        return outputs 