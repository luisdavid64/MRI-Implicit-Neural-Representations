import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, w0=30, is_first=False, is_last=False, last_tanh=False):
        super().__init__()
        self.in_f = in_f
        self.w0 = w0
        self.linear = nn.Linear(in_f, out_f)
        self.bias_layer = nn.Linear(1, out_f)  # Add a bias layer
        self.is_first = is_first
        self.is_last = is_last
        self.init_weights()
        self.last_tanh = last_tanh

    def init_weights(self):
        b = 1 / self.in_f if self.is_first else np.sqrt(6 / self.in_f) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)
            self.bias_layer.weight.data.zero_()
            self.bias_layer.bias.data.zero_()

    def modulation(self, coords):
        # Define modulation function here based on the distance from the center
        dist_to_center = (coords[...,1]**2 + coords[...,2]**2).unsqueeze(dim=-1)
        modulation_factor = torch.exp(-np.pi* dist_to_center)
        return modulation_factor

    def forward(self, x):
        modulation_factor = self.modulation(x)
        distance_bias = torch.relu(self.bias_layer(x)) + 1
        x = modulation_factor * torch.sin(self.w0 * distance_bias * x)

        # Use tanh to squeeze output to -1,1
        if self.last_tanh:
            return torch.tanh(x)
        return x if self.is_last else torch.sin(x)

class SIREN(nn.Module):
    def __init__(self, params):
        super(SIREN, self).__init__()

        num_layers = params['network_depth']
        hidden_dim = params['network_width']
        input_dim = params['network_input_size']
        output_dim = params['network_output_size']
        last_linear = True
        if "network_last_linear" in params:
            last_linear = params["network_last_linear"]
        last_tanh = False
        if "last_tanh" in params:
            last_tanh = params["last_tanh"]

        layers = [SirenLayer(input_dim, hidden_dim, is_first=True)]
        for i in range(1, num_layers - 1):
            layers.append(SirenLayer(hidden_dim, hidden_dim))
        layers.append(SirenLayer(hidden_dim, output_dim, is_last=last_linear, last_tanh=last_tanh))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)

        return out