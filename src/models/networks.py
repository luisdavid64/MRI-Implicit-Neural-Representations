import numpy as np

import torch
import torch.nn as nn

############ Input Positional Encoding ############
class Positional_Encoder():
    def __init__(self, params, device):
        self.device = device
        self.B = None
        self.embedding_type = params["embedding"]
        if params['embedding'] == 'gauss':
            self.B = torch.randn((params['embedding_size'], params['coordinates_size'])) * params['scale']
            self.B = self.B.to(device)
        elif params['embedding'] == "LogF":
            self.B = 2.**torch.linspace(0., params["scale"], steps=int(params["embedding_size"]/(2*params["coordinates_size"]))).reshape(-1,1)
            self.B = self.B.to(device)
        elif params["embedding"] == 'none':
            pass
        else:
            raise NotImplementedError

    def embedding(self, x):
        if self.embedding_type == "LogF":
            emb1 = torch.cat((torch.sin((2.*np.pi*x[:,:1]) @ self.B.T),torch.cos((2.*np.pi*x[:,:1]) @ self.B.T)),dim=-1)
            emb2 = torch.cat((torch.sin((2.*np.pi*x[:,1:2]) @ self.B.T),torch.cos((2.*np.pi*x[:,1:2]) @ self.B.T)),1)
            emb3 = torch.cat((torch.sin((2.*np.pi*x[:,2:3]) @ self.B.T),torch.cos((2.*np.pi*x[:,2:3]) @ self.B.T)),1)
            x_embedding = torch.cat([emb1,emb2,emb3],dim=-1)
            return x_embedding
        if self.B is not None:
            x_embedding = (2. * np.pi * x) @ self.B.t()
            x_embedding = torch.cat([torch.sin(x_embedding), torch.cos(x_embedding)], dim=-1)
            return x_embedding
        else:
            return x



############ Fourier Feature Network ############
class Swish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

class FFN(nn.Module):
    def __init__(self, params):
        super(FFN, self).__init__()

        num_layers = params['network_depth']
        hidden_dim = params['network_width']
        input_dim = params['network_input_size']
        output_dim = params['network_output_size']

        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for i in range(1, num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out



############ SIREN Network ############
class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, w0=30, is_first=False, is_last=False, last_tanh=False):
        super().__init__()
        self.in_f = in_f
        self.w0 = w0
        self.linear = nn.Linear(in_f, out_f)
        self.is_first = is_first
        self.is_last = is_last
        self.init_weights()
        self.last_tanh = last_tanh

    def init_weights(self):
        b = 1 / \
            self.in_f if self.is_first else np.sqrt(6 / self.in_f) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)

    def forward(self, x):
        x = self.linear(x)
        # Use tanh to squeeze output to -1,1
        if self.last_tanh:
            return torch.tanh(x)
        return x if self.is_last else torch.sin(self.w0 * x)


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


class RealGaborLayer(nn.Module):
    '''
        Implicit representation with Gabor nonlinearity
        
        Inputs;
            in_features: Input features
            out_features; Output features
            bias: if True, enable bias for the linear operation
            is_first: Legacy SIREN parameter
            omega_0: Legacy SIREN parameter
            omega: Frequency of Gabor sinusoid term
            scale: Scaling of Gabor Gaussian term
    '''
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega0=10.0, sigma0=10.0,
                 trainable=False):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first
        
        self.in_features = in_features
        
        self.freqs = nn.Linear(in_features, out_features, bias=bias)
        self.scale = nn.Linear(in_features, out_features, bias=bias)
        
    def forward(self, input):
        omega = self.omega_0 * self.freqs(input)
        scale = self.scale(input) * self.scale_0
        
        return torch.cos(omega)*torch.exp(-(scale**2))

class ComplexGaborLayer(nn.Module):
    '''
        Implicit representation with complex Gabor nonlinearity
        
        Inputs;
            in_features: Input features
            out_features; Output features
            bias: if True, enable bias for the linear operation
            is_first: Legacy SIREN parameter
            omega_0: Legacy SIREN parameter
            omega0: Frequency of Gabor sinusoid term
            sigma0: Scaling of Gabor Gaussian term
            trainable: If True, omega and sigma are trainable parameters
    '''
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega0=10.0, sigma0=40.0,
                 trainable=False):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first
        
        self.in_features = in_features
        
        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat
            
        # Set trainable parameters if they are to be simultaneously optimized
        self.omega_0 = nn.Parameter(self.omega_0*torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(self.scale_0*torch.ones(1), trainable)
        
        self.linear = nn.Linear(in_features,
                                out_features,
                                bias=bias,
                                dtype=dtype)
    
    def forward(self, input):
        lin = self.linear(input)
        omega = self.omega_0 * lin
        scale = self.scale_0 * lin
        
        return torch.exp(1j*omega - scale.abs().square())
    
class WIRE(nn.Module):
    def __init__(self, 
                 params,
                ):
        super().__init__()
        
        # All results in the paper were with the default complex 'gabor' nonlinearity
        self.nonlin = ComplexGaborLayer
        
        # Since complex numbers are two real numbers, reduce the number of 
        # hidden parameters by 2
        dtype = torch.cfloat
        self.complex = True
        self.wavelet = 'gabor'    
        hidden_layers = params['network_depth']
        hidden_features = params['network_width']
        in_features = params['network_input_size']
        out_features = params['network_output_size']
        first_omega_0 = params["first_omega_0"]
        hidden_omega_0 = params["hidden_omega_0"]
        scale = params["scale"]

        hidden_features = int(hidden_features/np.sqrt(2))
        
        # Legacy parameter
        self.pos_encode = False
            
        self.net = []
        self.net.append(self.nonlin(in_features,
                                    hidden_features, 
                                    omega0=first_omega_0,
                                    sigma0=scale,
                                    is_first=True,
                                    trainable=False))

        for i in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features,
                                        hidden_features, 
                                        omega0=hidden_omega_0,
                                        sigma0=scale))

        final_linear = nn.Linear(hidden_features,
                                 out_features,
                                 dtype=dtype)            
        self.net.append(final_linear)
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        output = self.net(coords)
        
        if self.wavelet == 'gabor':
            return output.real
         
        return output
    
class LinearWeightedAvg(nn.Module):
    def __init__(self, n_inputs, n_heads, device):
        super(LinearWeightedAvg, self).__init__()
        self.weights = []
        for _ in range(n_heads):
            self.weights.append(nn.ParameterList([nn.Parameter(torch.tensor([1/n_heads]).to(device)) for i in range(n_inputs)]))

    def forward(self, input, weight_idx):
        res = 0
        for idx, inp in enumerate(input):
            res += inp * self.weights[weight_idx][idx]
        return res

class MultiHeadWrapper(nn.Module):
    def __init__(self, 
                 backbone=None,
                 no_heads=4,
                 params=None,
                 device="cpu",
                 last_tanh=True,
                 detach_outs=True
                ):
        super().__init__()
        self.backbone = backbone
        self.no_heads = params["no_heads"]
        assert no_heads > 0
        self.heads = []
        for _ in range(self.no_heads):
            self.heads.append(SIREN(params).to(device=device))
        # self.weighted_avg = LinearWeightedAvg(no_heads, no_heads, device).to(device=device)
        config = {
            "network_input_size": 2,
            "network_output_size": no_heads,
            "network_depth": 5,           
            "network_width": 128,         
        }
        self.weighted_avg = FFN(config).to(device=device)
        self.last_tanh = last_tanh
        self.detach_outs = detach_outs
        # self.weighted_avg = nn.Linear(no_heads*output_dim+1, 2).to(device=device)
    
    def forward(self, coords, dist_to_center):
        x = coords
        if self.backbone:
            x = self.backbone(x)
        # Get radial distance
        weights = self.weighted_avg(dist_to_center.unsqueeze(dim=-1))
        res = 0
        out = [head(x) for head in self.heads]
        if self.detach_outs:
            # Detach out to prevent gradients from updating other networks
            out_detached = [o.detach().requires_grad_(True) for o in out]
            # out_detached = [o.detach().clone().requires_grad_(True) for o in out]
            res = torch.sum(weights.unsqueeze(1) * torch.stack(out_detached, dim=2), dim=2)
        else:
            #alternatively, just use outs for grads
            res = torch.sum(weights.unsqueeze(1) * torch.stack(out, dim=2), dim=2)

        # Constrain range
        if self.last_tanh:
            # res = torch.tanh(res)
            res = torch.clamp(res,min=-1, max=1)

        # res = self.weighted_avg(out, weight_idx)
        # Get overall result and final output
        return out, res

class MultiHeadWrapperLossEnsemble(nn.Module):
    def __init__(self, 
                 backbone=None,
                 no_heads=4,
                 params=None,
                 device="cpu",
                 last_tanh=True,
                 detach_outs=True
                ):
        super().__init__()
        self.backbone = backbone
        self.no_heads = 2*params["no_heads"]
        assert no_heads > 0
        self.heads = []
        for _ in range(self.no_heads):
            self.heads.append(SIREN(params).to(device=device))
        # self.weighted_avg = LinearWeightedAvg(no_heads, no_heads, device).to(device=device)
        config = {
            "network_input_size": 512,
            "network_output_size": self.no_heads,
            "network_depth": 3,           
            "network_width": 256,         
        }
        self.weighted_avg = FFN(config).to(device=device)
        self.last_tanh = last_tanh
        self.detach_outs = detach_outs
        # self.weighted_avg = nn.Linear(no_heads*output_dim+1, 2).to(device=device)
    def forward(self, coords, dist_to_center):
        x = coords
        if self.backbone:
            x = self.backbone(x)
        # Get radial distance
        weights = self.weighted_avg(x)
        res = 0
        out = [head(x) for head in self.heads]
        if self.detach_outs:
            # Detach out to prevent gradients from updating other networks
            out_detached = [o.detach().requires_grad_(True) for o in out]
            # out_detached = [o.detach().clone().requires_grad_(True) for o in out]
            res = torch.sum(weights.unsqueeze(1) * torch.stack(out_detached, dim=2), dim=2)
        else:
            #alternatively, just use outs for grads
            res = torch.sum(weights.unsqueeze(1) * torch.stack(out, dim=2), dim=2)

        # Constrain range
        if self.last_tanh:
            # res = torch.tanh(res)
            res = torch.clamp(res,min=-1, max=1)
        return out, res

class ScalerWrapper(nn.Module):
    def __init__(self, 
                 backbone=None,
                 device="cpu",
                 last_tanh=True,
                ):
        super().__init__()
        self.backbone = backbone
        config = {
            "network_input_size": 2,
            "network_output_size": 1,
            "network_depth": 8,           
            "network_width": 512,         
        }
        self.scaler = FFN(config).to(device=device)
        self.last_tanh = last_tanh
        # self.weighted_avg = nn.Linear(no_heads*output_dim+1, 2).to(device=device)
    def forward(self, coords, dist_to_center):
        x = coords
        if self.backbone:
            x = self.backbone(x)
        # Get radial distance
        scales = self.scaler(dist_to_center)
        # scales = torch.clamp(scales, min=0, max=1)
        res = x * torch.exp(-scales)
        return res