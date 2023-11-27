import torch
device = ("cuda" if torch.cuda.is_available() else "cpu")
import fastmri
from torch.nn import MarginRankingLoss

class TLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()
    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        if X.dtype == torch.float:
            X = torch.view_as_complex(X) #* filter_value
        if Y.dtype == torch.float:
            Y = torch.view_as_complex(Y)

        assert X.is_complex()
        assert Y.is_complex()

        mag_input = torch.abs(X)
        mag_target = torch.abs(Y)
        cross = torch.abs(X.real * Y.imag - X.imag * Y.real)

        angle = torch.atan2(X.imag, X.real) - torch.atan2(Y.imag, Y.real)
        ploss = torch.abs(cross) / (mag_input + 1e-8)

        aligned_mask = (torch.cos(angle) < 0).bool()

        final_term = torch.zeros_like(ploss)
        final_term[aligned_mask] = mag_target[aligned_mask] + (mag_target[aligned_mask] - ploss[aligned_mask])
        final_term[~aligned_mask] = ploss[~aligned_mask]
        return (final_term + torch.nn.functional.mse_loss(mag_input, mag_target)).mean()

class CenterLoss(torch.nn.Module):
    """
    HDR loss function with frequency filtering (v4)
    """
    def __init__(self, config):
        super().__init__()
        self.sigma = float(config['hdr_ff_sigma'])
        self.eps = float(config['hdr_eps'])
        self.factor = float(config['hdr_ff_factor'])
        self.min_sample = int(config['min_sample'])
        # self.rank_loss = MarginRankingLoss(margin=0)
    
    def radial_mask(self, dist, percent):
        return dist <= percent

    def forward(self, input, target, kcoords):
        # error_loss = ((input - target)**2).mean()
        input = input.to(device)
        target = target.to(device)
        kcoords = kcoords.to(device)
        # dist_to_center2 = kcoords[...,1]**2 + kcoords[...,2]**2
        # filter_value = torch.exp(-dist_to_center2/(2*self.sigma**2)).unsqueeze(-1)

        if input.dtype == torch.float:
            input = torch.view_as_complex(input) #* filter_value
        if target.dtype == torch.float:
            target = torch.view_as_complex(target)

        error = input - target

        error_loss = (error.abs()/(input.detach().abs()+(self.eps)))**2

        # target_abs = torch.abs(target)
        # input_abs = torch.abs(input)
        # Magnitude loss
        abs_loss = ((target.abs() - input.abs()))**2

        # assert input.shape == target.shape
        return  error_loss.mean() + 0.5*abs_loss.mean(), 0

        
class LogSpaceLoss(torch.nn.Module):
    """
    HDR loss function with frequency filtering (v4)
    """
    def __init__(self, config):
        super().__init__()
        self.sigma = float(config['hdr_ff_sigma'])
        self.eps = float(config['hdr_eps'])
        self.factor = float(config['hdr_ff_factor'])

    def forward(self, input, target):
        input = input.cpu()
        target = target.cpu()
        input_abs =None
        if input.shape[-1] > 2:
            input_abs = input[...,2]
            input_com = torch.clone(input[...,0:2])
        if target.shape[-1] > 2:
            target_abs = target[...,2]
            target_com = torch.clone(target[...,0:2])

        mag_loss = 0
        if input_abs != None:
            mag_loss = torch.nn.functional.mse_loss(input_abs, target_abs)
            abs_value = fastmri.complex_abs(input_com) 
            abs_loss = torch.nn.functional.mse_loss(input_abs, abs_value)
            mag_loss = mag_loss + abs_loss
            
        if input_com.dtype == torch.float:
            input_com = torch.view_as_complex(input_com) #* filter_value
        if target_com.dtype == torch.float:
            target_com = torch.view_as_complex(target_com)
        
        # assert input.shape == target.shape
        error = input_com - target_com
        error_loss = ((error.abs()/(input_com.detach().abs()+self.eps))**2).mean()
        return error_loss + mag_loss


class HDRLoss_FF(torch.nn.Module):
    """
    HDR loss function with frequency filtering (v4)
    """
    def __init__(self, config):
        super().__init__()
        self.sigma = float(config['hdr_ff_sigma'])
        self.eps = float(config['hdr_eps'])
        self.factor = float(config['hdr_ff_factor'])

    def forward(self, input, target, kcoords, weights=None, reduce=True):
        input = input.to(device)
        target = target.to(device)
        kcoords = kcoords.to(device)
        # coords shape again: 
        dist_to_center2 = kcoords[...,1]**2 + kcoords[...,2]**2
        filter_value = torch.exp(-dist_to_center2/(2*self.sigma**2)).unsqueeze(-1)

        if input.dtype == torch.float:
            input = torch.view_as_complex(input) #* filter_value
        if target.dtype == torch.float:
            target = torch.view_as_complex(target)
        
        assert input.shape == target.shape
        error = input - target
        # error = error * filter_value

        loss = (error.abs()/(input.detach().abs()+self.eps))**2
        if weights is not None:
            loss = loss * weights.unsqueeze(-1)

        reg_error = (input - input * filter_value)
        reg = self.factor * (reg_error.abs()/(input.detach().abs()+self.eps))**2

        if reduce:
            return loss.mean() + reg.mean(), reg.mean()
        else:
            return loss, reg
        

class AdaptiveHDRLoss(torch.nn.Module):
    """
    HDR loss function with frequency filtering (v4)
    """
    def __init__(self, config):
        super().__init__()
        self.sigma = float(config['hdr_ff_sigma'])
        self.eps = float(config['eps'])
        self.factor = float(config['hdr_ff_factor'])

    def forward(self, input, target, reduce=True):

        if input.dtype == torch.float:
            input = torch.view_as_complex(input) #* filter_value
        if target.dtype == torch.float:
            target = torch.view_as_complex(target)
        
        assert input.shape == target.shape
        error = input - target

        loss = (-error.abs()/((input.detach().abs()+self.eps)**2))**2

        if reduce:
            return loss.mean()