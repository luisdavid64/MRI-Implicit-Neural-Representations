import torch
device = ("cuda" if torch.cuda.is_available() else "cpu")
import fastmri

class RadialL2Loss(torch.nn.Module):
    """
    Outdated Loss function
    """
    def __init__(self, eps=1e-9):
        super(RadialL2Loss, self).__init__()
        self.eps = eps
    def forward(self, x, y):
        loss = torch.nn.functional.mse_loss(x,y)
        # Magnitude loss
        loss += 0.1 * torch.nn.functional.mse_loss(fastmri.complex_abs(x), fastmri.complex_abs(y))
        return loss

class MSLELoss(torch.nn.Module):
    """
    An MSE Loss with values squeezed with a log function
    """
    def __init__(self, eps=1e-9):
        super(MSLELoss, self).__init__()
        self.eps = eps
    def forward(self, x, y):
        loss = torch.nn.functional.mse_loss(torch.log(x+1+self.eps),torch.log(y+1 +self.eps))
        return loss


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

class FocalFrequencyLoss(torch.nn.Module):
    """The torch.nn.Module class that implements focal frequency loss - a
    frequency domain loss function for optimizing generative models.

    Ref:
    Focal Frequency Loss for Image Reconstruction and Synthesis. In ICCV 2021.
    <https://arxiv.org/pdf/2012.12821.pdf>

    Args:
        loss_weight (float): weight for focal frequency loss. Default: 1.0
        alpha (float): the scaling factor alpha of the spectrum weight matrix for flexibility. Default: 1.0
        log_matrix (bool): whether to adjust the spectrum weight matrix by logarithm. Default: False
        batch_matrix (bool): whether to calculate the spectrum weight matrix using batch-based statistics. Default: False
    """

    def __init__(self, loss_weight=1.0, alpha=1.0, log_matrix=True, batch_matrix=False):
        super(FocalFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        # spectrum weight matrix
        if matrix is not None:
            # if the matrix is predefined
            weight_matrix = matrix.detach()
        else:
            # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
            matrix_tmp = (recon_freq - real_freq) ** 2
            # Distance from center (effectively filtering)
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha

            # whether to adjust the spectrum weight matrix by logarithm
            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)

            # whether to calculate the spectrum weight matrix using batch-based statistics
            if self.batch_matrix:
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                matrix_tmp = matrix_tmp / matrix_tmp.max()
                # matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]

            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()

        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
            'The values of spectrum weight matrix should be in the range [0, 1], '
            'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))

        # frequency distance using (squared) Euclidean distance
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]

        # dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * freq_distance
        return torch.mean(loss)

    def forward(self, pred, target, matrix=None, **kwargs):
        # calculate focal frequency loss
        return self.loss_formulation(pred, target, matrix) * self.loss_weight

class TanhL2Loss(torch.nn.Module):
    """
    An L2 Loss with values squeezed with a tanh function
    """
    def __init__(self, with_mag=False, rho=0.5):
        super(TanhL2Loss, self).__init__()
        self.with_mag = with_mag
        self.rho = rho
        self.sigma = 2
    def forward(self, x, y, kcoords):
        loss = torch.mean(torch.pow((torch.tanh(x) - torch.tanh(y)), 2))
        # Add real and imaginary part and we get mag
        if self.with_mag:
            # Control the size of the periphery
            xabs = torch.sqrt(x[...,0]**2 + x[...,1]**2)
            yabs = torch.sqrt(y[...,0]**2 + y[...,1]**2)
            reg = torch.mean(torch.pow((torch.tanh(xabs) - torch.tanh(yabs)), 2))
            loss += self.rho * reg 
        return loss, 0

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
        input = input.to(device)
        target = target.to(device)
        kcoords = kcoords.to(device)
        error_loss = ((input - target)**2)
        dist_to_center2 = kcoords[...,1]**2 + kcoords[...,2]**2
        filter_value = torch.exp(-dist_to_center2/(2*self.sigma**2)).unsqueeze(-1)

        if input.dtype == torch.float:
            input = torch.view_as_complex(input) #* filter_value
        if target.dtype == torch.float:
            target = torch.view_as_complex(target)

        error = input - target
        error_loss = (error.abs()/(input.detach().abs()+self.eps))**2

        target_abs = torch.abs(target)
        input_abs = torch.abs(input)
        abs_loss = ((target - input).abs()/(input.detach().abs()+self.eps))**2
        reg_error = (input - input * filter_value)
        reg = self.factor * (reg_error.abs()/(input.detach().abs()+self.eps))**2
        #Magnitude loss
        N_BANDS=2
        center_loss = torch.tensor([0.0], device=device) 
        for masking_dist in range (1,N_BANDS + 1):
            masking_ratio = (masking_dist - 1) / N_BANDS 
            if masking_ratio == 0: masking_ratio = 0.1
            masking_ratio_2 = (masking_dist) / N_BANDS
            mask_1 = self.radial_mask(dist_to_center2, masking_ratio)
            mask_2 = self.radial_mask(dist_to_center2, masking_ratio_2)
            mask_2 = mask_2 & ~mask_1 
            masked_1 = input_abs[mask_1]
            masked_2 = input_abs[mask_2]
            # Take as many as there exists in both
            n =  min(self.min_sample,min(len(masked_1), len(masked_2)))
            if n == 0: continue
            # Choose these randomly, as before we only the first
            a = torch.randperm(masked_1.size(0))[:n]
            b = torch.randperm(masked_2.size(0))[:n]
            diff_pred = masked_1[a] - masked_2[b]
            diff_gt = target_abs[mask_1][a] - target_abs[mask_2][b]
            # If they are close together in radial space then it doesn't matter?
            center_loss += (((diff_gt - diff_pred))**2).mean()

        # assert input.shape == target.shape
        return  0.1 * error_loss.mean() + 0.9 * (abs_loss.mean() + reg.mean()) + 0.1*center_loss, 0

        
class LogSpaceLoss(torch.nn.Module):
    """
    This loss function is a simplified HDR loss
    """
    def __init__(self, config):
        super().__init__()
        self.sigma = float(config['hdr_ff_sigma'])
        self.eps = float(config['hdr_eps'])
        self.factor = float(config['hdr_ff_factor'])

    def forward(self, input, target):
        if input.dtype == torch.float:
            input = torch.view_as_complex(input) #* filter_value
        if target.dtype == torch.float:
            target = torch.view_as_complex(target)
        
        # assert input.shape == target.shape
        error = input - target
        error_loss = ((error.abs()/(input.detach().abs()+self.eps))**2).mean()
        return error_loss


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

        # loss = (error.abs()/(input.detach().abs()+self.eps))**2
        loss = torch.log(error.abs()/(input.detach().abs()+self.eps))**2
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

class ConsistencyLoss(torch.nn.Module):
    """

    This loss function is to be used with overlapping regions
    To supervise on shared points of the regions.
    Args:
        bounds: list of boundaries of circles centered at origin.

    """
    def __init__(self, bounds):
        super().__init__()
        self.bounds = bounds

    """

    Args:
    - input: prediction of the network/networks
    - dist: coordinate distances of the prediction from the center. 

    Returns:
    - loss: PyTorch Variable holding a scalar giving the 'consistency' loss
        
    """
    def forward(self, input, dist):
        loss = 0
        for i in range(len(self.bounds)-1):
            bound = self.bounds[i]
            ind = torch.where((dist < bound[0]) 
                            | (dist > bound[1]))
            if ind[0].numel():
                # We detach first tensor as we want effect only on subsequent layers
                loss += torch.nn.functional.mse_loss(input[i][ind].detach(), input[i+1][ind])
        return loss

def tv_loss(img, weight=0.0001):
    """
    Compute total variation loss.

    Inputs:
    - img: PyTorch Variable of shape (H, W,2) holding an input coil.
    - weight: Influence of total variation

    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    # w_variance = torch.sum(torch.pow(img[:,:-1,:] - img[:,1:,:], 2))
    w_variance = torch.nn.functional.l1_loss(img[:,:-1,:],img[:,1:,:])
    # h_variance = torch.sum(torch.pow(img[:-1,:,:] - img[1:,:,:], 2))
    h_variance = torch.nn.functional.l1_loss(img[:-1,:,:],img[1:,:,:]) 
    loss = weight*(h_variance + w_variance)
    return loss
            
            
