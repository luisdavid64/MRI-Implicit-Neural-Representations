import torch
device = ("cuda" if torch.cuda.is_available() else "cpu")
# class CenterLoss(torch.nn.Module):
#     """
#     HDR loss function with frequency filtering (v4)
#     """
#     def __init__(self, config):
#         super().__init__()
#         self.sigma = float(config['hdr_ff_sigma'])
#         self.eps = float(config['hdr_eps'])
#         self.factor = float(config['hdr_ff_factor'])
#         # self.rank_loss = MarginRankingLoss(margin=0)

#     def forward(self, input, target, kcoords):
#         input = input.to(device)
#         target = target.to(device)
#         kcoords = kcoords.cpu()
#         dist_to_center2 = kcoords[...,1]**2 + kcoords[...,2]**2

#         if input.dtype == torch.float:
#             input = torch.view_as_complex(input) #* filter_value
#         if target.dtype == torch.float:
#             target = torch.view_as_complex(target)

#         target_abs = torch.abs(target)
#         input_abs = torch.abs(input)
#         a = torch.randint(0, input_abs.size(0), (input_abs.size(0),), device=device)
#         b = torch.randint(0, target_abs.size(0), (target_abs.size(0),), device=device)
#         diff_pred = input_abs[a] - input_abs[b]
#         diff_gt = target_abs[a] - target_abs[b]
#         # If they are close together in radial space then it doesn't matter?
#         weight = 10*(torch.abs(dist_to_center2[a] - dist_to_center2[b])) + 1
#         rank_loss = torch.mean(weight*(diff_pred - diff_gt)**2)

#         # assert input.shape == target.shape
#         error = input - target
#         error_loss = ((error.abs()/(input.detach().abs()+self.eps))**2).mean()
#         return error_loss + rank_loss, 0

class CenterLoss(torch.nn.Module):
    """
    HDR loss function with frequency filtering (v4)
    """
    def __init__(self, config):
        super().__init__()
        self.sigma = float(config['hdr_ff_sigma'])
        self.eps = float(config['hdr_eps'])
        self.factor = float(config['hdr_ff_factor'])
        # self.rank_loss = MarginRankingLoss(margin=0)
    
    def radial_mask(self, dist, percent):
        return dist <= percent

    def forward(self, input, target, kcoords):
        # error_loss = ((input - target)**2).mean()
        input = input.to(device)
        target = target.to(device)
        kcoords = kcoords.to(device)
        dist_to_center2 = kcoords[...,1]**2 + kcoords[...,2]**2
        mask = self.radial_mask(dist_to_center2, 0.3)

        if input.dtype == torch.float:
            input = torch.view_as_complex(input) #* filter_value
        if target.dtype == torch.float:
            target = torch.view_as_complex(target)

        error = input - target

        error_loss = (error.abs()/(input.detach().abs()+self.eps))**2

        target_abs = torch.abs(target)
        input_abs = torch.abs(input)
        # Magnitude loss
        abs_loss = ((target_abs - input_abs)/(input.detach().abs()+self.eps))

        center_loss = 4*torch.nn.functional.mse_loss(input_abs[mask],target_abs[mask]) 

        # assert input.shape == target.shape
        return error_loss.mean() + abs_loss.mean() + center_loss, 0