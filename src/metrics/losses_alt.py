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

class CenterLoss2(torch.nn.Module):
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
        dist_to_center2 = kcoords[...,1]**2 + kcoords[...,2]**2

        if input.dtype == torch.float:
            input = torch.view_as_complex(input) #* filter_value
        if target.dtype == torch.float:
            target = torch.view_as_complex(target)

        error = input - target

        error_loss = (error.abs()/(input.detach().abs()+self.eps))**2

        target_abs = torch.abs(target)
        input_abs = torch.abs(input)
        # Magnitude loss
        abs_loss = ((target_abs - input_abs)/(input.detach().abs()+self.eps))**2

        center_loss = torch.tensor([0.0], device=device) 
        for masking_dist in range (1,6):
            masking_ratio = (masking_dist - 1) / 5.0 
            if masking_ratio == 0: masking_ratio = 0.1
            masking_ratio_2 = (masking_dist) / 5.0
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
            center_loss += ((diff_pred - diff_gt)**2).mean()

        # assert input.shape == target.shape
        return error_loss.mean() + abs_loss.mean() + center_loss, 0