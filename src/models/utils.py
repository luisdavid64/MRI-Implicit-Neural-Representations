import os
import yaml
import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from tabulate import tabulate
from data.nerp_datasets import MRIDataset, MRIDatasetWithDistances
from skimage.metrics import structural_similarity
import numpy as np
import matplotlib.pyplot as plt
import fastmri

def get_device(net_name):
    device = ("cuda" if torch.cuda.is_available() else 
                        ("mps" if torch.backends.mps.is_available() else "cpu"))

    # if "WIRE" in net_name and device == "mps":
    #     return torch.device("cpu")
    return torch.device(device)

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.Loader)

def prepare_sub_folder(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory


def get_data_loader(data, data_root, set, batch_size, transform=True,
                    num_workers=0,  sample=0, slice=0, challenge="multicoil", shuffle=True, full_norm=False, normalization="max", use_dists=False):
    
    MRIData = MRIDataset
    if use_dists:
        MRIData = MRIDatasetWithDistances

    if data in ['brain', 'knee']:
        dataset = MRIData(data_class=data, data_root=data_root, set=set, transform=transform, sample=sample, slice=slice, full_norm=full_norm, normalization = normalization)  #, img_dim)

    loader = DataLoader(dataset=dataset, 
                        batch_size=batch_size, 
                        shuffle=shuffle, 
                        drop_last=False, 
                        num_workers=num_workers,
                        pin_memory=True
                        )

    val_loader = DataLoader(dataset=dataset, 
                        batch_size=batch_size, 
                        shuffle=False, 
                        drop_last=False, 
                        num_workers=num_workers,
                        pin_memory=True
                )
    return dataset, loader, val_loader


def save_image_3d(tensor, slice_idx, file_name):
    '''
    tensor: [bs, c, h, w, 1]
    '''
    image_num = len(slice_idx)
    tensor = tensor[0, slice_idx, ...].permute(0, 3, 1, 2).cpu().data  # [c, 1, h, w]
    image_grid = vutils.make_grid(tensor, nrow=image_num, padding=0, normalize=True, scale_each=True)
    vutils.save_image(image_grid, file_name, nrow=1)



def map_coordinates(input, coordinates):
    ''' PyTorch version of scipy.ndimage.interpolation.map_coordinates
    input: (B, H, W, C)
    coordinates: (2, ...)
    '''
    bs, h, w, c = input.size()

    def _coordinates_pad_wrap(h, w, coordinates):
        coordinates[0] = coordinates[0] % h
        coordinates[1] = coordinates[1] % w
        return coordinates

    co_floor = torch.floor(coordinates).long()
    co_ceil = torch.ceil(coordinates).long()
    d1 = (coordinates[1] - co_floor[1].float())
    d2 = (coordinates[0] - co_floor[0].float())
    co_floor = _coordinates_pad_wrap(h, w, co_floor)
    co_ceil = _coordinates_pad_wrap(h, w, co_ceil)

    f00 = input[:, co_floor[0], co_floor[1], :]
    f10 = input[:, co_floor[0], co_ceil[1], :]
    f01 = input[:, co_ceil[0], co_floor[1], :]
    f11 = input[:, co_ceil[0], co_ceil[1], :]
    d1 = d1[None, :, :, None].expand(bs, -1, -1, c)
    d2 = d2[None, :, :, None].expand(bs, -1, -1, c)

    fx1 = f00 + d1 * (f10 - f00)
    fx2 = f01 + d1 * (f11 - f01)
    
    return fx1 + d2 * (fx2 - fx1)

def ssim(x, xhat):
    if torch.is_tensor(x):
        x = x.numpy()
    if torch.is_tensor(xhat):
        xhat = xhat.numpy()
    data_range = np.maximum(x.max(), xhat.max()) - np.minimum(x.min(), xhat.min())
    return structural_similarity(x,xhat, data_range=data_range)

def psnr(x, xhat, epsilon=1e-10):
    ''' Compute Peak Signal to Noise Ratio in dB

        Inputs:
            x: Ground truth signal
            xhat: Reconstructed signal

        Outputs:
            snrval: PSNR in dB
    '''
    denom = torch.mean((x - xhat) ** 2)

    snrval = 10*torch.log10(torch.max(x)/(denom + epsilon))

    return snrval

# Save MRI image with matplotlib
def save_im(image, image_directory, image_name, is_kspace=False, smoothing_factor=8, vmax=None, vmin=None):
    if not is_kspace:
        if vmin and vmax:
            plt.imsave(os.path.join(image_directory, image_name), np.abs(image.numpy()), format="png", cmap="gray", vmin=vmin, vmax=vmax)
        else:
            plt.imsave(os.path.join(image_directory, image_name), np.abs(image.numpy()), format="png", cmap="gray")
    else:
        kspace_grid = fastmri.complex_abs(image.detach()).squeeze(dim=0)
        kspace_grid = fastmri.rss(kspace_grid, dim=0)
        sf = torch.tensor(smoothing_factor, dtype=torch.float32)
        kspace_grid *= torch.expm1(sf) / kspace_grid.max()
        kspace_grid = torch.log1p(kspace_grid)  # Adds 1 to input for natural log.
        kspace_grid /= kspace_grid.max()  # Standardization to 0~1 range.
        kspace_grid = kspace_grid.squeeze().to(device='cpu', non_blocking=True)
        plt.imsave(os.path.join(image_directory, image_name), kspace_grid.numpy(), format="png", cmap="gray")

    plt.clf()

def stats_per_coil(im_recon, C):
    stats_coil = []
    for i in range(C):
        mean = (im_recon[i,:,:,:].mean())
        std = (im_recon[i,:,:,:].std())
        max = (im_recon[i,:,:,:].max())
        min = (im_recon[i,:,:,:].min())
        stats_coil.append(
            (i, mean, std, max, min)
        )
    headers = ["coil", "mean", "std", "max", "min"]
    table = tabulate(stats_coil, headers=headers)
    title = "{} Reconstruction Statistics Per Coil".format("K-space")
    print("{}\n{}".format(title,table))