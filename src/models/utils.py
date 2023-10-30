import os
import yaml
import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from src.data.nerp_datasets import ImageDataset_3D
from src.image_dataloader.dataloader import H5Dataset

device = torch.device("cuda" if torch.cuda.is_available() else 
                    ("mps" if torch.backends.mps.is_available() else "cpu"))


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


def get_data_loader(data, img_path, img_dim, img_slice, train, test, batch_size, transform=True,
                    num_workers=0,  return_data_idx=False):
    
    if data == 'nerp':
        dataset = ImageDataset_3D(img_path, img_dim)

    elif data in ['brain', 'knee']:
        dataset = H5Dataset(data_class=data, train=train, test=test, transform=transform)  #, img_dim)

    loader = DataLoader(dataset=dataset, 
                        batch_size=batch_size, 
                        shuffle=train, 
                        drop_last=train, 
                        num_workers=num_workers)
    return loader


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
