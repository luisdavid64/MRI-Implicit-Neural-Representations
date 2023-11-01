import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import fastmri
import h5py
from pathlib import Path
from fastmri.data import transforms as T
from matplotlib import pyplot as plt

def normalize_image(data, full_norm=False):
    data_max = data.max()
    if not full_norm:
        return data / data_max
    data_min = data.min()
    return (data - data_min) / (data_max - data_min)

def create_grid_3d(c, h, w):
    grid_z, grid_y, grid_x = torch.meshgrid([torch.linspace(0, 1, steps=c), \
                                            torch.linspace(0, 1, steps=h), \
                                            torch.linspace(0, 1, steps=w)])
    grid = torch.stack([grid_z, grid_y, grid_x], dim=-1)
    return grid

def create_coords(c, h, w):
    Z, Y, X = torch.meshgrid(torch.linspace(-1, 1, c),
                              torch.linspace(-1, 1, h),
                              torch.linspace(-1, 1, w))
    grid = torch.hstack((Z.reshape(-1, 1),
                            Y.reshape(-1, 1),
                            X.reshape(-1, 1)))
    return grid

def display_tensor_stats(tensor):
    shape, vmin, vmax, vmean, vstd = tensor.shape, tensor.min(), tensor.max(), torch.mean(tensor), torch.std(tensor)
    print('shape:{} | min:{:.3f} | max:{:.3f} | mean:{:.3f} | std:{:.3f}'.format(shape, vmin, vmax, vmean, vstd))


class ImageDataset_3D(Dataset):

    def __init__(self, img_path, img_dim):
        '''
        img_dim: new image size [z, h, w]
        '''
        self.img_dim = (img_dim, img_dim, img_dim) if type(img_dim) == int else tuple(img_dim)
        image = np.load(img_path)['data']  # [C, H, W]

        # Crop slices in z dim
        center_idx = int(image.shape[0] / 2)
        num_slice = int(self.img_dim[0] / 2)
        image = image[center_idx-num_slice:center_idx+num_slice, :, :]
        im_size = image.shape
        print(image.shape, center_idx, num_slice)

        # Complete 3D input image as a squared x-y image
        if not(im_size[1] == im_size[2]):
            zerp_padding = np.zeros([im_size[0], im_size[1], np.int((im_size[1]-im_size[2])/2)])
            image = np.concatenate([zerp_padding, image, zerp_padding], axis=-1)

        # Resize image in x-y plane
        image = torch.tensor(image, dtype=torch.float32)[None, ...]  # [B, C, H, W]
        image = F.interpolate(image, size=(self.img_dim[1], self.img_dim[2]), mode='bilinear', align_corners=False)

        # Scaling normalization
        image = image / torch.max(image)  # [B, C, H, W], [0, 1]
        self.img = image.permute(1, 2, 3, 0)  # [C, H, W, 1]
        display_tensor_stats(self.img)

    def __getitem__(self, idx):
        grid = create_grid_3d(*self.img_dim)
        return grid, self.img

    def __len__(self):
        return 1

class MRIDataset(Dataset):
    def __init__(self, data_class='brain', challenge='multicoil', set="train", transform=True, sample=0, slice=0):
        # self.batch_size = batch_size
        self.challenge = challenge
        self.transform = transform
        self.data_class = data_class  # brain or knee
        self.set = set
        self.root = "data/{}_{}_{}/".format(self.data_class, self.challenge, self.set)

        path = Path(self.root)
        files = sorted(path.glob('*.h5'))

        # Choose a sample number form the files
        file = files[sample]
        self.file_name = file

        data = h5py.File(str(file.resolve()))['kspace'][()]
        # Choose a slice
        data = data[slice]
        data = T.to_tensor(data)
        if self.transform:
            data = self.__perform_fft(data)

        # Make range of image [0,1]
        data = normalize_image(data=data)

        display_tensor_stats(data)
        self.shape = data.shape # (Coil Dim, Height, Width)
        C,H,W,S = self.shape
        # Flatten image and grid
        # What to do with complex numbers?
        self.image = data.reshape((C*H*W),S) # Dim: (C*H*W,1), flattened 2d image with coil dim
        self.coords = create_coords(C,H,W) # Dim: (C*H*W,3), flattened 2d coords with coil dim

    @classmethod
    def __perform_fft(cls, k_space):

        transformed = fastmri.ifft2c(k_space)
        # transformed = fastmri.complex_abs(transformed)
        # transformed = fastmri.rss(transformed, dim=0)  # coil dimension
        # transformed = transformed.unsqueeze(dim=0).unsqueeze(dim=-1)

        return transformed
    
    @property
    def file(self):
        return self.file_name

    @property
    def img_shape(self):
        return self.shape

    def __getitem__(self, idx):
        return self.coords[idx], self.image[idx]

    def __len__(self):
        return len(self.image)  #self.X.shape[0]

if __name__ == "__main__":
    x = MRIDataset()