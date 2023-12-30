import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import fastmri
import h5py
from pathlib import Path
from fastmri.data import transforms as T
from matplotlib import pyplot as plt
import os
from tabulate import tabulate
import xml.etree.ElementTree as etree
from torchvision.transforms.functional import equalize
from typing import (
    Sequence,
)
from math import ceil
from torch.distributions import Normal


def gaussian_kernel_1d(sigma: float, num_sigmas: float = 10.) -> torch.Tensor:
    
    radius = ceil(num_sigmas * sigma)
    support = torch.arange(-radius, radius + 1, dtype=torch.float)
    kernel = Normal(loc=0, scale=sigma).log_prob(support).exp_()
    # Ensure kernel weights sum to 1, so that image brightness is not altered
    return kernel.mul_(1 / kernel.sum())

def gaussian_filter_2d(img: torch.Tensor, sigma: float) -> torch.Tensor:
    
    kernel_1d = gaussian_kernel_1d(sigma)  # Create 1D Gaussian kernel
    
    padding = len(kernel_1d) // 2  # Ensure that image size does not change
    img = img  # Need 4D data for ``conv2d()``
    # Convolve along columns and rows
    img = torch.nn.functional.conv2d(img, weight=kernel_1d.view(1, 1, -1, 1), padding=(padding, 0))
    img = torch.nn.functional.conv2d(img, weight=kernel_1d.view(1, 1, 1, -1), padding=(0, padding))
    return img  # Make 2D again

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def et_query(
    root: etree.Element,
    qlist: Sequence[str],
    namespace: str = "http://www.ismrm.org/ISMRMRD",
) -> str:
    """
    ElementTree query function.

    This can be used to query an xml document via ElementTree. It uses qlist
    for nested queries.

    Args:
        root: Root of the xml to search through.
        qlist: A list of strings for nested searches, e.g. ["Encoding",
            "matrixSize"]
        namespace: Optional; xml namespace to prepend query.

    Returns:
        The retrieved data as a string.
    """
    s = "."
    prefix = "ismrmrd_namespace"

    ns = {prefix: namespace}

    for el in qlist:
        s = s + f"//{prefix}:{el}"

    value = root.find(s, ns)
    if value is None:
        raise RuntimeError("Element not found")

    return str(value.text)


def complex_center_crop(data, shape):
    """
    Apply a center crop to the input image or batch of complex images.

    Args:
        data (torch.Tensor): The complex input tensor to be center cropped. It should
            have at least 3 dimensions and the cropping is applied along dimensions
            -3 and -2 and the last dimensions should have a size of 2.
        shape (int, int): The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        torch.Tensor: The center cropped image
    """
    # Make sure crop fits one dimension at least
    if data.shape[-2] < shape[1]:
        shape = (data.shape[-2], data.shape[-2])
    assert 0 < shape[0] <= data.shape[-3]
    assert 0 < shape[1] <= data.shape[-2]
    w_from = (data.shape[-3] - shape[0]) // 2
    h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to, :]

def normalize_image(data, full_norm=False):
    
    C,_,_,_ = data.shape
    # data_flat = data.reshape(C,-1)
    # norm = torch.abs(data_flat).max()
    norm = fastmri.complex_abs(data).max()
    return data/norm 
    
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

def display_tensor_stats(tensor, with_plot=False):
    if with_plot:
        plt.boxplot(torch.view_as_complex(tensor).abs().reshape(-1), vert=False)  # vert=False makes it a horizontal boxplot
        plt.title('Plot of K-space')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        # Show the plot
        plt.show()
    shape, vmin, vmax, vmean, vstd = tensor.shape, tensor.min(), tensor.max(), torch.mean(tensor), torch.std(tensor)
    print('shape:{} | min:{:.5f} | max:{:.5f} | mean:{:.5f} | std:{:.5f}'.format(shape, vmin, vmax, vmean, vstd))


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
    def __init__(self, 
                 data_class='brain', 
                 data_root="data",
                 challenge='multicoil', 
                 set="train", 
                 transform=True, 
                 sample=0, slice=0, 
                 full_norm=False, 
                 custom_file_or_path = None,
                 per_coil_stats=True,
                 centercrop=True,
                 normalization="max"
                 ):
        # self.batch_size = batch_size
        self.challenge = challenge
        self.transform = transform
        self.data_class = data_class  # brain or knee
        self.data_root = data_root
        self.set = set

        if custom_file_or_path is None or custom_file_or_path == "":
            self.root = "{}/{}_{}_{}/".format(self.data_root,self.data_class, self.challenge, self.set)
        else:
            self.root = custom_file_or_path


        # Load single image
        data, crop_size = self.__load_files(self.root, sample)

        # Choose a slice
        data = data[slice]
        data = T.to_tensor(data)
        if self.transform:
            data = self.__perform_fft(data)
            if centercrop:
                data = complex_center_crop(data, crop_size)
            data = normalize_image(data=data, full_norm=full_norm)
        else:
            data = self.__perform_fft(data)
            # Normalize data in image space
            if centercrop:
                data = complex_center_crop(data, crop_size)
            # data = normalize_image(data=data, full_norm=full_norm)
            data = fastmri.fft2c(data=data)
            data = self.__normalize_kspace(data, type=normalization)

        display_tensor_stats(data, with_plot=False)
        self.shape = data.shape # (Coil Dim, Height, Width)
        C,H,W,S = self.shape
        # Flatten image and grid
        # What to do with complex numbers?
        if per_coil_stats:
            stats_coil = []
            for i in range(C):
                mean = (data[i,:,:,:].mean())
                std = (data[i,:,:,:].std())
                max = (data[i,:,:,:].max())
                min = (data[i,:,:,:].min())
                stats_coil.append(
                    (i, mean, std, max, min)
                )
            headers = ["coil", "mean", "std", "max", "min"]
            table = tabulate(stats_coil, headers=headers)
            title = "{} Data Statistics Per Coil".format("Image" if transform else "K-space")
            print("{}\n{}".format(title,table))

        self.image = data.reshape((C*H*W),S) # Dim: (C*H*W,1), flattened 2d image with coil dim
        self.coords = create_coords(C,H,W) # Dim: (C*H*W,3), flattened 2d coords with coil dim

    @classmethod
    def __normalize_kspace(cls, k_space, type="max", eps=1e-9):
        print(type)
        if type == "abs_max":
            mx = fastmri.complex_abs(k_space).max().item()
            k_space = k_space/mx
        elif type == "max":
            mx = torch.abs(k_space).max().item()
            k_space = k_space/mx
        elif type == "gaussian_blur":
            mx = torch.abs(k_space).max().item()
            k_space = k_space/mx
            k_space = k_space.permute(0,3,1,2)
            for i in range(k_space.shape[1]):
                k_space[:,i:i+1,...] = gaussian_filter_2d(k_space[:,i:i+1,...], 0.1)
            k_space = k_space.permute(0,2,3,1)
        elif type == "max_std":
            mx = torch.abs(k_space).max().item()
            k_space = k_space/mx
            k_space = (k_space - k_space.mean()) / k_space.std()
            # Renormalize to 1
            k_space = k_space/k_space.max()
        elif type == "tonemap":
            k_space = k_space / ((k_space + 1))
            k_space = k_space / k_space.max().item()
            mu = k_space.mean(dim=(1,2,3),keepdim=True)
            k_space = k_space - mu
        elif type == "coil": 
            max_per_coil = fastmri.complex_abs(k_space).reshape(k_space.shape[0],-1).max(dim=-1,keepdim=True)[0]
            k_space = k_space/max_per_coil.unsqueeze(2).unsqueeze(3)
        elif type == "stand":
            mean = k_space.mean()
            std = k_space.std()
            k_space = (k_space - mean)/ (std + eps)
 
        # Else: no normalization
        return k_space



    @classmethod
    def __perform_fft(cls, k_space):

        transformed = fastmri.ifft2c(k_space)
        return transformed

    @classmethod
    def retrieve_size(self, file):
        et_root = etree.fromstring(file["ismrmrd_header"][()])
        enc = ["encoding", "encodedSpace", "matrixSize"]
        enc_size = (
            int(et_query(et_root, enc + ["x"])),
            int(et_query(et_root, enc + ["y"])),
            int(et_query(et_root, enc + ["z"])),
        )
        rec = ["encoding", "reconSpace", "matrixSize"]
        recon_size = (
            int(et_query(et_root, rec + ["x"])),
            int(et_query(et_root, rec + ["y"])),
            int(et_query(et_root, rec + ["z"])),
        )

        lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
        enc_limits_center = int(et_query(et_root, lims + ["center"]))
        enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1
        padding_left = enc_size[1] // 2 - enc_limits_center
        padding_right = padding_left + enc_limits_max
        return recon_size
    
    @classmethod
    def __load_files(cls, path_or_file, load_only_one_path_idx=None):
        
        """Load's the files or single file
        @path_or_file: Following types are supported, file name or path as string or single path name
        @load_only_one_path_idx: If multiple files or path is provided, this can be set to load single file
        """
        
        if path_or_file.endswith(".h5"):
            # then we can assume that it is single file
            file = h5py.File(path_or_file, 'r')
            data = file['kspace'][()]
            crop_size = cls.retrieve_size(file) 
            file.close()

            cls.file_name = Path(path_or_file)
            return data, crop_size
        else:
            # Then it is path

            # Malformed scans
            fnames_filter = ['file_brain_AXT2_200_2000446.h5',
                        'file_brain_AXT2_201_2010556.h5',
                        'file_brain_AXT2_208_2080135.h5',
                        'file_brain_AXT2_207_2070275.h5',
                        'file_brain_AXT2_208_2080163.h5',
                        'file_brain_AXT2_207_2070549.h5',
                        'file_brain_AXT2_207_2070254.h5',
                        'file_brain_AXT2_202_2020292.h5',
                        ]
            

            path = Path(path_or_file)
            files_paths = sorted(path.glob('*.h5'))
            files_paths = [file for file in files_paths if (file not in fnames_filter)]

            # Assert check that we have loaded files
            assert len(files_paths) > 0, f"No files in the path {path_or_file}"

            if load_only_one_path_idx is not None:
                # Choose a sample number form the files
                file_path = files_paths[load_only_one_path_idx]
                cls.file_name = file_path

                file = h5py.File(file_path.resolve(), 'r')
                crop_size = cls.retrieve_size(file) 
                data = file['kspace'][()]
                file.close()
                return data, crop_size
            else:
                # if we want to load all files
                raise NotImplementedError("Multi path loading is not currently supported yet")


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


# Some Dataset variations including different data

class MRIDatasetWithDistances(MRIDataset):
    def __init__(self, 
                 data_class='brain', 
                 data_root="data",
                 challenge='multicoil', 
                 set="train", 
                 transform=True, 
                 sample=0, slice=0, 
                 full_norm=False, 
                 custom_file_or_path = None,
                 per_coil_stats=True,
                 centercrop=True,
                 normalization="max",
                 cat_coil=True,
                 cat_dists=True
                 ):
        super().__init__(
                 data_class, 
                 data_root,
                 challenge, 
                 set, 
                 transform, 
                 sample, 
                 slice, 
                 full_norm,
                 custom_file_or_path,
                 per_coil_stats,
                 centercrop,
                 normalization,
        )
        self.dist_to_center = torch.sqrt(self.coords[...,1]**2 + self.coords[...,2]**2)
        if cat_dists:
            self.coords = torch.cat((self.coords,self.dist_to_center.unsqueeze(dim=-1)),dim=-1)
        self.cat_coil = cat_coil

    def __getitem__(self, idx):
        if self.cat_coil:
            return self.coords[idx], self.image[idx], self.coords[idx,[0,-1]]
        else: 
            return self.coords[idx], self.image[idx], self.dist_to_center[idx]

class MRIDatasetDistanceAndAngle(MRIDataset):
    def __init__(self, 
                 data_class='brain', 
                 data_root="data",
                 challenge='multicoil', 
                 set="train", 
                 transform=True, 
                 sample=0, slice=0, 
                 full_norm=False, 
                 custom_file_or_path = None,
                 per_coil_stats=True,
                 centercrop=True,
                 normalization="max"
                 ):
        super().__init__(
                 data_class, 
                 data_root,
                 challenge, 
                 set, 
                 transform, 
                 sample, 
                 slice, 
                 full_norm,
                 custom_file_or_path,
                 per_coil_stats,
                 centercrop,
                 normalization
        )
        self.dist_to_center = torch.sqrt(self.coords[...,1]**2 + self.coords[...,2]**2)
        self.angle = torch.arctan(self.coords[...,1]/self.coords[...,2])
        # coil, distance, angle
        self.coords = torch.stack([self.coords[...,0], self.dist_to_center, self.angle], dim=-1)

    def reset_distances(self, part_radii):
        first = part_radii[1]
        self.coords[...,1] = self.coords[...,1] - first
        self.coords[...,1] = self.coords[...,1] / self.coords[...,1].max()
        self.coords[...,2] = self.coords[...,2] / self.coords[...,2].max()



    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        return self.coords[idx], self.image[idx], self.dist_to_center[idx], self.labels[idx]





if __name__ == "__main__":
    x = MRIDataset(transform=False)