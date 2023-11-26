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

os.environ['KMP_DUPLICATE_LIB_OK']='True'

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
    assert 0 < shape[0] <= data.shape[-3]
    assert 0 < shape[1] <= data.shape[-2]
    w_from = (data.shape[-3] - shape[0]) // 2
    h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to, :]

def extract_smaps(kspace, low_freq_percentage=8):
    """Extract raw sensitivity maps for kspaces

    This function will first select a low frequency region in all the kspaces,
    then Fourier invert it, and finally perform a normalization by the root
    sum-of-square.
    kspace has to be of shape: nslices x ncoils x height x width

    Arguments:
        kspace (torch.Tensor): the kspace whose sensitivity maps you want extracted.
        low_freq_percentage (int): the low frequency region to consider for
            sensitivity maps extraction, given as a percentage of the width of
            the kspace. In fastMRI, it's 8 for an acceleration factor of 4, and
            4 for an acceleration factor of 8. Defaults to 8.

    Returns:
        torch.Tensor: extracted raw sensitivity maps.
    """
    k_shape = torch.tensor(kspace.shape[-2:])
    n_low_freq = torch.tensor(k_shape * low_freq_percentage / 100, dtype=torch.int32)
    center_dimension = torch.tensor(k_shape / 2, dtype=torch.int32)
    low_freq_lower_locations = center_dimension - n_low_freq // 2
    low_freq_upper_locations = center_dimension + n_low_freq // 2
    
    ### Masking strategy
    x_range = torch.arange(0, k_shape[0])
    y_range = torch.arange(0, k_shape[1])
    X_range, Y_range = torch.meshgrid(x_range, y_range)
    X_mask = (X_range <= low_freq_upper_locations[0]) & (X_range >= low_freq_lower_locations[0])
    Y_mask = (Y_range <= low_freq_upper_locations[1]) & (Y_range >= low_freq_lower_locations[1])
    low_freq_mask = torch.transpose(X_mask & Y_mask, 0, 1).unsqueeze(0).unsqueeze(0)
    low_freq_mask = low_freq_mask.expand_as(kspace)
    ###
    
    low_freq_kspace  = kspace * low_freq_mask.to(kspace.dtype)
    
    # Assuming ortho_ifft2d is a function performing 2D Inverse Fourier Transform
    # You might need to replace this with the PyTorch equivalent
    # (e.g., torch.fft.ifft2 or torch.fft.ifftn)
    coil_image_low_freq = fastmri.ifft2(low_freq_kspace)
    
    # no need to norm this since they all have the same norm
    low_freq_rss = torch.norm(coil_image_low_freq, dim=1)
    coil_smap = coil_image_low_freq / low_freq_rss.unsqueeze(1)
    
    # for now, we do not perform background removal based on low_freq_rss
    # could be done with 1D k-means or fixed background_thresh, with torch.where
    
    return coil_smap


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
                 centercrop=(320,320),
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
        data = self.__load_files(self.root, sample)

        # Choose a slice
        data = data[slice]
        data = T.to_tensor(data)
        if self.transform:
            data = self.__perform_fft(data)
            if centercrop:
                data = complex_center_crop(data, centercrop)
            data = normalize_image(data=data, full_norm=full_norm)
        else:
            data = self.__normalize_per_coil(data)
            data = self.__perform_fft(data)
            # Normalize data in image space
            if centercrop:
                data = complex_center_crop(data, centercrop)
            data = normalize_image(data=data, full_norm=full_norm)
            data = fastmri.fft2c(data=data)
            # data_abs = fastmri.complex_abs(data=data).unsqueeze(-1)
            # Attach absoltue values to end
            # data = torch.view_as_complex(data)
            # print(torch.unique(torch.eq(torch.abs(data).unsqueeze(-1),data_abs), return_counts=True))
            # data = torch.cat((data,data_abs), dim=-1)

        display_tensor_stats(data, with_plot=False)
        # display_tensor_stats(data[...,0:2])
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
    def __normalize_per_coil(cls, k_space):
        # mx = fastmri.complex_abs(k_space).max().item()
        # k_space = k_space/mx
        max_per_coil = fastmri.complex_abs(k_space).reshape(k_space.shape[0],-1).max(dim=-1,keepdim=True)[0]
        k_space = k_space/max_per_coil.unsqueeze(2).unsqueeze(3)
        return k_space



    @classmethod
    def __perform_fft(cls, k_space):

        transformed = fastmri.ifft2c(k_space)
        return transformed
    
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
            file.close()

            cls.file_name = Path(path_or_file)
            return data
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
                data = file['kspace'][()]
                file.close()
                return data
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

if __name__ == "__main__":
    x = MRIDataset(transform=False)