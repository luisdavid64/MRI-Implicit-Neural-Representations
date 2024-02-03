import torch
from torch.utils.data import Dataset
import fastmri
import h5py
from pathlib import Path
from fastmri.data import transforms as T
from tabulate import tabulate
import xml.etree.ElementTree as etree
from undersampling.undersampler import Undersampler
from typing import Tuple
from .utils import *

class MRIDataset(Dataset):
    """
    Dataset for INR from a fastmri sample.
    
    Args:
    - data_class: [brain, knee] which type of sample to use
    - data_root: path to dataset
    - challenge: currently multicoil only
    - set: train or test set of FastMRI challenge
    - transform: if true, get MRI image, else return kspace
    - sample: sample no to retrieve
    - slice: slice no to retrieve
    - custom_file_or_path: alternative retrieval with direct path to a file
    - per_coil_stats: prints stats of MRI image per-coil
    - centercrop: whether to perform a center crop
    - normalization: type of normalization to use

    """
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

        # Transform in our context means apply FFT
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
            title = "{} Data Statistics Per Coil for sample {} at slice {}".format("Image" if transform else "K-space",
                                                                                   sample, slice)
            print("{}\n{}".format(title,table))

        # Flatten image and grid
        self.flatten_image_and_create_coords(data)

    
    def flatten_image_and_create_coords(self, data : torch.Tensor):
        # It will take data and reshape it for flatten image and it will create cordinates for it
        C,H,W,S = data.shape
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
        return self.coords[idx], self.image[idx], list(), list()

    def __len__(self):
        return len(self.image)  #self.X.shape[0]

class MRIDatasetUndersampling(MRIDataset):
    """
        This Dataset enhances MRIDataset with undersampling
    """
    def __init__(self, data_class='brain', data_root="data", challenge='multicoil', set="train", transform=True, sample=0, slice=0, full_norm=False, custom_file_or_path=None, per_coil_stats=True, centercrop=True, normalization="max", undersampling=None):
        # Initialize undersampling attributes
        self.undersampling_argument, self.undersampling_params = self.parse_undersampling_argument(undersampling)

        super().__init__(data_class, data_root, challenge, set, transform, sample, slice, full_norm, custom_file_or_path, per_coil_stats, centercrop, normalization)


    
    def parse_undersampling_argument(self, arg : str) -> Tuple[str, list]:
        """
        This method takes an argument in the 'function-params' type format and parses it to set a list.
    
        Args:
            argument (str): The input argument in the format 'function-params', e.g., "grid-3*3", "radial-2", "random_line-0.5".
    
        Returns:
            list: A parsed list based on the input argument.
        """
        
        param_parsed = list()
        # But it also supports none
        if arg == None or arg.lower() == "none":
            # Return none, and emptly list
            return arg, param_parsed
        
        parts = arg.split("-")
        assert len(parts) == 2, f"Argument {arg} is incorrect"
        argument_type, param = parts[0], parts[1]
        

        
        if argument_type == "grid":
            # Safety checks
            assert "*" in param, "Please use * symbol for stating grid size"
            
            # Get dimensions
            dimensions = param.split("*")

            assert len(dimensions)==2, f"Grid dimensions provided ({param}) for undersampling is wrong please provide x*y format"

            param_parsed.append(int(dimensions[0]))
            param_parsed.append(int(dimensions[1]))

            # str as argument type, parameters list
            return argument_type, param_parsed
        elif argument_type == "random_line":
            # Here we are assuming param is a float value in between 0 to 1
            random_value_p = float(param)
            assert (random_value_p <= 1.0) and (random_value_p >= 0), "P value is not in range [0,1]"

            # Add param value to param_parsed list
            param_parsed.append(random_value_p)   

            # Return argument types as str, then p value in the list
            return argument_type, param_parsed
        elif argument_type == "radial":
            # Here we are assuming param is a float value in between 0 to 1
            random_value_p = float(param)
            # Add param value to param_parsed list
            param_parsed.append(random_value_p)   

            # Return argument types as str, then p value in the list
            return argument_type, param_parsed
        else:
            raise ValueError(f"Argument {argument_type} is not supported")

    def flatten_image_and_create_coords(self, data : torch.Tensor):
        C,H,W,S = data.shape
        if self.undersampling_argument == None or self.undersampling_argument.lower() == "none":
            self.image = data.reshape((C*H*W),S) # Dim: (C*H*W,1), flattened 2d image with coil dim
            self.coords = create_coords(C,H,W) # Dim: (C*H*W,3), flattened 2d coords with coil dim
            
            # Then we do not need to continue exit from here
            return

        # if we want undersampling we need to create Undersampler with its constructor
        self.undersampler = Undersampler(self.undersampling_argument)
        
        # call the undersampler withy apply and provide the params list and data
        # this return
        #   mask applied/undersampled data, normal cordinates, mask for cordinates
        data_undersampled, coords, coords_mask = self.undersampler.apply(data, self.undersampling_params)

        self.image = data_undersampled.reshape((C*H*W),S) # Dim: (C*H*W,1)
        self.shape = data_undersampled.shape
        self.coords = coords # Dim: (C*H*W,3)
        self.coords_mask = coords_mask # Dim: (C*H*W,3) undersampling points

    def __len__(self):
        return len(self.coords)
    
    # Here we need to override __getitem__
    # This returns: 
        # cordinates, image, cordinate mask occording to undersampling
    def __getitem__(self, idx):
        
        return self.coords[idx], self.image[idx], list(), self.coords_mask[idx]


# Some Dataset variations including different data

class MRIDatasetWithDistances(MRIDatasetUndersampling):
    """
        This Dataset enhances MRIDataset with distance from origin information 
        at each coordinate point 
    """
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
                 cat_coil=False,
                 cat_dists=False,
                 undersampling = None
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
                 undersampling
        )
        self.dist_to_center = torch.sqrt(self.coords[...,1]**2 + self.coords[...,2]**2)
        if cat_dists:
            self.coords = torch.cat((self.coords,self.dist_to_center.unsqueeze(dim=-1)),dim=-1)
        self.cat_coil = cat_coil

    def __getitem__(self, idx):
        if self.cat_coil:
            return self.coords[idx], self.image[idx], self.coords[idx,[0,-1]], list()
            # return self.coords[idx,[1,2]], self.image[idx], self.coords[idx,[0,-1]]
        else: 
            return self.coords[idx], self.image[idx], self.dist_to_center[idx], list()

class MRICoilWrapperDataset(Dataset):
    """
        A wrapper for our MRIDataset that samples data one coil at a time
        instead of per pixel. This allows us to compute regularization on
        undersampled pixels, such as Total Variation
    """

    def __init__(self, 
                 dataset,
                 undersampling=True,
                 coord_size=3
                 ):
        self.dataset = dataset
        self.coord_size = coord_size
        # Set the length equal to #coils
        C,H,W,S = self.dataset.shape
        self.len = self.dataset.shape[0]
        self.img_shape = self.dataset.img_shape
        self.file = self.dataset.file
        self.shape = self.dataset.shape
        self.image = self.dataset.image.reshape((C,H,W,S))
        self.coords = self.dataset.coords.reshape((C,H,W,self.coord_size))
        if hasattr(self.dataset, 'dist_to_center'):
            self.dist_to_center = self.dataset.dist_to_center.reshape((C,H,W,1))
        if hasattr(self.dataset, 'coords_mask'):
            self.coords_mask = self.dataset.coords_mask.reshape((C,H,W,self.coord_size))
        self.undersampling = undersampling
    
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img = self.image[idx].reshape(-1, 2)
        coords = self.coords[idx].reshape(-1,self.coord_size)
        if type(self.dataset) is MRIDatasetWithDistances:
            dists = self.dist_to_center[idx].reshape(-1,1)
            if self.undersampling != None:
                mask = self.coords_mask[idx].reshape(-1,self.coord_size)
                return coords, img, dists, mask
            return coords, img, dists, list()
        elif type(self.dataset) is MRIDatasetUndersampling:
            mask = self.coords_mask[idx].reshape(-1,self.coord_size)
            return coords, img, list(), mask
        else:
            return coords, img, list(), list()