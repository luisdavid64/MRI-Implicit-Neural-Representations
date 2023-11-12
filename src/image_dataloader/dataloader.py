import os
from pathlib import Path

import fastmri
import h5py
import torch
from fastmri.data import transforms as T
from torch.utils.data import Dataset


def create_grid(h, w):
    grid_y, grid_x = torch.meshgrid([torch.linspace(0, 1, steps=h),
                                     torch.linspace(0, 1, steps=w)])
    grid = torch.stack([grid_y, grid_x], dim=-1)
    return grid


class H5Dataset(Dataset):
    def __init__(self, data_class='brain', challenge='multicoil', train=False, test=False, transform=False, custom_file_or_path = None):
        # self.batch_size = batch_size
        self.challenge = challenge
        self.transform = transform
        self.data_class = data_class  # brain or knee
        self.train = train
        self.test = test

        # Assert check for train and test boolean values
        assert train is True or test is True, "train or test should be True, they cannot be false in the same time"

        if custom_file_or_path is None or custom_file_or_path == "":
            if self.train:
                self.root = "{}_{}_train/".format(self.data_class, self.challenge)
            elif self.test:
                self.root = "{}_{}_test/".format(self.data_class, self.challenge)
        else:
            self.root = custom_file_or_path

        
        data = self.__load_files(self.root, load_only_one_path_idx=0)

        if self.transform:
            data = self.__perform_fft(data)
        self.X = data

        # for file in files:
        #     data = h5py.File(str(file.resolve()))['kspace'][()]
        #     if self.transform:
        #         data = self.__perform_fft(data)
        #
        #     if x_none:
        #         self.X = data
        #         x_none = False
        #     else:
        #         self.X = torch.cat((self.X, data), dim=0)
    
    @classmethod
    def __load_files(cls, path_or_file, load_only_one_path_idx=None):
        
        """Load's the files or single file
        @path_or_file: Following types are supported, file name or path as string or single path name
        @load_only_one_path_idx: If multiple files or path is provided, this can be set to load single file
        """
        # Let's check if path_or_file is path or file name
        if not path_or_file.endswith(".h5"):
            # then we can assume that it is path
            root_path = Path(path_or_file)
            files = sorted(root_path.glob('*.h5'))

            
            if load_only_one_path_idx is not None:
                # for 1 file
                file = files[load_only_one_path_idx]

                
                file = h5py.File(str(file.resolve()), 'r')
                data = file['kspace'][()]
                file.close()
                return data
            else:
                # If we want to pass all 
                raise NotImplementedError("Multi path loading is not currently supported")
        else:
            # which means that we have only provided file name like "XYZ.h5"

            file = h5py.File(path_or_file, 'r')
            data = file['kspace'][()]
            file.close()
            return data


        
    @classmethod
    def __perform_fft(cls, k_space):

        transformed = fastmri.ifft2c(k_space)
        transformed = fastmri.complex_abs(transformed)
        transformed = fastmri.rss(transformed, dim=1)  # coil dimension

        return transformed

    def __getitem__(self, index):
        grid = create_grid(*self.X.shape)
        # return grid[index], self.X[index]
        return grid, self.X

    def __len__(self):
        return 1  #self.X.shape[0]



