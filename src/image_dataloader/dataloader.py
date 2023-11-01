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
    def __init__(self, data_class='brain', challenge='multicoil', train=False, test=False, transform=False):
        # self.batch_size = batch_size
        self.challenge = challenge
        self.transform = transform
        self.data_class = data_class  # brain or knee
        self.train = train
        self.test = test
        if self.train:
            self.root = "{}_{}_train/".format(self.data_class, self.challenge)
        elif self.test:
            self.root = "{}_{}_test/".format(self.data_class, self.challenge)

        # self.root = (self.data_class, root)
        path = Path(self.root)
        files = sorted(path.glob('*.h5'))

        # self.X = None
        # x_none = True

        # for 1 file.
        file = files[0]

        data = h5py.File(str(file.resolve()))['kspace'][()]
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



