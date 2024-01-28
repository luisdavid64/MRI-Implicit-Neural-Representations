# Current system path
import os
from pathlib import Path

# Get the current working directory
working_path = Path(os.getcwd()).joinpath(Path("src"))
import sys
sys.path.append(str(working_path))


import unittest
import os
from undersampling.undersampler import Undersampler
import torch

class TestRemoveOddRowsAndCreateGrid(unittest.TestCase):

    def test_grid_based_sampling_legacy(self):
        batch_size = 2
        x_dim = 320
        y_dim = 320
        channels = 2

        images_tensor = torch.rand((batch_size, x_dim, y_dim, channels))

        removed_odd_rows_tensor, coordinate_grid = Undersampler.undersample_grid(images_tensor, 2,2)

        # Assert the shape of the returned tensors
        self.assertEqual(removed_odd_rows_tensor.shape, (batch_size, x_dim // 2, y_dim // 2, channels))
        self.assertEqual(coordinate_grid.shape, (batch_size * (x_dim // 2) * (y_dim//2), 3))

    def test_random_line_legacy(self):
        batch_size = 2
        x_dim = 8
        y_dim = 8
        channels = 2

        images_tensor = torch.rand((batch_size, x_dim, y_dim, channels))
        undersampeld_image, coordinate_grid = Undersampler.undersample_random_line(images_tensor, 0.5)

        self.assertEqual(undersampeld_image.shape[1] * undersampeld_image.shape[2] * channels, coordinate_grid.shape[0] )

    def test_grid_based_sampling(self):

        batch_size = 2
        x_dim = 12 # make sure that it is divisible with grid_x
        y_dim = 12 # make sure that it is divisible with grid_y
        # otherwise test cases will not work properly
        channels = 3

        images_tensor = torch.rand((channels, x_dim, y_dim, batch_size))

        # Let's use grid based undersampling
        undersampler = Undersampler("grid")
        
        # use grid-3*3 undersampling
        grid_x, grid_y = 3,3

        assert x_dim % grid_x == 0 and y_dim % grid_y == 0, "Make sure that dimensions are divisible with grid otherwise tests will not calculate correct dimensions"
        undersampeld_image, grid, grid_mask = undersampler.apply(images_tensor, [grid_x,grid_y])

        # Shape check
        self.assertEqual(
            undersampeld_image.shape[0]*undersampeld_image.shape[1]*undersampeld_image.shape[2], 
            grid.shape[0]
            )

        # Sum of the mask should be number of undersampled points
        self.assertEqual(
            grid_mask.sum()//3, # Divide the sum to dimention
            grid.shape[0] // (grid_x * grid_y) # We are unersampling one point in this grid
        )
    def test_random_line_based_sampling(self):

        batch_size = 3
        x_dim = 64
        y_dim = 64
        channels = 2

        images_tensor = torch.rand((channels, x_dim, y_dim, batch_size))

        # Let's use grid based undersampling
        undersampler = Undersampler("random_line")
        
        # use p value in between [0, 1]
        # Let's use 1 which means it will get the whole image so that we can test it
        p_value = 1.0  
        undersampeld_image, grid, grid_mask = undersampler.apply(images_tensor, [p_value])

        # Shape check
        self.assertEqual(
            undersampeld_image.shape[0]*undersampeld_image.shape[1]*undersampeld_image.shape[2], 
            grid.shape[0]
        )

        # Sum of the mask should be roughly number of undersampled points
        # Since there is randominity of this approach, it is very hard to make assert statement
        # That is why we choose p value as 1
        self.assertEqual(
            grid_mask.sum()//3, # Divide the sum to dimention
            grid.shape[0] # We are unersampling one point in this grid
        )

    def test_radial_based_sampling(self):
        batch_size = 3
        x_dim = 64
        y_dim = 64
        channels = 2

        images_tensor = torch.rand((channels, x_dim, y_dim, batch_size))

        # Let's use grid based undersampling
        undersampler = Undersampler("radial")

        acceleration = 2
        undersampeld_image, grid, grid_mask = undersampler.apply(images_tensor, [acceleration])

        # Shape check
        self.assertEqual(
            undersampeld_image.shape[0]*undersampeld_image.shape[1]*undersampeld_image.shape[2], 
            grid.shape[0]
        )


if __name__ == '__main__':
    unittest.main()