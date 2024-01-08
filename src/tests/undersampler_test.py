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

    def test_remove_odd_rows_and_create_grid(self):
        batch_size = 2
        x_dim = 320
        y_dim = 320
        channels = 2

        images_tensor = torch.rand((batch_size, x_dim, y_dim, channels))

        removed_odd_rows_tensor, coordinate_grid = Undersampler.undersample_grid(images_tensor, 2,2)

        # Assert the shape of the returned tensors
        self.assertEqual(removed_odd_rows_tensor.shape, (batch_size, x_dim // 2, y_dim // 2, channels))
        self.assertEqual(coordinate_grid.shape, (batch_size * (x_dim // 2) * (y_dim//2), 3))

        

if __name__ == '__main__':
    unittest.main()