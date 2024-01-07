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

        removed_odd_rows_tensor, coordinate_grid = Undersampler.undersample_even_rows_and_create_grid(images_tensor)

        # Assert the shape of the returned tensors
        self.assertEqual(removed_odd_rows_tensor.shape, (batch_size, x_dim // 2, y_dim, channels))
        self.assertEqual(coordinate_grid.shape, (batch_size * (x_dim // 2) * y_dim, 3))

        # Assert that odd rows are removed
        for i in range(x_dim // 2):
            self.assertTrue(torch.allclose(images_tensor[:, i * 2, :, :], removed_odd_rows_tensor[:, i, :, :]))

if __name__ == '__main__':
    unittest.main()