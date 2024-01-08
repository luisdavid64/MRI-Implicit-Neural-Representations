
from typing import Tuple
import torch


class Undersampler():
    
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def undersample_grid(images_tensor: torch.Tensor, grid_x: int = 3, grid_y: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
        assert images_tensor.dim() == 4, "For processing, please provide a 4-dimensional tensor as [batch_size, image_x, image_y, channel_n]"
        C,H,W,S = images_tensor.size()

        # Remove odd rows from the input tensor
        removed_odd_rows_tensor = images_tensor[:, ::grid_x, ::grid_y, :]
        #removed_odd_rows_tensor = images_tensor[:, :, :, :]

        # Create a coordinate grid based on the new dimensions after removing odd rows
        new_H = removed_odd_rows_tensor.shape[1]
        new_W = removed_odd_rows_tensor.shape[2]

        Z, Y, X = torch.meshgrid(torch.linspace(-1, 1, C),
                                torch.linspace(-1, 1, new_H),
                                torch.linspace(-1, 1, new_W))

        # Reshape and stack the grids
        grid = torch.hstack((Z.reshape(-1, 1),
                            Y.reshape(-1, 1),
                            X.reshape(-1, 1)))

        

        return removed_odd_rows_tensor, grid

if __name__ == '__main__':
    # Example usage
    batch_size = 1
    x_dim = 8
    y_dim = 8
    channels = 1

    images_tensor = torch.rand((batch_size, x_dim, y_dim, channels))

    # Remove odd rows from the tensor and create a corresponding grid
    removed_odd_rows_tensor, coordinate_grid = Undersampler.undersample_grid(images_tensor)

    # Print the original tensor, removed tensor, and coordinate grid for verification
    print("Original Tensor:")
    print(images_tensor.shape)
    print("\nTensor with Odd Rows Removed:")
    print(removed_odd_rows_tensor.shape)
    print("\nCoordinate Grid:")
    print(coordinate_grid.shape)