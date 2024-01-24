
from typing import Tuple
import torch
import math
import numpy as np

GOLDEN_RATIO = (1 + np.sqrt(5)) / 2

def center_crop(data, shape):
    """
    Apply a center crop to the input real image or batch of real images.

    Args:
        data (torch.Tensor): The input tensor to be center cropped. It should have at
            least 2 dimensions and the cropping is applied along the last two dimensions.
        shape (int, int): The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        torch.Tensor: The center cropped image
    """
    assert 0 < shape[0] <= data.shape[-2]
    assert 0 < shape[1] <= data.shape[-1]
    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to]

def get_square_ordered_idxs(square_side_size: int, square_id: int) -> Tuple[Tuple, ...]:
    """Returns ordered (clockwise) indices of a sub-square of a square matrix.

    Parameters
    ----------
    square_side_size: int
        Square side size. Dim of array.
    square_id: int
        Number of sub-square. Can be 0, ..., square_side_size // 2.

    Returns
    -------
    ordered_idxs: List of tuples.
        Indices of each point that belongs to the square_id-th sub-square
        starting from top-left point clockwise.
    """
    assert square_id in range(square_side_size // 2)

    ordered_idxs = list()

    for col in range(square_id, square_side_size - square_id):
        ordered_idxs.append((square_id, col))

    for row in range(square_id + 1, square_side_size - (square_id + 1)):
        ordered_idxs.append((row, square_side_size - (square_id + 1)))

    for col in range(square_side_size - (square_id + 1), square_id, -1):
        ordered_idxs.append((square_side_size - (square_id + 1), col))

    for row in range(square_side_size - (square_id + 1), square_id, -1):
        ordered_idxs.append((row, square_id))

    return tuple(ordered_idxs)

class Undersampler():
    
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def undersample_grid(images_tensor: torch.Tensor, grid_x: int = 3, grid_y: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
        assert images_tensor.dim() == 4, "For processing, please provide a 4-dimensional tensor as [batch_size, image_x, image_y, channel_n]"
        C,H,W,S = images_tensor.size()

        # Remove odd rows from the input tensor
        removed_odd_rows_tensor = images_tensor[:, ::grid_x, ::grid_y, :]

        # Create a coordinate grid based on the new dimensions
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
    
    def undersample_random_line(images_tensor: torch.Tensor, p: float) -> Tuple[torch.Tensor, torch.Tensor]:
        assert images_tensor.dim() == 4, "For processing, please provide a 4-dimensional tensor as [batch_size, image_x, image_y, channel_n]"
        C,H,W,S = images_tensor.size()


        mask_x = torch.rand(H) < math.sqrt(p)
        mask_y = torch.rand(W) < math.sqrt(p)


        # Remove odd rows from the input tensor
        removed_odd_rows_tensor = images_tensor[:, mask_x,:, :]
        removed_odd_rows_tensor = removed_odd_rows_tensor[:, :, mask_y, :]

        # Apply same mask to linspace
        Z, Y, X = torch.meshgrid(torch.linspace(-1, 1, C),
                                torch.linspace(-1, 1, H)[mask_x],
                                torch.linspace(-1, 1, W)[mask_y])

        # Reshape and stack the grids
        grid = torch.hstack((Z.reshape(-1, 1),
                            Y.reshape(-1, 1),
                            X.reshape(-1, 1)))

        

        return removed_odd_rows_tensor, grid


    @staticmethod
    def undersample_radial(images_tensor: torch.Tensor, acceleration) -> Tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.RandomState()
        assert images_tensor.dim() == 4, "For processing, please provide a 4-dimensional tensor as [batch_size, image_x, image_y, channel_n]"
        C,H,W,S = images_tensor.size()
        shape = images_tensor.shape
        max_dim = max(shape[1:3]) - max(shape[1:3]) % 2
        min_dim = min(shape[1:3]) - min(shape[1:3]) % 2
        num_nested_squares = max_dim // 2
        M = int(np.prod(shape[1:3]) / (acceleration * (max_dim / 2 - (max_dim - min_dim) * (1 + min_dim / max_dim) / 4)))
        mask = np.zeros((max_dim, max_dim), dtype=np.float32)

        t = rng.randint(low=0, high=1e4, size=1, dtype=int).item()

        for square_id in range(num_nested_squares):
            ordered_indices = get_square_ordered_idxs(
                square_side_size=max_dim,
                square_id=square_id,
            )
            # J: size of the square, J=2,…,N, i.e., the number of points along one side of the square
            J = 2 * (num_nested_squares - square_id)
            # K: total number of points along the perimeter of the square K=4·J-4;
            K = 4 * (J - 1)

            for m in range(M):
                indices_idx = int(np.floor(np.mod((m + t * M) / GOLDEN_RATIO, 1) * K))
                mask[ordered_indices[indices_idx]] = 1.0

        pad = ((shape[1] % 2, 0), (shape[2] % 2, 0))

        mask = np.pad(mask, pad, constant_values=0)
        mask = center_crop(torch.from_numpy(mask.astype(bool)), shape[1:3])
        mask = ~mask
        Z, Y, X = torch.meshgrid(torch.linspace(-1, 1, C),
                                torch.linspace(-1, 1, H),
                                torch.linspace(-1, 1, W))
        Z=Z[:,mask]
        Y=Y[:,mask]
        X=X[:,mask]
        # Reshape and stack the grids
        grid = torch.hstack((Z.reshape(-1, 1),
                            Y.reshape(-1, 1),
                            X.reshape(-1, 1)))
        removed_radial_tensor = images_tensor
        removed_radial_tensor[:,mask,:] = 0
        # removed_radial_tensor = images_tensor[:, mask, :] 

        return removed_radial_tensor, grid



if __name__ == '__main__':
    # Example usage
    batch_size = 2
    x_dim = 8
    y_dim = 8
    channels = 2

    images_tensor = torch.rand((batch_size, x_dim, y_dim, channels))
    undersampeld_image, coordinate_grid = Undersampler.undersample_random_line(images_tensor, 0.5)
    #undersampeld_image, coordinate_grid = Undersampler.undersample_grid(images_tensor)
