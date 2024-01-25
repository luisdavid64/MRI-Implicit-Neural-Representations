import numpy as np
from typing import Tuple
import torch

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

def verify_acc_factor(mask):
    acc = torch.numel(mask)/torch.count_nonzero(mask)
    return str(acc.item())