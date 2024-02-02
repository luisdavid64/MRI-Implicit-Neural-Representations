import torch
from math import ceil
from torch.distributions import Normal
import xml.etree.ElementTree as etree
from typing import (
    Sequence,
)
import fastmri
from matplotlib import pyplot as plt

def gaussian_kernel_1d(sigma: float, num_sigmas: float = 10.) -> torch.Tensor:
    
    radius = ceil(num_sigmas * sigma)
    support = torch.arange(-radius, radius + 1, dtype=torch.float)
    kernel = Normal(loc=0, scale=sigma).log_prob(support).exp_()
    # Ensure kernel weights sum to 1, so that image brightness is not altered
    return kernel.mul_(1 / kernel.sum())

def gaussian_filter_2d(img: torch.Tensor, sigma: float) -> torch.Tensor:
    
    kernel_1d = gaussian_kernel_1d(sigma)  # Create 1D Gaussian kernel
    
    padding = len(kernel_1d) // 2  # Ensure that image size does not change
    img = img  # Need 4D data for ``conv2d()``
    # Convolve along columns and rows
    img = torch.nn.functional.conv2d(img, weight=kernel_1d.view(1, 1, -1, 1), padding=(padding, 0))
    img = torch.nn.functional.conv2d(img, weight=kernel_1d.view(1, 1, 1, -1), padding=(0, padding))
    return img  # Make 2D again

def et_query(
    root: etree.Element,
    qlist: Sequence[str],
    namespace: str = "http://www.ismrm.org/ISMRMRD",
) -> str:
    """
    ElementTree query function.

    This can be used to query an xml document via ElementTree. It uses qlist
    for nested queries.

    Args:
        root: Root of the xml to search through.
        qlist: A list of strings for nested searches, e.g. ["Encoding",
            "matrixSize"]
        namespace: Optional; xml namespace to prepend query.

    Returns:
        The retrieved data as a string.
    """
    s = "."
    prefix = "ismrmrd_namespace"

    ns = {prefix: namespace}

    for el in qlist:
        s = s + f"//{prefix}:{el}"

    value = root.find(s, ns)
    if value is None:
        raise RuntimeError("Element not found")

    return str(value.text)


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
    # Make sure crop fits one dimension at least
    if data.shape[-2] < shape[1]:
        shape = (data.shape[-2], data.shape[-2])
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
    """
        This method pretty prints some statistics about a tensor
    """
    if with_plot:
        plt.boxplot(torch.view_as_complex(tensor).abs().reshape(-1), vert=False)  # vert=False makes it a horizontal boxplot
        plt.title('Plot of K-space')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        # Show the plot
        plt.show()
    shape, vmin, vmax, vmean, vstd = tensor.shape, tensor.min(), tensor.max(), torch.mean(tensor), torch.std(tensor)
    print('shape:{} | min:{:.5f} | max:{:.5f} | mean:{:.5f} | std:{:.5f}'.format(shape, vmin, vmax, vmean, vstd))