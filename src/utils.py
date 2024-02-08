
import argparse
from skimage.metrics import structural_similarity
import numpy as np
import torch
from PIL import Image

def set_default_configs(config):
    """

    This method sets missing configurations.

    """
    if "per_coil" not in config:
        config["per_coil"] = False
    if "use_tv" not in config:
        config["use_tv"] = False
    if "regularization" not in config:
        config["regularization"] = dict()
        config["regularization"]["type"] = "none" 
    if "undersampling" not in config:
        config["undersampling"] = None

    return config

def ssim(x, xhat):
    """

    Calculate SSIM of two files. Note: Due to img compression
    Results may be different, so better to avoid 

    """

    if torch.is_tensor(x):
        x = x.numpy()
    if torch.is_tensor(xhat):
        xhat = xhat.numpy()
    x = x / x.max()
    xhat = xhat / xhat.max()
    data_range = np.maximum(x.max(), xhat.max()) - np.minimum(x.min(), xhat.min())
    return structural_similarity(x,xhat, data_range=data_range)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('gt', type=str, help='Path to gt file.')
    parser.add_argument('pred', type=str, help="Path to pred")
    args = parser.parse_args()
    # Calculate SSIM
    first = Image.open(args.gt).convert('L')  # 'L' mode for grayscale
    second = Image.open(args.pred).convert('L')

    first = np.array(first) / 255
    second = np.array(second) /255
    print(second.shape)
    s = ssim(first, second)

    print("SSIM: {:.4}".format(s))
