"""

Calculate SSIM of two files

"""
import argparse
import cv2 as cv
from skimage.metrics import structural_similarity
import numpy as np
import torch

def ssim(x, xhat):
    if torch.is_tensor(x):
        x = x.numpy()
    if torch.is_tensor(xhat):
        xhat = xhat.numpy()
    data_range = np.maximum(x.max(), xhat.max()) - np.minimum(x.min(), xhat.min())
    return structural_similarity(x,xhat, data_range=data_range)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('gt', type=str, help='Path to gt file.')
    parser.add_argument('pred', type=str, help="Path to pred")
    args = parser.parse_args()
    # Calculate SSIM
    first = cv.imread(args.gt)

    second = cv.imread(args.pred)

    first = cv.resize(first, (2576,1125))
    second = cv.resize(second, (2576,1125))
    first = cv.cvtColor(first, cv.COLOR_BGR2GRAY)
    second = cv.cvtColor(second, cv.COLOR_BGR2GRAY)
    s = ssim(first, second)

    print("SSIM: {:.4}".format(s))
