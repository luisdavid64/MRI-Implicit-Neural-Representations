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

# Calculate SSIM
first = cv.imread("outputs/config_fourier_multiscale/brain/img_Fourier_512_512_8_LSL_lr0.0003_encoder_gauss_scale4_size2562024-01-07_20-31-37/images/train.png")

#second = cv.imread("captured.png")
second = cv.imread("outputs/config_fourier_multiscale/brain/img_Fourier_512_512_8_LSL_lr0.0003_encoder_gauss_scale4_size2562024-01-07_20-31-37/images/recon_250_36.69.png")

first = cv.resize(first, (2576,1125))
second = cv.resize(second, (2576,1125))
first = cv.cvtColor(first, cv.COLOR_BGR2GRAY)
second = cv.cvtColor(second, cv.COLOR_BGR2GRAY)
s = ssim(first, second)

print(s)