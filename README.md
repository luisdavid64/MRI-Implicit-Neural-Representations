# Learning Neural Implicit Representations of MRI Data


# Data

We use [FastMRI's multi coil dataset](https://fastmri.org/) for our experiments. For easy use with local config files, download the data and symlink it to ./data.

# Requirements

```
    conde create --name inr
    conda activate inr
    pip install -r requirements.txt
```

# Running code

To run an experiment create a [configuration file as in here](src/config/local). Some basic configuration files already exist and the basic usage is

```
python src/train.py --config path/to/config
```

# Running Multi-Scale Training

To train a multiscale network on k-space data, run the following command:

```
python src/train_kspace_multiscale.py --config path/to/config
```

Note that the clustering setup must be included in the config as in [this file](./src/config/local/config_fourier_multiscale.yaml)

# Config Parameters

The configuration file is the main entry point to modify the behavior of the INR during training.

## Model Choices:

* Single-Scale: `["SIREN", "WIRE", "WIRE2D", "FFN", "FOURIER", "GABOR"]`

* Multi-Scale: `["Fourier", "BoundedFourier"]`

## Loss Choices: 

Available Criteria: `["L2", "MSLE", "FFL", "L1", "HDR", "tanh"]`

In Particular, HDR loss is presented in [1] and Tanh loss in [2]



## Clustering

Here we provide an example on how to set up the ring clustering
```
    partition:
      no_steps: 40 # Initial number of rings
      no_models: 4 # Final number of rings after clustering

```

## Undersampling

Here we provide an example on how to set up the undersampling param
```
# Supported formats:
#   grid-x*y          example: grid-3*3
#   random_line-p     example: random_line-0.5 
#   radial-acc_fac    example: radial-4
#   none              No undersamping
undersampling: grid-2*1

```


# References 

[1] Huang, W., Li, H., Cruz, G., Pan, J., Rueckert, D., & Hammernik, K. (2022). Neural Implicit k-Space for Binning-free Non-Cartesian Cardiac MR Imaging. Information Processing in Medical Imaging.

[2] Chen, X., Liu, Y., Zhang, Z., Qiao, Y., & Dong, C. (2021). HDRUNet: Single Image HDR Reconstruction with Denoising and Dequantization. 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 354-363.