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