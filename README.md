# Learning Neural Implicit Representations of MRI Data


# Data

We use [FastMRI's multi coil dataset](https://fastmri.org/) for our experiments. For easy use with local config files, download the data and symlink it to directory [data](/data/).

# Requirements

```
    conde create --name inr
    conda activate inr
    pip install -r requirements.txt
```

# Running code

To run an experiment create a [configuration file as in here](src/config/local). Some basic configuration files already exist and the basic usage is

```

python src/train.py src/train.py --config path/to/config

```