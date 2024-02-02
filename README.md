# Learning Successor Features with Distributed Hebbian Temporal Memory

## Quick install

There are two setup guides:

- [quick & short version](#quick-install) is here below.
- [extended version](./install.md).

Before cloning the repository, make sure Git LFS is installed (see [help](./install.md#step-2-install-git-lfs)). Then:

```bash
# create new env with the required packages via conda, then activate it
conda create --name dhtm python=3.9 numpy matplotlib jupyterlab ruamel.yaml tqdm wandb mock imageio seaborn
conda activate dhtm

# install with pip the packages that cannot be installed with conda
pip install hexy prettytable "pytest>=4.6.5"

#  cd to the hima subdirectory in the project root and install hima package
cd <project_root>
pip install -e .
```

## Run examples

Sign up to [wandb](https://wandb.ai/) and get access token in your profile settings to authorize locally further on.

### Run one experiment

``` bash
PLACE HERE THE COMMAND TO RUN ONE EXPERIMENT
```

Do not forget to change `entity` parameter in the corresponding config file to match your [wandb](https://wandb.ai/) login. When wandb asks you to login for the first time, use your access token obtained earlier.

### Run Sweep

Wandb [sweep](https://docs.wandb.ai/guides/sweeps) runs series of experiments with different seeds and parameters.

```bash
PLACE HERE
```

## License

© 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University). All rights reserved.

Licensed under the [AGPLv3](./LICENSE) license.
