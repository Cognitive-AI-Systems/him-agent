# Installation

- [Installation](#installation)
  - [Install requirements](#install-requirements)
    - [\[Optional\] Step 1. Install conda](#optional-step-1-install-conda)
    - [Step 2. Install Git LFS](#step-2-install-git-lfs)
      - [Comment: working with data files](#comment-working-with-data-files)
    - [Steps 3. Install requirements to dedicated python environment](#steps-3-install-requirements-to-dedicated-python-environment)
      - [Option 1: use conda and pip](#option-1-use-conda-and-pip)
      - [Option 2: use pip only](#option-2-use-pip-only)
    - [Step 3. Install `htm.core`](#step-3-install-htmcore)
    - [\[Optional\] Step 4. Install PyRep](#optional-step-4-install-pyrep)
    - [\[Optional\] Step 5. Install AnimalAI 3](#optional-step-5-install-animalai-3)
    - [Step 6. Install `hima` package](#step-6-install-hima-package)
    - [\[Optional\] Step 7. Install Pinball](#optional-step-7-install-pinball)
    - [Project structure](#project-structure)

## Install requirements

### [Optional] Step 1. Install conda

Using conda as an environment and package manager is optional. Using pip as a package manager and  _pipenv_/_virtualenv_/etc. as Python environment managers is a valid option too.

However, we recommend to prioritize using conda over pip, because it allows managing not only python packages, as pip does, and also resolves package dependencies smarter, ensuring that the environment has no version conflicts (see [understanding pip and conda](https://www.anaconda.com/blog/understanding-conda-and-pip) for details). Having said that, we recommend trying to install as much packages as you can with conda first, and use pip only as the last resort (sadly, not every requirement is available in conda channels).

If you are new to conda, we recommend getting conda with [Miniconda](https://docs.conda.io/en/latest/miniconda.html) instead of Anaconda, because it's just a minimal console version of the conda tool (see [the difference](https://stackoverflow.com/a/45421527/1094048)).

_If you feel confident, you may also consider [Miniforge](https://github.com/conda-forge/miniforge) as an alternative to Miniconda — the difference is only that it sets the default conda package channel to [conda-forge](https://github.com/conda-forge/miniforge), which is community-driven, so it gets updates faster in general and contains more packages. Another alternative - now to conda itself - is to use [Mamba](https://github.com/mamba-org/mamba). Mamba mimics conda API, even uses it and fully backward compatible with it, so you can use both together; Mamba has enhanced dependency-resolving performance and async package installation, and also extends conda functionality. Last two options combined, [Mamba + Miniforge = Mambaforge](https://github.com/conda-forge/miniforge#mambaforge), are distributed by Miniforge community._

### Step 2. Install Git LFS

To setup Git LFS you need to [install](https://github.com/git-lfs/git-lfs#downloading) `git-lfs` first:

```bash
# Linux:
apt install git-lfs

# Mac:
brew install git-lfs
```

Then "link" git and git-lfs together by:

```bash
# system-wise
git lfs install --system

# or only for the repository you're in
git lfs install
```

After this you just work as usual.

_Check out `.gitattributes` for the project's git-lfs rules. To print all tracked files use `git lfs ls-files`._

#### Comment: working with data files

Storing binary or any other "data" files in a repository can quickly bloat its size. They rarely change, and if so, mostly as a whole file (i.e they are added or deleted, not edited). But git still treats them as text files and tries to track their content, putting it to the index.

If only was a way to track some signature of these files, store their content somewhere else and pull them only on demand! Fortunately, that's exactly what Git[hub] LFS do :) It stands for Large File Storage. Here's the quote from Git LFS site:

> Git Large File Storage (LFS) replaces large files such as audio samples, videos, datasets, and graphics with text pointers inside Git, while storing the file contents on a remote server like GitHub.com or GitHub Enterprise.

### Steps 3. Install requirements to dedicated python environment

File `requirements.txt` contains all required dependencies. Also, as commented in `requirements.txt`, you will need to install four additional dependencies:

1. Our fork of the `htm.core` package.
2. [optional] AnimalAI
3. [optional] Pinball

In this section we provide you with two options: a) install requirements using both conda and pip or b) install everything with pip. In both cases everything is installed into the dedicated conda environment. The name of the environment is up to you, while we use `hima` for this guide.

#### Option 1: use conda and pip

Create new conda environment with all requirements that can be installed with conda; then install what's left with pip:

```bash
conda create --name hima python=3.9 numpy matplotlib jupyterlab ruamel.yaml tqdm wandb mock imageio seaborn pygraphviz colormap prettytable pytest
conda activate hima
pip install hexy
```

#### Option 2: use pip only

Create new conda environment and install everything with pip:

```bash
conda create --name hima python=3.9
conda activate hima
pip install -r requirements.txt
```

### Step 3. Install `htm.core`

Not needed

### [Optional] Step 4. Install PyRep

If you are planning to run experiments with arm manipulators, you should also install [PyRep](https://github.com/stepjam/PyRep).

### [Optional] Step 5. Install AnimalAI 3

If you are planning to run experiments in [AnimalAI](https://github.com/mdcrosby/animal-ai) environment, you should also install it.

### Step 6. Install `hima` package

Install our `hima` package:

```bash
cd <hima_project_root>
pip install -e .
```

_The last command installs `hima` with the "development mode" flag. It allows you importing modules from the `hima` package outside of the package root folder, e.g. in Jupyter Notebooks or in another projects. Development mode flag `-e` prevents the package sources to be copied to the package index. Instead, the symlink is created, which means that the edits are "visible" immediately and don't require you to reinstall/update the package manually after any changes in its sources._

### [Optional] Step 7. Install Pinball

If you are planning to run experiments in [Pinball](https://github.com/ZhekaHauska/pinball) environment, you should also install it.
