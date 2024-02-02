# Learning Successor Features with Distributed Hebbian Temporal Memory

## Installation
1. Install `htm.core` from the [repository](https://anonymous.4open.science/r/htm_core-DD2D) according to its instructions.
2. Install `requirements.txt`, `python>=3.9` is required.
3. Run `python install -e .` in the root of this repository containing `setup.py`.
4. Setup [wandb](https://wandb.ai/site) logging system.
5. Install [AnimalAI](https://github.com/Kinds-of-Intelligence-CFI/animal-ai)
6. Define environment with the following variables:

```
ANIMALAI_EXE=path/to/animal-ai/exe/file
ANIMALAI_ROOT=path/to/animal-ai/project/root
GRIDWORLD_ROOT=path/to/him-agent/hima/experiments/successor_representations/configs/environment/gridworld/setups
OPENBLAS_NUM_THREADS=1
```

## Running experiments
To run an experiment from the paper: 
1. specify a path to a corresponding config `RUN_CONF=path/to/config.yaml`.
2. run command: `python him-agent/hima/experiments/successor_representations/runners/test_icml.py`

All configs are in the folder `him-agent/hima/experiments/successor_representations/configs/runner/icml24`.
The folder contains configs for the following experiments:
```
## Gridworld experiments with changing fully observable environemnt (Figure 3): 
mdp_gridworld/ 
    gridworld_mpd_dhtm.yaml
    gridworld_mpd_cscg.yaml
    gridworld_mpd_qtable.yaml
    gridworld_mpd_srtable.yaml
    
## Gridworld experiments with changing partially observable environment (Figure 4):
pomdp_gridworld/
    girdworld_dhtm.yaml
    gridworld_cscg.yaml
    gridworld_lstm.yaml
    
## AnimalAI experiment (Figure 5)
animalai_dhtm.yaml
```
To reproduce figures from the paper, run each config at least 5 times with wandb logging enabled.
