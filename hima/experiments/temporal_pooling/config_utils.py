#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import wandb
import yaml

from hima.common.config_utils import parse_str


# TODO: split into parse_arg str -> (key_path: list, value: Any) and overwrite
def overwrite_config(config: dict, key_path: str, value: str):
    # accepts everything non-parseable as is, i.e. as a str
    value = parse_str(value)
    key_path = key_path.removeprefix('--')

    # NOTE: to distinguish sweep params from the config params in wandb
    # interface, we introduced a trick - it's allowed to specify sweep param
    # with insignificant additional dots (e.g. `.path..to...key.`)
    # We ignore them here while parsing the hierarchical path stored in the key.

    # ending dots are removed first to guarantee that after split by dots
    # the last item is the actual correct name stored in the config dict slice
    key_path = key_path.rstrip('.')

    # sequentially unfold nested dicts following the path stored in the key
    tokens = key_path.split('.')
    c = config
    for key in tokens[:-1]:
        if not key:
            # ignore empty items introduced with additional dots
            continue

        # the sub-key can be integer - an index in an array
        key = parse_str(key)
        # unfold the next level of the hierarchy
        c = c[key]

    # finally, overwrite the value of the last key in the path
    key = parse_str(tokens[-1])
    c[key] = value


def make_logger(config: dict):
    if not config.get('log', None):
        # not specified or empty
        return None

    # TODO: aggregate all wandb-related args into logger['log']
    logger = wandb.init(project=config['project'], entity=config['entity'], config=config)
    return logger


def compile_config(run_args, config_path_prefix: str = '../configs/', config_extension: str = 'yaml'):
    config_name = run_args[0]
    with open(f'{config_path_prefix}{config_name}.{config_extension}', 'r') as config_io:
        config = yaml.load(config_io, Loader=yaml.Loader)

    for arg in run_args[1:]:
        key_path, value = arg.split('=')
        overwrite_config(config, key_path, value)

    return config
