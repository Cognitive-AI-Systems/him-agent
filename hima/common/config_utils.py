#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from ast import literal_eval
from pathlib import Path
from typing import Iterable, Any, Optional, Union, NoReturn

# TODO: rename file to `config` and add top-level description comment


# Register config-related conventional constants here. Start them with `_` as non-importable!
from ruamel import yaml

from hima.common.utils import ensure_list

_TYPE_KEY = '_type_'
_TO_BE_INDUCED_VALUE = 'TBI'


TConfigOverrideKV = tuple[list, Any]


def filtered(d: dict, keys_to_remove: Iterable[str], depth: int) -> dict:
    """
    Return a shallow copy of the provided dictionary without the items
    that match `keys_to_remove`.

    The `depth == 1` means filtering `d` itself,
        `depth == 2` — with its dict immediate descendants
        and so on.
    """
    if not isinstance(d, dict) or depth <= 0:
        return d

    return {
        k: filtered(v, keys_to_remove, depth - 1)
        for k, v in d.items()
        if k not in keys_to_remove
    }


# sadly, cannot specify the correct type hint here, which is tuple[dict, Optional[Any], ...]
def extracted(d: dict, *keys: str) -> tuple:
    """
    Return a copy of the dictionary without specified keys and each extracted value
    (or None if a specified key was absent).

    Examples
    --------
    >>> extracted({'a': 1, 'b': 2, 'c': 3}, 'a', 'c')
    ({'b': 2}, 1, 3)
    """
    values = tuple([d.get(k, None) for k in keys])
    filtered_dict = filtered(d, keys, depth=1)
    return (filtered_dict, ) + values


def extracted_type(config: dict) -> tuple[dict, Optional[str]]:
    return extracted(config, _TYPE_KEY)


def override_config(
        config: dict,
        overrides: Union[TConfigOverrideKV, list[TConfigOverrideKV]]
) -> None:
    overrides = ensure_list(overrides)
    for key_path, value in overrides:
        c = config
        for key_token in key_path[:-1]:
            c = c[key_token]
        c[key_path[-1]] = value


def parse_arg(arg: Union[str, tuple[str, Any]]) -> TConfigOverrideKV:
    if isinstance(arg, str):
        # "--key=value" --> ["--key", "value"]
        key_path, value = arg.split('=', maxsplit=1)

        # "--key" --> "key"
        key_path = key_path.removeprefix('--')
    else:
        # tuple from wandb config
        key_path, value = arg

    # We parse key tokens as they can represent array indices
    # We skip empty key tokens (see [1] in the end of the file for an explanation)
    key_path = [
        parse_str(key_token)
        for key_token in key_path.split('.')
        if key_token
    ]
    value = parse_str(value)

    return key_path, value


def parse_str(s: str) -> Any:
    """Parse string value to the most appropriate type."""

    # noinspection PyShadowingNames
    def boolify(s):
        if s in ['True', 'true']:
            return True
        if s in ['False', 'false']:
            return False
        raise ValueError('Not Boolean Value!')

    # NB: try/except is widely accepted pythonic way to parse things

    # NB: order here is important
    for caster in (boolify, int, float, literal_eval):
        try:
            return caster(s)
        except ValueError:
            pass
    return s


def read_config(filepath: str):
    filepath = Path(filepath)
    with filepath.open('r') as config_io:
        return yaml.load(config_io, Loader=yaml.Loader)


# [1]: Using sweeps we have a little problem with config logging. All parameters
# provided to a run from sweep are logged to wandb automatically. At the same time, when
# we also log our compiled config dictionary, its content is flattened such that
# each param key is represented as `path.to.nested.dict.key`. Note that we declare
# params in sweep config the same way. Therefore, each sweep run will have such params
# duplicated in wandb and there's no correct way to distinguish them. However, wandb
# does it! Also, only sweep runs will have params duplicated. Simple runs don't have
# the second entry because they don't have sweep param args.
#
# Problem: when you want to filter or group by param in wandb interface,
# you cannot be sure which of the duplicated entries to choose, while they're different
# — the only entry that is presented in all runs [either sweep or simple] is the entry
# from our config, not from a sweep.
#
# Solution: That's why we introduced a trick - it's allowed to specify sweep param
# with insignificant additional dots (e.g. `path..to...key.`) to de-duplicate entries.
# We ignore these dots [or empty path elements introduced by them after split-by-dots]
# while parsing the nested key path.
