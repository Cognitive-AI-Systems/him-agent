# -----------------------------------------------------------------------------------------------
# © 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research Institute" (AIRI);
# Moscow Institute of Physics and Technology (National Research University). All rights reserved.
# 
# Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
# -----------------------------------------------------------------------------------------------

from typing import List, Any, Sequence, Tuple

import numpy as np

from htm_rl.common.sdr import SparseSdr
from htm_rl.common.utils import isnone


class IntBucketEncoder:
    """
    Encodes integer values from the range [0, `n_values`) as SDR with `output_sdr_size` total bits.
    SDR bit space is divided into `n_values` possibly overlapping buckets of `bucket_size` bits
        where i-th bucket starts from i * `buckets_step` index.
    Each value is encoded with corresponding bucket of active bits (see example).
    This's a sparse encoder, so the output is in sparse SDR format.

    Non-overlapping example, i.e. when `buckets_step` == `bucket_size` = 4, for `n_values` = 3:
        0000 1111 0000
    """

    ALL = -1

    n_values: int
    output_sdr_size: int

    _bucket_size: int
    _buckets_step: int

    def __init__(self, n_values: int, bucket_size: int, buckets_step: int = None):
        """
        Initializes encoder.

        :param n_values: int defines a range [0, n_values) of values that can be encoded
        :param bucket_size: number of bits in a bucket used to encode single value
        :param buckets_step: number of bits between beginnings of consecutive buckets
        """
        self._bucket_size = bucket_size
        self._buckets_step = isnone(buckets_step, bucket_size)
        self.n_values = n_values

        max_value = self.n_values - 1
        self.output_sdr_size = self._bucket_starting_pos(max_value) + self._bucket_size

    @property
    def n_active_bits(self) -> int:
        return self._bucket_size

    def encode(self, x: int) -> SparseSdr:
        """Encodes value x to sparse SDR format using bucket-based non-overlapping encoding."""
        assert x is None or x == self.ALL or 0 <= x < self.n_values, \
            f'Value must be in [0, {self.n_values}] or {self.ALL} or None; got {x}'

        if x is None:
            return np.array([], dtype=np.int)
        if x == self.ALL:
            return np.arange(self.output_sdr_size, dtype=np.int)

        left = self._bucket_starting_pos(x)
        right = left + self._bucket_size
        return np.arange(left, right, dtype=np.int)

    def decode_bit(self, bit_int: int) -> int:
        bucket_ind = bit_int // self._buckets_step
        if bucket_ind >= self.n_values:
            bucket_ind = self.n_values - 1
        return bucket_ind

    def activation_fraction(self, activation):
        return activation / self._bucket_size

    def _bucket_starting_pos(self, i):
        return i * self._buckets_step


class IntRandomEncoder:
    """
    Encodes integer values from range [0, `n_values`) as SDR with `total_bits` bits.
    Each value x is encoded with `total_bits` * `sparsity` random bits.
    Any two encoded values may overlap. Encoding scheme is initialized once and then remains fixed.
    """

    output_sdr_size: int

    _encoding_map: np.array

    def __init__(self, n_values: int, total_bits: int, sparsity: float, seed: int):
        """
        Initializes encoder.

        :param n_values: defines a range [0, n_values) of values that can be encoded
        :param total_bits: total number of bits in a resulted SDR
        :param sparsity: sparsity level of resulted SDR
        :param seed: random seed for random encoding scheme generation
        """
        self.output_sdr_size = total_bits

        value_bits = int(total_bits * sparsity)
        rng_generator = np.random.default_rng(seed=seed)

        self._encoding_map = np.empty(shape=(n_values, value_bits), dtype=np.int)
        for x in range(n_values):
            self._encoding_map[x, :] = rng_generator.choice(total_bits, size=value_bits, replace=False)

    @property
    def n_values(self) -> int:
        return self._encoding_map.shape[0]

    @property
    def n_active_bits(self):
        return self._encoding_map.shape[1]

    def encode(self, x: int) -> SparseSdr:
        """Encodes value x to sparse SDR format using random overlapping encoding."""
        return self._encoding_map[x]


class IntArrayEncoder:
    n_types: int
    output_sdr_size: int

    def __init__(
            self, shape: tuple[int, ...], n_types: int,
            # Keep for init interface compatibility with IntBucketEncoder
            bucket_size: int = None, buckets_step: int = None
    ):
        n_values = np.prod(shape)
        self.n_types = n_types
        self.output_sdr_size = self.n_types * n_values

    def encode(self, x: np.ndarray = None, mask: np.ndarray = None):
        if x is None:
            x = ~mask

        x = x.flatten()
        if mask is not None:
            indices = np.flatnonzero(mask)
        else:
            indices = np.arange(x.size)
        return indices * self.n_types + x[indices]


class SdrConcatenator:
    """Concatenates sparse SDRs."""
    output_sdr_size: int

    _shifts: Sequence[int]

    def __init__(self, input_sources: list[Any]):
        if input_sources and isinstance(input_sources[0], int):
            input_sizes = input_sources
        else:
            input_sizes = [source.output_sdr_size for source in input_sources]
        cumulative_sizes = np.cumsum(input_sizes)

        # NB: note that zero shift at the beginning is omitted
        self._shifts = cumulative_sizes[:-1]
        self.output_sdr_size = cumulative_sizes[-1]

    def concatenate(self, *sparse_sdrs):
        """Concatenates `sparse_sdrs` fixing their relative indexes."""
        size = sum(len(sdr) for sdr in sparse_sdrs)
        result = np.empty(size, dtype=np.int)

        # to speed up things do not apply zero shift to the first sdr
        first = sparse_sdrs[0]
        l, r = 0, len(first)
        result[l:r] = first

        # apply corresponding shifts to the rest inputs
        for i in range(1, len(sparse_sdrs)):
            sdr = sparse_sdrs[i]
            l = r
            r = r + len(sdr)
            result[l:r] = sdr
            result[l:r] += self._shifts[i - 1]
        return result
