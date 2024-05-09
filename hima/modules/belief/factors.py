#  Copyright (c) 2024 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from __future__ import annotations

from hima.modules.htm.connections import Connections
from hima.modules.belief.utils import EPS, INT_TYPE, UINT_DTYPE, REAL_DTYPE, REAL64_DTYPE, \
    sigmoid, sample_categorical_variables, softmax
import numpy as np


class Factors:
    def __init__(
            self,
            n_cells,
            n_vars,
            n_hidden_states,
            n_hidden_vars,
            n_vars_per_factor,
            max_factors_per_var,
            factor_lr,
            synapse_lr,
            segment_activity_lr,
            var_score_lr,
            initial_log_factor_value,
            initial_synapse_value,
            max_segments,
            fraction_of_segments_to_prune,
            max_segments_per_cell,
            min_log_factor_value=-1,
            seed=None
    ):
        """
            hidden vars are those that we predict, or output vars
        """
        self.fraction_of_segments_to_prune = fraction_of_segments_to_prune
        self.max_segments = max_segments
        self.initial_log_factor_value = initial_log_factor_value
        self.initial_synapse_value = initial_synapse_value
        self.var_score_lr = var_score_lr
        self.segment_activity_lr = segment_activity_lr
        self.factor_lr = factor_lr
        self.min_log_factor_value = min_log_factor_value
        self.synapse_lr = synapse_lr
        self.max_factors_per_var = max_factors_per_var
        self.n_vars_per_factor = n_vars_per_factor
        self.n_cells = n_cells
        self.n_vars = n_vars
        self.n_hidden_states = n_hidden_states
        self.max_factors = n_hidden_vars * max_factors_per_var
        self.max_segments_per_cell = max_segments_per_cell
        self.seed = seed
        self._rng = np.random.default_rng(self.seed)

        self.connections = Connections(
            numCells=n_cells,
            connectedThreshold=0.5
        )

        self.factor_connections = Connections(
            numCells=n_vars,
            connectedThreshold=0.5
        )

        self.receptive_fields = np.full(
            (max_segments, n_vars_per_factor),
            fill_value=-1,
            dtype=INT_TYPE
        )

        self.synapse_efficiency = np.full(
            (max_segments, n_vars_per_factor),
            fill_value=self.initial_synapse_value,
            dtype=REAL64_DTYPE
        )

        self.log_factor_values_per_segment = np.full(
            max_segments,
            fill_value=initial_log_factor_value,
            dtype=REAL64_DTYPE
        )

        self.segment_activity = np.ones(
            max_segments,
            dtype=REAL64_DTYPE
        )

        self.factor_for_segment = np.full(
            max_segments,
            fill_value=-1,
            dtype=INT_TYPE
        )

        self.factor_vars = np.full(
            (self.max_factors, n_vars_per_factor),
            fill_value=-1,
            dtype=INT_TYPE
        )

        self.var_score = np.ones(
            n_vars,
            dtype=REAL64_DTYPE
        )

        self.segments_in_use = np.empty(0, dtype=UINT_DTYPE)
        self.active_segments = np.empty(0, dtype=UINT_DTYPE)
        self.active_segments_loglikelihood = np.empty(0, dtype=REAL64_DTYPE)
        self.factors_in_use = np.empty(0, dtype=UINT_DTYPE)
        self.factor_score = np.empty(0, dtype=REAL_DTYPE)

    def update_factor_score(self):
        # sum factor values for every factor
        if len(self.segments_in_use) > 0:
            factor_for_segment = self.factor_for_segment[self.segments_in_use]
            log_factor_values = self.log_factor_values_per_segment[self.segments_in_use]
            segment_activation_freq = self.segment_activity[self.segments_in_use]

            sort_ind = np.argsort(factor_for_segment)
            factors_sorted = factor_for_segment[sort_ind]
            segments_sorted_values = log_factor_values[sort_ind]
            segments_sorted_freq = segment_activation_freq[sort_ind]

            factors_with_segments, split_ind, counts = np.unique(
                factors_sorted,
                return_index=True,
                return_counts=True
            )

            score = np.exp(segments_sorted_values) * segments_sorted_freq
            factor_score = np.add.reduceat(score, split_ind) / counts

            # destroy factors without segments
            mask = np.isin(self.factors_in_use, factors_with_segments, invert=True)
            factors_without_segments = self.factors_in_use[mask]

            for factor in factors_without_segments:
                self.factor_connections.destroySegment(factor)
                self.factor_vars[factor] = np.full(self.n_vars_per_factor, fill_value=-1)

            self.factors_in_use = factors_with_segments.copy()
            self.factor_score = factor_score
        else:
            self.factor_score = np.empty(0)

    def prune_segments(self, n_segments):
        log_value = self.log_factor_values_per_segment[self.segments_in_use]
        activity = self.segment_activity[self.segments_in_use]

        score = (
                np.exp(log_value) * activity
        )

        segments_to_prune = self.segments_in_use[
            np.argpartition(score, n_segments)[:n_segments]
        ]

        self._destroy_segments(segments_to_prune)

        return segments_to_prune

    def _destroy_segments(self, segments):
        filter_destroyed_segments = np.isin(
            self.segments_in_use, segments, invert=True
        )
        self.segments_in_use = self.segments_in_use[filter_destroyed_segments]

        for segment in segments:
            self.connections.destroySegment(segment)

    def update_factors(
            self,
            segments_to_reinforce,
            segments_to_punish,
            prune=False,
    ):
        w = self.log_factor_values_per_segment[segments_to_reinforce]
        self.log_factor_values_per_segment[
            segments_to_reinforce
        ] += np.log1p(self.factor_lr * (np.exp(-w) - 1))

        self.log_factor_values_per_segment[
            segments_to_punish
        ] += np.log1p(-self.factor_lr)

        active_segments = np.concatenate([segments_to_reinforce, segments_to_punish])
        non_active_segments = self.segments_in_use[
            np.isin(self.segments_in_use, active_segments, invert=True)
        ]

        self.segment_activity[active_segments] += self.segment_activity_lr * (
                1 - self.segment_activity[active_segments]
        )
        self.segment_activity[non_active_segments] -= (
                self.segment_activity_lr * self.segment_activity[non_active_segments]
        )

        w = self.log_factor_values_per_segment[active_segments]
        self._destroy_segments(active_segments[w < self.min_log_factor_value])

        # prune segments based on their activity and factor value
        if prune:
            n_segments_to_prune = int(
                self.fraction_of_segments_to_prune * len(self.segments_in_use)
            )
            self.prune_segments(n_segments_to_prune)

    def update_synapses(
            self,
            active_segments,
            active_cells
    ):
        active_synapses_dense = np.isin(
            self.receptive_fields,
            active_cells
        ).astype(REAL64_DTYPE)

        active_segments_dense = np.zeros(self.receptive_fields.shape[0])
        active_segments_dense[active_segments] = 1

        delta = active_segments_dense.reshape((-1, 1)) - self.synapse_efficiency
        delta *= active_synapses_dense

        self.synapse_efficiency += self.synapse_lr * delta

    def calculate_segment_likelihood(
            self,
            messages,
            active_segments
    ):
        self.active_segments = active_segments.copy()

        messages = messages[self.receptive_fields[active_segments]]
        synapse_efficiency = self.synapse_efficiency[active_segments]

        dependent_part = messages * synapse_efficiency
        dependent_part = np.mean(synapse_efficiency, axis=-1) * np.log(
            np.mean(dependent_part, axis=-1) + EPS
        )

        independent_part = np.sum((1 - synapse_efficiency) * np.log(messages), axis=-1)

        self.active_segments_loglikelihood = dependent_part + independent_part

        return self.active_segments_loglikelihood

    def sample_segments(self):
        # TODO make it correct for multiple factors
        # probability that no segments will be activated
        p = np.prod(1 - sigmoid(self.active_segments_loglikelihood))
        gamma = self._rng.random()
        if gamma > p:
            winners = sample_categorical_variables(
                softmax(self.active_segments_loglikelihood).reshape(
                    1, -1
                ),
                self._rng
            )
        else:
            winners = []
        return winners

    def segments_to_receptive_field(self, segments):
        return np.unique(self.receptive_fields[segments].flatten())

