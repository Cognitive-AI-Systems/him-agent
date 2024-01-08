#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

import matplotlib.pyplot as plt

from hima.modules.htm.connections import Connections
from hima.modules.belief.utils import softmax, normalize, sample_categorical_variables
from hima.modules.belief.utils import EPS, INT_TYPE, UINT_DTYPE, REAL_DTYPE, REAL64_DTYPE

from htm.bindings.sdr import SDR
from htm.bindings.math import Random

from scipy.stats import entropy
from itertools import combinations
import numpy as np
import warnings
import pygraphviz as pgv
import colormap


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
            connection_threshold: int
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
        self.synapse_lr = synapse_lr
        self.max_factors_per_var = max_factors_per_var
        self.n_vars_per_factor = n_vars_per_factor
        self.n_cells = n_cells
        self.n_vars = n_vars
        self.n_hidden_states = n_hidden_states
        self.max_factors = n_hidden_vars * max_factors_per_var
        self.max_segments_per_cell = max_segments_per_cell
        self.connection_threshold = connection_threshold

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

        filter_destroyed_segments = np.isin(
            self.segments_in_use, segments_to_prune, invert=True
        )
        self.segments_in_use = self.segments_in_use[filter_destroyed_segments]

        for segment in segments_to_prune:
            self.connections.destroySegment(segment)

        return segments_to_prune
    
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
        messages = messages[self.receptive_fields[active_segments]]
        synapse_efficiency = self.synapse_efficiency[active_segments]

        likelihood = np.array([
            np.sum([
                (np.sum(messages[l, list(indices)] * synapse_efficiency[l, list(indices)]) / len(indices)) *
                np.prod(messages[l, list(indices)] ** (1 - synapse_efficiency[l, list(indices)]))
                for k in range(self.connection_threshold, self.n_vars_per_factor + 1)
                for indices in combinations(range(self.n_vars_per_factor), k)
            ])
            for l in range(synapse_efficiency.shape[0])
        ])
        
        return np.log(likelihood)


class Layer:
    """
        This class represents a layer of the neocortex model.
    """
    context_factors: Factors | None
    internal_factors: Factors | None

    def __init__(
            self,
            n_obs_vars: int,
            n_obs_states: int,
            cells_per_column: int,
            context_factors_conf: dict = None,
            internal_factors_conf: dict = None,
            n_context_vars: int = 0,
            n_context_states: int = 0,
            n_external_vars: int = 0,
            n_external_states: int = 0,
            external_vars_boost: float = 0,
            unused_vars_boost: float = 0,
            inverse_temp_context: float = 1.0,
            inverse_temp_internal: float = 1.0,
            cell_activation_threshold: float = EPS,
            developmental_period: int = 10000,
            enable_context_connections: bool = True,
            enable_internal_connections: bool = True,
            cells_activity_lr: float = 0.1,
            seed: int = None,
    ):
        self._rng = np.random.default_rng(seed)

        if seed:
            self._legacy_rng = Random(seed)
        else:
            self._legacy_rng = Random()

        self.lr = 1.0
        self.timestep = 1
        self.developmental_period = developmental_period
        self.n_obs_vars = n_obs_vars
        self.n_hidden_vars = n_obs_vars
        self.n_obs_states = n_obs_states
        self.n_external_vars = n_external_vars
        self.n_external_states = n_external_states
        self.n_context_vars = n_context_vars
        self.n_context_states = n_context_states
        self.external_vars_boost = external_vars_boost
        self.unused_vars_boost = unused_vars_boost
        self.cells_activity_lr = cells_activity_lr

        self.cells_per_column = cells_per_column
        self.n_hidden_states = cells_per_column * n_obs_states

        self.internal_cells = self.n_hidden_vars * self.n_hidden_states
        self.external_input_size = self.n_external_vars * self.n_external_states
        self.context_input_size = self.n_context_vars * self.n_context_states
        
        self.total_cells = (
                self.internal_cells +
                self.external_input_size +
                self.context_input_size
        )
        self.total_vars = (
                self.n_hidden_vars +
                self.n_context_vars +
                self.n_external_vars
        )

        self.input_sdr_size = n_obs_vars * n_obs_states

        self.inverse_temp_context = inverse_temp_context
        self.inverse_temp_internal = inverse_temp_internal

        self.n_columns = self.n_obs_vars * self.n_obs_states

        # low probability clipping
        self.cell_activation_threshold = cell_activation_threshold

        self.internal_active_cells = SDR(self.internal_cells)
        self.external_active_cells = SDR(self.external_input_size)
        self.context_active_cells = SDR(self.context_input_size)

        self.internal_forward_messages = np.zeros(
            self.internal_cells,
            dtype=REAL64_DTYPE
        )
        self.external_messages = np.zeros(
            self.external_input_size,
            dtype=REAL64_DTYPE
        )
        self.context_messages = np.zeros(
            self.context_input_size,
            dtype=REAL64_DTYPE
        )

        self.internal_cells_activity = np.zeros_like(
            self.internal_forward_messages
        )

        self.prediction_cells = None
        self.prediction_columns = None

        # cells are numbered in the following order:
        # internal cells | context cells | external cells
        self.internal_cells_range = (
            0,
            self.internal_cells
        )
        self.context_cells_range = (
            self.internal_cells_range[1],
            self.internal_cells_range[1] + self.context_input_size
        )
        self.external_cells_range = (
            self.context_cells_range[1],
            self.context_cells_range[1] + self.external_input_size
        )

        self.enable_context_connections = enable_context_connections
        self.enable_internal_connections = enable_internal_connections

        if context_factors_conf is not None and self.enable_context_connections:
            context_factors_conf['n_cells'] = self.total_cells
            context_factors_conf['n_vars'] = self.total_vars
            context_factors_conf['n_hidden_states'] = self.n_hidden_states
            context_factors_conf['n_hidden_vars'] = self.n_hidden_vars
            self.context_factors = Factors(**context_factors_conf)
        else:
            self.context_factors = None
            self.enable_context_connections = False

        if internal_factors_conf is not None and self.enable_internal_connections:
            internal_factors_conf['n_cells'] = self.total_cells
            internal_factors_conf['n_vars'] = self.total_vars
            internal_factors_conf['n_hidden_states'] = self.n_hidden_states
            internal_factors_conf['n_hidden_vars'] = self.n_hidden_vars
            self.internal_factors = Factors(**internal_factors_conf)
        else:
            self.internal_factors = None
            self.enable_internal_connections = False

        assert (self.context_factors is not None) or (self.internal_factors is not None)

    def set_external_messages(self, messages=None):
        # update external cells
        if messages is not None:
            self.external_messages = messages
        elif self.external_input_size != 0:
            self.external_messages = normalize(
                np.zeros(self.external_input_size).reshape((self.n_external_vars, -1))
            ).flatten()

    def set_context_messages(self, messages=None):
        # update external cells
        if messages is not None:
            self.context_messages = messages
        elif self.context_input_size != 0:
            self.context_messages = normalize(
                np.zeros(self.context_input_size).reshape((self.n_context_vars, -1))
            ).flatten()

    def make_state_snapshot(self):
        return (
            # mutable attributes:
            self.internal_forward_messages.copy(),
            # immutable attributes:
            self.external_messages,
            self.context_messages,
            self.prediction_cells,
            self.prediction_columns
        )

    def restore_last_snapshot(self, snapshot):
        if snapshot is None:
            return

        (
            self.internal_forward_messages,
            self.external_messages,
            self.context_messages,
            self.prediction_cells,
            self.prediction_columns
        ) = snapshot

        # explicitly copy mutable attributes:
        self.internal_forward_messages = self.internal_forward_messages.copy()

    def reset(self):
        self.internal_forward_messages = np.zeros(
            self.internal_cells,
            dtype=REAL64_DTYPE
        )
        self.external_messages = np.zeros(
            self.external_input_size,
            dtype=REAL64_DTYPE
        )
        self.context_messages = np.zeros(
            self.context_input_size,
            dtype=REAL64_DTYPE
        )

        self.prediction_cells = None
        self.prediction_columns = None

    def predict(self, include_context_connections=True, include_internal_connections=False, **_):
        # step 1: predict cells based on context and external messages
        # block internal messages
        # think about it as thalamus orchestration of the neocortex
        if include_context_connections and self.enable_context_connections:
            messages = np.zeros(self.total_cells)
            messages[
                self.context_cells_range[0]:
                self.context_cells_range[1]
            ] = self.context_messages

            messages[
                self.external_cells_range[0]:
                self.external_cells_range[1]
            ] = self.external_messages

            self._propagate_belief(
                messages,
                self.context_factors,
                self.inverse_temp_context,
            )

        # step 2: update predictions based on internal and external connections
        # block context and external messages
        if include_internal_connections and self.enable_internal_connections:
            previous_internal_messages = self.internal_forward_messages.copy()

            messages = np.zeros(self.total_cells)

            messages[
                self.internal_cells_range[0]:
                self.internal_cells_range[1]
            ] = self.internal_forward_messages

            self._propagate_belief(
                messages,
                self.internal_factors,
                self.inverse_temp_internal,
            )

            # consolidate previous and new messages
            self.internal_forward_messages *= previous_internal_messages
            self.internal_forward_messages = normalize(
                self.internal_forward_messages.reshape(
                    (self.n_hidden_vars, self.n_hidden_states)
                )
            ).flatten()

        self.prediction_cells = self.internal_forward_messages.copy()
        self.prediction_columns = self.prediction_cells.reshape(
            (self.n_columns, self.cells_per_column)
        ).sum(axis=-1)

    def observe(
            self,
            observation: np.ndarray,
            learn: bool = True
    ):
        """
            observation: pattern in sparse representation
        """
        # update messages
        cells = self._get_cells_for_observation(observation)
        obs_factor = np.zeros_like(self.internal_forward_messages)
        obs_factor[cells] = 1
        self.internal_forward_messages *= obs_factor

        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            self.internal_forward_messages = normalize(
                self.internal_forward_messages.reshape((self.n_hidden_vars, -1)),
                obs_factor.reshape((self.n_hidden_vars, -1))
            ).flatten()

        # update connections
        if learn and self.lr > 0:
            # sample cells from messages (1-step Monte-Carlo learning)
            # internal cells cooldown to avoid self-loops
            internal_messages = self.internal_forward_messages.copy()
            internal_messages *= (1 - self.internal_cells_activity)
            internal_messages = normalize(internal_messages.reshape((self.n_hidden_vars, -1)))

            self.internal_active_cells.sparse = self._sample_cells(
                internal_messages
            )
            self.context_active_cells.sparse = self._sample_cells(
                self.context_messages.reshape((self.n_context_vars, -1))
            )
            if len(self.external_messages) > 0:
                self.external_active_cells.sparse = self._sample_cells(
                    self.external_messages.reshape((self.n_external_vars, -1))
                )

            # learn context segments
            # use context cells and external cells to predict internal cells
            if self.enable_context_connections:
                self._learn(
                    np.concatenate(
                        [
                            (
                                self.context_cells_range[0] +
                                self.context_active_cells.sparse
                            ),
                            (
                                self.external_cells_range[0] +
                                self.external_active_cells.sparse
                            )
                        ]
                    ),
                    self.internal_active_cells.sparse,
                    self.context_factors,
                    prune_segments=(self.timestep % self.developmental_period) == 0
                )

            # learn internal segments
            if self.enable_internal_connections:
                self._learn(
                    self.internal_active_cells.sparse,
                    self.internal_active_cells.sparse,
                    self.internal_factors,
                    prune_segments=(self.timestep % self.developmental_period) == 0
                )

            self.internal_cells_activity += self.cells_activity_lr * (
                    self.internal_active_cells.dense - self.internal_cells_activity
            )

        self.timestep += 1

    def _propagate_belief(
            self,
            messages: np.ndarray,
            factors: Factors,
            inverse_temperature=1.0,
    ):
        """
        Calculate messages for internal cells based on messages from all cells.
            messages: should be an array of size total_cells
        """
        # filter dendrites that have low activation likelihood
        active_cells = SDR(self.total_cells)

        active_cells.sparse = np.flatnonzero(
            messages >= self.cell_activation_threshold
        )

        num_connected_segment = factors.connections.computeActivity(
            active_cells,
            False
        )

        active_segments = np.flatnonzero(
            num_connected_segment >= factors.connection_threshold
        )
        cells_for_active_segments = factors.connections.mapSegmentsToCells(active_segments)

        log_next_messages = np.full(
            self.internal_cells,
            fill_value=-np.inf,
            dtype=REAL_DTYPE
        )

        # excitation activity
        if len(active_segments) > 0:
            factors_for_active_segments = factors.factor_for_segment[active_segments]
            log_factor_value = factors.log_factor_values_per_segment[active_segments]

            log_likelihood = factors.calculate_segment_likelihood(
                messages,
                active_segments
            )

            log_excitation_per_segment = log_likelihood + log_factor_value

            # uniquely encode pairs (factor, cell) for each segment
            cell_factor_id_per_segment = (
                    factors_for_active_segments * self.total_cells
                    + cells_for_active_segments
            )

            # group segments by factors
            sorting_inxs = np.argsort(cell_factor_id_per_segment)
            cells_for_active_segments = cells_for_active_segments[sorting_inxs]
            cell_factor_id_per_segment = cell_factor_id_per_segment[sorting_inxs]
            log_excitation_per_segment = log_excitation_per_segment[sorting_inxs]

            cell_factor_id_excitation, reduce_inxs = np.unique(
                cell_factor_id_per_segment, return_index=True
            )

            # approximate log sum with max
            log_excitation_per_factor = np.maximum.reduceat(log_excitation_per_segment, reduce_inxs)

            # group segments by cells
            cells_for_factors = cells_for_active_segments[reduce_inxs]

            sort_inxs = np.argsort(cells_for_factors)
            cells_for_factors = cells_for_factors[sort_inxs]
            log_excitation_per_factor = log_excitation_per_factor[sort_inxs]

            cells_with_factors, reduce_inxs = np.unique(cells_for_factors, return_index=True)

            log_prediction_for_cells_with_factors = np.add.reduceat(
                log_excitation_per_factor, indices=reduce_inxs
            )

            log_next_messages[cells_with_factors] = log_prediction_for_cells_with_factors

        log_next_messages = log_next_messages.reshape((self.n_hidden_vars, self.n_hidden_states))

        # shift log value for stability
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)

            means = log_next_messages.mean(
                axis=-1,
                where=~np.isinf(log_next_messages)
            ).reshape((-1, 1))

        means[np.isnan(means)] = 0

        log_next_messages -= means

        log_next_messages = inverse_temperature * log_next_messages

        next_messages = normalize(np.exp(log_next_messages))

        next_messages = next_messages.flatten()

        assert ~np.any(np.isnan(next_messages))

        self.internal_forward_messages = next_messages

    def _learn(
            self,
            active_cells,
            next_active_cells,
            factors: Factors,
            prune_segments=False,
            update_factor_score=True,
    ):
        (
            segments_to_reinforce,
            segments_to_punish,
            cells_to_grow_new_segments
        ) = self._calculate_learning_segments(
            active_cells,
            next_active_cells,
            factors
        )

        new_segments = self._grow_new_segments(
            cells_to_grow_new_segments,
            active_cells,
            factors,
            update_factor_score=update_factor_score
        )

        factors.segments_in_use = np.append(
            factors.segments_in_use,
            new_segments[np.isin(new_segments, factors.segments_in_use, invert=True)]
        )

        factors.update_factors(
            np.concatenate(
                [
                    segments_to_reinforce,
                    new_segments
                ]
            ),
            segments_to_punish,
            prune=prune_segments,
        )

        factors.update_synapses(
            np.concatenate(
                [
                    segments_to_reinforce,
                    new_segments,
                    segments_to_punish
                ]
            ),
            active_cells
        )

        # update var score
        vars_for_correct_segments = np.unique(
            self._vars_for_cells(factors.receptive_fields[segments_to_reinforce].flatten())
        )

        vars_for_incorrect_segments = np.unique(
            self._vars_for_cells(factors.receptive_fields[segments_to_punish].flatten())
        )

        factors.var_score[vars_for_correct_segments] += factors.var_score_lr * (
                1 - factors.var_score[vars_for_correct_segments]
        )

        factors.var_score[vars_for_incorrect_segments] -= factors.var_score_lr * factors.var_score[
            vars_for_incorrect_segments
        ]

    def _calculate_learning_segments(self, active_cells, next_active_cells, factors: Factors):
        # determine which segments are learning and growing
        active_cells_sdr = SDR(self.total_cells)
        active_cells_sdr.sparse = active_cells

        num_connected = factors.connections.computeActivity(
            active_cells_sdr,
            False
        )

        active_segments = np.flatnonzero(num_connected >= factors.connection_threshold)

        cells_for_active_segments = factors.connections.mapSegmentsToCells(active_segments)

        mask = np.isin(cells_for_active_segments, next_active_cells)
        segments_to_learn = active_segments[mask]
        segments_to_punish = active_segments[~mask]

        cells_to_grow_new_segments = next_active_cells[
            np.isin(next_active_cells, cells_for_active_segments, invert=True)
        ]

        return (
            segments_to_learn.astype(UINT_DTYPE),
            segments_to_punish.astype(UINT_DTYPE),
            cells_to_grow_new_segments.astype(UINT_DTYPE)
        )

    def _get_cells_for_observation(self, obs_states):
        vars_for_obs_states = obs_states // self.n_obs_states
        all_vars = np.arange(self.n_obs_vars)
        vars_without_states = all_vars[np.isin(all_vars, vars_for_obs_states, invert=True)]

        cells_for_empty_vars = self._get_cells_in_vars(vars_without_states)

        cells_in_columns = (
                (
                    obs_states * self.cells_per_column
                ).reshape((-1, 1)) +
                np.arange(self.cells_per_column, dtype=UINT_DTYPE)
            ).flatten()

        return np.concatenate([cells_for_empty_vars, cells_in_columns])

    def _get_cells_in_vars(self, variables):
        internal_vars_mask = variables < self.n_hidden_vars
        context_vars_mask = (
                (variables >= self.n_hidden_vars) &
                (variables < (self.n_hidden_vars + self.n_context_vars))
        )
        external_vars_mask = (
                variables >= (self.n_hidden_vars + self.n_context_vars)
        )

        cells_in_internal_vars = (
                (variables[internal_vars_mask] * self.n_hidden_states).reshape((-1, 1)) +
                np.arange(self.n_hidden_states, dtype=UINT_DTYPE)
        ).flatten()

        cells_in_context_vars = (
                ((variables[context_vars_mask] - self.n_hidden_vars) *
                 self.n_context_states).reshape((-1, 1)) +
                np.arange(self.n_context_states, dtype=UINT_DTYPE)
        ).flatten() + self.context_cells_range[0]

        cells_in_external_vars = (
                ((variables[external_vars_mask] - self.n_hidden_vars - self.n_context_vars) *
                 self.n_external_states).reshape((-1, 1)) +
                np.arange(self.n_external_states, dtype=UINT_DTYPE)
        ).flatten() + self.external_cells_range[0]

        return np.concatenate(
            [
                cells_in_internal_vars,
                cells_in_context_vars,
                cells_in_external_vars
            ]
        )

    def _filter_cells_by_vars(self, cells, variables):
        cells_in_vars = self._get_cells_in_vars(variables)

        mask = np.isin(cells, cells_in_vars)

        return cells[mask]

    def _sample_cells(self, messages, shift=0):
        """
            messages.shape = (n_vars, n_states)
        """
        if messages.sum() == 0:
            return np.empty(0, dtype=UINT_DTYPE)

        n_vars, n_states = messages.shape

        next_states = sample_categorical_variables(
            messages,
            self._rng
        )
        # transform states to cell ids
        next_cells = next_states + np.arange(
            0,
            n_states*n_vars,
            n_states
        )
        next_cells += shift

        return next_cells.astype(UINT_DTYPE)

    def _vars_for_cells(self, cells):
        internal_cells_mask = (
                (cells >= self.internal_cells_range[0]) &
                (cells < self.internal_cells_range[1])
            )
        internal_cells = cells[internal_cells_mask]

        context_cells_mask = (
                (cells >= self.context_cells_range[0]) &
                (cells < self.context_cells_range[1])
        )
        context_cells = cells[context_cells_mask]

        external_cells_mask = (
                    (cells >= self.external_cells_range[0]) &
                    (cells < self.external_cells_range[1])
            )
        external_cells = cells[external_cells_mask]

        internal_cells_vars = internal_cells // self.n_hidden_states
        context_cells_vars = (
            self.n_hidden_vars +
            (context_cells - self.context_cells_range[0]) // self.n_context_states
        )
        external_cells_vars = (
            self.n_hidden_vars + self.n_hidden_vars +
            (external_cells - self.external_cells_range[0]) // self.n_external_states
        )

        vars_ = np.empty_like(cells, dtype=UINT_DTYPE)
        vars_[internal_cells_mask] = internal_cells_vars
        vars_[context_cells_mask] = context_cells_vars
        vars_[external_cells_mask] = external_cells_vars
        return vars_

    def _grow_new_segments(
            self,
            new_segment_cells,
            growth_candidates,
            factors: Factors,
            update_factor_score=True
    ):
        candidate_vars = np.unique(self._vars_for_cells(growth_candidates))
        # free space for new segments
        n_segments_after_growing = len(factors.segments_in_use) + len(new_segment_cells)
        if n_segments_after_growing > factors.max_segments:
            n_segments_to_prune = n_segments_after_growing - factors.max_segments
            factors.prune_segments(n_segments_to_prune)

        if update_factor_score:
            factors.update_factor_score()

        factor_score = factors.factor_score.copy()
        factors_with_segments = factors.factors_in_use

        # filter non-active factors
        active_vars = SDR(self.total_vars)
        active_vars.sparse = candidate_vars
        n_active_vars = factors.factor_connections.computeActivity(
            active_vars,
            False
        )
        active_factors_mask = n_active_vars >= factors.n_vars_per_factor

        factor_score = factor_score[active_factors_mask[factors_with_segments]]
        factors_with_segments = factors_with_segments[active_factors_mask[factors_with_segments]]

        new_segments = list()

        # each cell corresponds to one variable
        # we iterate only for internal cells here
        for cell in new_segment_cells:
            n_segments = factors.connections.numSegments(cell)

            # this condition is usually loose,
            # so it's just a placeholder for extreme cases
            if n_segments >= factors.max_segments_per_cell:
                continue

            # get factors for cell
            var = cell // self.n_hidden_states
            cell_factors = np.array(
                factors.factor_connections.segmentsForCell(var)
            )
            cell_factors = cell_factors[np.isin(
                cell_factors, factors_with_segments
            )]

            score = np.zeros(factors.max_factors_per_var)
            candidate_factors = np.full(factors.max_factors_per_var, fill_value=-1)

            if len(cell_factors) > 0:
                mask = np.isin(factors_with_segments, cell_factors)

                score[:len(cell_factors)] = factor_score[mask]
                candidate_factors[:len(cell_factors)] = factors_with_segments[mask]

            factor_id = self._rng.choice(
                candidate_factors,
                size=1,
                p=softmax(score)
            )

            if factor_id != -1:
                variables = factors.factor_vars[factor_id]
            else:
                # exclude self-loop
                candidate_vars_for_cell = candidate_vars[np.isin(candidate_vars, var, invert=True)]
                # select cells for a new factor
                var_score = factors.var_score.copy()

                used_vars, counts = np.unique(
                    factors.factor_vars[factors.factors_in_use].flatten(),
                    return_counts=True
                )
                # TODO can we make it more static to not compute it in cycle?
                var_score[used_vars] *= np.exp(-self.unused_vars_boost * counts)
                var_score[self.n_hidden_vars + self.n_context_vars:] += self.external_vars_boost

                var_score = var_score[candidate_vars_for_cell]

                # sample size can't be bigger than number of variables
                sample_size = min(factors.n_vars_per_factor, len(candidate_vars_for_cell))

                if sample_size == 0:
                    return np.empty(0, dtype=UINT_DTYPE)

                variables = self._rng.choice(
                    candidate_vars_for_cell,
                    size=sample_size,
                    p=softmax(var_score),
                    replace=False
                )

                factor_id = factors.factor_connections.createSegment(
                    var,
                    maxSegmentsPerCell=factors.max_factors_per_var
                )

                factors.factor_connections.growSynapses(
                    factor_id,
                    variables,
                    0.6,
                    self._legacy_rng,
                    maxNew=factors.n_vars_per_factor
                )

                factors.factor_vars[factor_id] = variables
                factors.factors_in_use = np.append(factors.factors_in_use, factor_id)

            candidates = self._filter_cells_by_vars(growth_candidates, variables)

            # don't create a segment that will never activate
            if len(candidates) < factors.n_vars_per_factor:
                continue

            new_segment = factors.connections.createSegment(cell, factors.max_segments_per_cell)

            factors.connections.growSynapses(
                new_segment,
                candidates,
                0.6,
                self._legacy_rng,
                maxNew=factors.n_vars_per_factor
            )

            factors.factor_for_segment[new_segment] = factor_id
            factors.log_factor_values_per_segment[new_segment] = factors.initial_log_factor_value
            factors.receptive_fields[new_segment] = candidates

            new_segments.append(new_segment)

        return np.array(new_segments, dtype=UINT_DTYPE)

    def draw_messages(self, messages, figsize=10, aspect_ratio=0.3, non_zero=True):
        n_cols = int(np.ceil(np.sqrt(self.n_hidden_vars / aspect_ratio)))
        n_rows = int(np.floor(n_cols * aspect_ratio))

        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(figsize, figsize*aspect_ratio))

        for var in range(len(messages)):
            message = messages[var]
            if non_zero:
                mask = message > 0
                message = message[mask]
                states = list(np.flatnonzero(mask))
            else:
                states = list(range(len(message)))

            if n_rows == 1:
                ax = axs[var]
            else:
                ax = axs[var // n_cols][var % n_cols]
            ax.grid()
            ax.set_ylim(0, 1)
            ax.bar(
                np.arange(len(message)),
                message,
                tick_label=states
            )
        return fig

    def draw_factor_graph(self, path=None, show_messages=False):
        g = pgv.AGraph(strict=False, directed=False)

        for factors, type_ in zip(
            (self.context_factors, self.internal_factors),
            ('c', 'i')
        ):
            if factors is not None:
                if type_ == 'c':
                    line_color = "#625f5f"
                else:
                    line_color = "#5968ff"
                # count segments per factor
                factors_in_use, n_segments = np.unique(
                    factors.factor_for_segment[factors.segments_in_use],
                    return_counts=True
                )
                cmap = colormap.Colormap().get_cmap_heat()
                factor_score = n_segments / n_segments.max()
                var_score = entropy(
                    self.internal_forward_messages.reshape((self.n_hidden_vars, -1)),
                    axis=-1
                )
                var_score /= (EPS + var_score.max())

                for fid, score in zip(factors_in_use, factor_score):
                    var_next = factors.factor_connections.cellForSegment(fid)
                    g.add_node(
                        f'f{fid}{type_}',
                        shape='box',
                        style='filled',
                        fillcolor=colormap.rgb2hex(
                            *(cmap(int(255*score))[:-1]),
                            normalised=True
                        ),
                        color=line_color
                    )
                    if show_messages:
                        g.add_node(
                            f'h{var_next}',
                            style='filled',
                            fillcolor=colormap.rgb2hex(
                                *(cmap(int(255*var_score[var_next]))[:-1]),
                                normalised=True
                            ),
                        )

                    g.add_edge(f'h{var_next}', f'f{fid}{type_}', color=line_color)
                    for var_prev in factors.factor_vars[fid]:
                        if var_prev < self.n_hidden_vars:
                            g.add_edge(f'f{fid}{type_}', f'h{var_prev}', color=line_color)
                        elif self.n_hidden_vars <= var_prev < self.n_hidden_vars + self.n_context_vars:
                            g.add_edge(f'f{fid}{type_}', f'c{var_prev}', color=line_color)
                        else:
                            g.add_edge(f'f{fid}{type_}', f'e{var_prev}', color=line_color)

        g.layout(prog='dot')
        return g.draw(path, format='png')
