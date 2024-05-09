#  Copyright (c) 2024 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

import json

from hima.modules.belief.utils import softmax, normalize, sample_categorical_variables
from hima.modules.belief.utils import EPS, UINT_DTYPE, REAL_DTYPE, REAL64_DTYPE
from hima.modules.belief.utils import get_data, send_string, NumpyEncoder
from hima.modules.belief.factors import Factors
from hima.modules.belief.cortial_column.layer import Layer
from hima.common.sdr import sparse_to_dense

from htm.bindings.sdr import SDR
from htm.bindings.math import Random

import matplotlib.pyplot as plt
from scipy.stats import entropy
import numpy as np
import warnings
import pygraphviz as pgv
import colormap
import socket

HOST = "127.0.0.1"
PORT = 5555

class BioDHTM(Layer):
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
            n_hidden_vars_per_obs_var: int = 1,
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
            replace_prior: bool = False,
            bursting_threshold: float = EPS,
            override_context: bool = True,
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
        self.n_hidden_vars_per_obs_var = n_hidden_vars_per_obs_var
        self.n_hidden_vars = n_obs_vars * n_hidden_vars_per_obs_var
        self.n_obs_states = n_obs_states
        self.n_external_vars = n_external_vars
        self.n_external_states = n_external_states
        self.n_context_vars = n_context_vars
        self.n_context_states = n_context_states
        self.external_vars_boost = external_vars_boost
        self.unused_vars_boost = unused_vars_boost
        self.cells_activity_lr = cells_activity_lr
        self.replace_uniform_prior = replace_prior
        self.bursting_threshold = bursting_threshold
        self.override_context = override_context

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

        if self.override_context:
            if self.internal_active_cells.size != self.context_active_cells.size:
                raise Warning(
                    "Context override will not work as context and internal sizes are different."
                )

        self.internal_messages = np.zeros(
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
            self.internal_messages
        )

        self.prediction_cells = None
        self.prediction_columns = None
        self.observation_messages = None

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

        self.cells_to_grow_new_context_segments = np.empty(0)
        self.new_context_segments = np.empty(0)

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

        self.state_information = 0

    def reset(self):
        self.internal_messages = np.zeros(
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

        self.context_active_cells.sparse = []
        self.internal_active_cells.sparse = []
        self.external_active_cells.sparse = []

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
            previous_internal_messages = self.internal_messages.copy()

            messages = np.zeros(self.total_cells)

            messages[
                self.internal_cells_range[0]:
                self.internal_cells_range[1]
            ] = self.internal_messages

            self._propagate_belief(
                messages,
                self.internal_factors,
                self.inverse_temp_internal,
            )

            # consolidate previous and new messages
            self.internal_messages *= previous_internal_messages
            self.internal_messages = normalize(
                self.internal_messages.reshape(
                    (self.n_hidden_vars, self.n_hidden_states)
                )
            ).flatten()

        self.prediction_cells = self.internal_messages.copy()

        self.prediction_columns = self.prediction_cells.reshape(
            -1, self.cells_per_column
        ).sum(axis=-1)

        self.prediction_columns = self.prediction_columns.reshape(
            -1, self.n_hidden_vars_per_obs_var, self.n_obs_states
        ).mean(axis=1).flatten()

    def observe(
            self,
            observation: np.ndarray,
            learn: bool = True
    ):
        """
            observation: pattern in sparse representation
        """
        # update messages
        self._update_posterior(observation)

        # update connections
        if learn and self.lr > 0:
            # sample cells from messages (1-step Monte-Carlo learning)
            if (
                    self.override_context
                    and
                    (self.context_active_cells.size == self.internal_active_cells.size)
                    and
                    (len(self.internal_active_cells.sparse) > 0)
            ):
                self.context_active_cells.sparse = self.internal_active_cells.sparse
            else:
                self.context_active_cells.sparse = self._sample_cells(
                    self.context_messages.reshape(self.n_context_vars, -1)
                )

            self.internal_active_cells.sparse = self._sample_cells(
                self.internal_messages.reshape(self.n_hidden_vars, -1)
            )

            if len(self.external_messages) > 0:
                self.external_active_cells.sparse = self._sample_cells(
                    self.external_messages.reshape(self.n_external_vars, -1)
                )

            # learn context segments
            # use context cells and external cells to predict internal cells
            if self.enable_context_connections:
                (
                    self.cells_to_grow_new_context_segments,
                    self.new_context_segments
                ) = self._learn(
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

    def _update_posterior(self, observation):
        self.observation_messages = sparse_to_dense(observation, size=self.input_sdr_size)
        cells = self._get_cells_for_observation(observation)
        obs_factor = sparse_to_dense(cells, like=self.internal_messages)

        messages = self.internal_messages.reshape(self.n_hidden_vars, -1)
        obs_factor = obs_factor.reshape(self.n_hidden_vars, -1)

        messages = normalize(messages * obs_factor, obs_factor)

        # detect bursting vars
        bursting_vars_mask = self._detect_bursting_vars(messages, obs_factor)

        if self.replace_uniform_prior:
            # replace priors for bursting vars
            if np.any(bursting_vars_mask):
                # TODO decrease probability to sample frequently active cells
                # TODO decrease probability to sample cells with many segments
                bursting_factor = obs_factor[bursting_vars_mask]
                winners = self._sample_cells(normalize(bursting_factor))
                bursting_factor = sparse_to_dense(
                    winners,
                    size=bursting_factor.size,
                    dtype=bursting_factor.dtype
                ).reshape(bursting_factor.shape)

                messages[bursting_vars_mask] = bursting_factor

        self.internal_messages = messages.flatten()

    def _detect_bursting_vars(self, messages, obs_factor):
        """
            messages: (n_vars, n_states)
            obs_factor: (n_vars, n_states)
        """
        n_states = obs_factor.sum(axis=-1)
        self.state_information = (
                np.log(n_states) +
                np.sum(
                    messages * np.log(
                        np.clip(
                            messages, EPS, None
                        )
                    ),
                    axis=-1
                )
        )

        return self.state_information < self.bursting_threshold

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
            num_connected_segment >= factors.n_vars_per_factor
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

            # avoid overflow
            log_prediction_for_cells_with_factors[
                log_prediction_for_cells_with_factors < -100
            ] = -np.inf

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

        self.internal_messages = next_messages

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

        return cells_to_grow_new_segments, new_segments

    def _calculate_learning_segments(self, active_cells, next_active_cells, factors: Factors):
        # determine which segments are learning and growing
        active_cells_sdr = SDR(self.total_cells)
        active_cells_sdr.sparse = active_cells

        num_connected = factors.connections.computeActivity(
            active_cells_sdr,
            False
        )

        active_segments = np.flatnonzero(num_connected >= factors.n_vars_per_factor)

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
        obs_states_per_var = obs_states - vars_for_obs_states * self.n_obs_states

        hid_vars = (
                np.tile(np.arange(self.n_hidden_vars_per_obs_var), len(vars_for_obs_states)) +
                vars_for_obs_states * self.n_hidden_vars_per_obs_var
        )
        hid_columns = (
                np.repeat(obs_states_per_var, self.n_hidden_vars_per_obs_var) +
                self.n_obs_states * hid_vars
        )

        all_vars = np.arange(self.n_hidden_vars)
        vars_without_states = all_vars[np.isin(all_vars, hid_vars, invert=True)]

        cells_for_empty_vars = self._get_cells_in_vars(vars_without_states)

        cells_in_columns = (
                (
                        hid_columns * self.cells_per_column
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
            n_states * n_vars,
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

    @staticmethod
    def draw_messages(
            messages,
            n_vars,
            figsize=10,
            aspect_ratio=0.3,
            non_zero=True
    ):
        messages = messages.reshape(n_vars, -1)
        n_cols = max(int(np.ceil(np.sqrt(n_vars / aspect_ratio))), 1)
        n_rows = max(int(np.floor(n_cols * aspect_ratio)), 1)

        fig, axs = plt.subplots(
            nrows=n_rows, ncols=n_cols, figsize=(figsize, figsize * aspect_ratio)
        )

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
                    self.internal_messages.reshape((self.n_hidden_vars, -1)),
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
                            *(cmap(int(255 * score))[:-1]),
                            normalised=True
                        ),
                        color=line_color
                    )
                    if show_messages:
                        g.add_node(
                            f'h{var_next}',
                            style='filled',
                            fillcolor=colormap.rgb2hex(
                                *(cmap(int(255 * var_score[var_next]))[:-1]),
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


class DHTM(Layer):
    """
        Distributed Hebbian Temporal Memory.
        see https://arxiv.org/abs/2310.13391
    """
    context_forward_factors: Factors
    context_backward_factors: Factors

    def __init__(
            self,
            n_obs_vars: int,
            n_obs_states: int,
            cells_per_column: int,
            context_factors_conf: dict,
            n_hidden_vars_per_obs_var: int = 1,
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
            cells_activity_lr: float = 0.1,
            use_backward_messages: bool = False,
            posterior_noise: float = 0.001,
            seed: int = None,
            visualization_server=(HOST, PORT),
            visualize=True
    ):
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        if self.seed:
            self._legacy_rng = Random(self.seed)
        else:
            self._legacy_rng = Random()

        self.lr = 1.0
        self.timestep = 1
        self.developmental_period = developmental_period
        self.n_obs_vars = n_obs_vars
        self.n_hidden_vars_per_obs_var = n_hidden_vars_per_obs_var
        self.n_hidden_vars = n_obs_vars * n_hidden_vars_per_obs_var
        self.n_obs_states = n_obs_states
        self.n_external_vars = n_external_vars
        self.n_external_states = n_external_states
        self.n_context_vars = n_context_vars
        self.n_context_states = n_context_states
        self.external_vars_boost = external_vars_boost
        self.unused_vars_boost = unused_vars_boost
        self.cells_activity_lr = cells_activity_lr
        self.use_backward_messages = use_backward_messages
        self.grow_backward_connections = use_backward_messages
        self.posterior_noise = posterior_noise

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

        self.internal_messages = np.zeros(
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
            self.internal_messages
        )

        self.prediction_cells = np.zeros_like(self.internal_messages)
        self.observation_messages = np.zeros(self.input_sdr_size)
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

        context_factors_conf['n_cells'] = self.total_cells
        context_factors_conf['n_vars'] = self.total_vars
        context_factors_conf['n_hidden_states'] = self.n_hidden_states
        context_factors_conf['n_hidden_vars'] = self.n_hidden_vars
        self.context_forward_factors = Factors(**context_factors_conf, seed=self.seed)
        self.context_backward_factors = Factors(**context_factors_conf, seed=self.seed)

        self.cells_to_grow_new_context_segments_forward = np.empty(0)
        self.new_context_segments_forward = np.empty(0)
        self.cells_to_grow_new_context_segments_backward = np.empty(0)
        self.new_context_segments_backward = np.empty(0)

        self.state_information = 0

        self.observation_messages_buffer = list()
        self.external_messages_buffer = list()
        self.forward_messages_buffer = list()
        self.backward_messages_buffer = list()
        self.prior_buffer = list()
        self.can_clear_buffers = False

        # instead of deliberately saving prior
        # we use fixed initial messages
        self.initial_forward_messages = sparse_to_dense(
            np.arange(
                self.n_hidden_vars
            ) * self.n_hidden_states,
            like=self.context_messages
        )
        self.initial_backward_messages = sparse_to_dense(
            np.arange(
                self.n_hidden_vars
            ) * self.n_hidden_states + 1,
            like=self.context_messages
        )
        self.initial_external_messages = sparse_to_dense(
            np.arange(
                self.n_external_vars
            ) * self.n_external_states,
            like=self.external_messages
        )

        self.visualize = visualize
        self.vis_server_address = visualization_server
        self.vis_server = None

        if self.visualize:
            self.connect_to_vis_server()

    def reset(self):
        if self.lr > 0:
            # add last step T messages
            if len(self.observation_messages_buffer) > 0:
                self.observation_messages_buffer.append(self.observation_messages.copy())
                self.external_messages_buffer.append(self.initial_external_messages.copy())
                self.forward_messages_buffer.append(self.internal_messages.copy())
                # for alignment with backward messages
                self.forward_messages_buffer.append(self.initial_backward_messages.copy())

                if self.use_backward_messages:
                    self._backward_pass()
                self._update_segments()

        self.internal_messages = np.zeros(
            self.internal_cells,
            dtype=REAL64_DTYPE
        )
        self.external_messages = self.initial_external_messages.copy()
        self.context_messages = self.initial_forward_messages.copy()

        self.context_active_cells.sparse = []
        self.internal_active_cells.sparse = []
        self.external_active_cells.sparse = []

        self.prediction_cells = np.zeros_like(self.internal_messages)
        self.observation_messages = np.zeros(self.input_sdr_size)
        self.prediction_columns = None

        # attempt to connect to visualization server
        if self.visualize and (self.vis_server is None):
            self.connect_to_vis_server()

        if self.vis_server:
            self._send_json_dict({'type': 'phase', 'phase': 'inference'})

    def clear_buffers(self):
        self.observation_messages_buffer.clear()
        self.external_messages_buffer.clear()
        self.forward_messages_buffer.clear()
        self.backward_messages_buffer.clear()
        self.prior_buffer.clear()

    def predict(self, context_factors=None, **_):
        if context_factors is None:
            context_factors = self.context_forward_factors

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
            context_factors,
            self.inverse_temp_context,
        )

        self.prediction_cells = self.internal_messages.copy()

        self.prediction_columns = self.prediction_cells.reshape(
            -1, self.cells_per_column
        ).sum(axis=-1)

        self.prediction_columns = self.prediction_columns.reshape(
            -1, self.n_hidden_vars_per_obs_var, self.n_obs_states
        ).mean(axis=1).flatten()

    def observe(
            self,
            observation: np.ndarray,
            learn: bool = True
    ):
        """
            observation: pattern in sparse representation
            forward_pass:
            h^k+1_t-1  u_t-1
                    \  |
            h^k_t-1 - [] - h^k_t
        """
        if learn and self.lr > 0:
            if self.can_clear_buffers:
                self.clear_buffers()
                self.can_clear_buffers = False

            # save t prediction (prior)
            self.prior_buffer.append(self.internal_messages.copy())
            # save t-1 messages
            self.observation_messages_buffer.append(self.observation_messages.copy())
            self.external_messages_buffer.append(self.external_messages.copy())
            self.forward_messages_buffer.append(self.context_messages.copy())

        observation_messages = sparse_to_dense(
            observation,
            size=self.input_sdr_size,
            dtype=REAL64_DTYPE
        )

        # update messages
        self.observation_messages = observation_messages
        self.internal_messages = self._get_posterior()

        if self.vis_server is not None:
            self._send_state()

        self.timestep += 1

    def _backward_pass(self):
        if self.vis_server:
            self._send_json_dict({'type': 'phase', 'phase': 'backward_pass'})
        #           u_t   h^k+1_t+1
        #           |   /
        #   h^k_t - [] - h^k_t+1
        self.internal_messages = self.initial_backward_messages.copy()
        self.backward_messages_buffer.append(self.internal_messages.copy())

        T = len(self.observation_messages_buffer)-1
        for t in range(T, 0, -1):
            if t == T:
                self.set_external_messages(self.initial_external_messages.copy())
            else:
                self.set_external_messages(self.external_messages_buffer[t].copy())

            self.set_context_messages(self.internal_messages.copy())
            self.predict(self.context_backward_factors)
            self.observation_messages = self.observation_messages_buffer[t].copy()
            self.internal_messages = self._get_posterior()
            self.backward_messages_buffer.append(self.internal_messages.copy())
            if self.vis_server:
                self._send_state()
        # add forward prior messages and reverse list for alignment with forward messages
        self.backward_messages_buffer.append(self.initial_forward_messages.copy())
        self.backward_messages_buffer = self.backward_messages_buffer[::-1]

    def _update_segments(self):
        if self.vis_server:
            self._send_json_dict({'type': 'phase', 'phase': 'learning'})

        # get prior
        self.set_context_messages(self.initial_forward_messages.copy())
        self.set_external_messages(self.initial_external_messages.copy())
        self.predict()
        self.internal_active_cells.sparse = self._sample_cells(
            self.context_messages.reshape(self.n_hidden_vars, -1)
        )

        for t in range(1, len(self.forward_messages_buffer)):
            if len(self.external_messages) > 0:
                self.external_active_cells.sparse = self._sample_cells(
                    self.external_messages.reshape(self.n_external_vars, -1)
                )
            self.context_active_cells.sparse = self.internal_active_cells.sparse.copy()

            # update forward messages
            if t < len(self.observation_messages_buffer):
                self.set_external_messages(self.external_messages_buffer[t].copy())
                self.observation_messages = self.observation_messages_buffer[t].copy()
                self.context_messages = self._get_posterior()
                self.forward_messages_buffer[t] = self.context_messages.copy()
                # combine forward and backward messages
                if self.use_backward_messages:
                    self.context_messages *= self.backward_messages_buffer[t]
                    self.context_messages = (
                        normalize(self.context_messages.reshape(self.n_hidden_vars, -1))
                    ).flatten()

                self.predict()

                # sample cells
                dependent_cells = self.context_forward_factors.segments_to_receptive_field(
                    self.context_forward_factors.sample_segments()
                )
                context_cells_mask = (
                        (dependent_cells >= self.context_cells_range[0]) &
                        (dependent_cells < self.context_cells_range[1])
                )
                dependent_cells = dependent_cells[context_cells_mask] - self.context_cells_range[0]

                vars_for_dependent_cells = dependent_cells // self.n_hidden_states
                vars_without_cells = np.isin(
                    np.arange(self.n_hidden_vars), vars_for_dependent_cells,
                    assume_unique=True, invert=True
                )

                independent_cells = sample_categorical_variables(
                    self.context_messages.reshape(self.n_hidden_vars, -1)[vars_without_cells],
                    self._rng
                ) + np.flatnonzero(vars_without_cells) * self.n_hidden_states

                self.internal_active_cells.sparse = np.concatenate(
                    [dependent_cells, independent_cells]
                )
            else:
                self.internal_active_cells.sparse = self._sample_cells(
                    self.initial_backward_messages.reshape(
                        self.n_hidden_vars, -1
                    )
                )

            # grow forward connections
            (
                self.cells_to_grow_new_context_segments_forward,
                self.new_context_segments_forward
            ) = self._learn(
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
                (
                    self.internal_cells_range[0] +
                    self.internal_active_cells.sparse
                ),
                self.context_forward_factors,
                prune_segments=(self.timestep % self.developmental_period) == 0
            )

            # grow backward connection symmetrically
            if self.grow_backward_connections:
                (
                    self.cells_to_grow_new_context_segments_backward,
                    self.new_context_segments_backward
                ) = self._learn(
                    np.concatenate(
                        [
                            (
                                self.context_cells_range[0] +
                                self.internal_active_cells.sparse
                            ),
                            (
                                self.external_cells_range[0] +
                                self.external_active_cells.sparse
                            )
                        ]
                    ),
                    (
                        self.internal_cells_range[0] +
                        self.context_active_cells.sparse
                    ),
                    self.context_backward_factors,
                    prune_segments=(self.timestep % self.developmental_period) == 0
                )

            self.internal_cells_activity += self.cells_activity_lr * (
                    self.internal_active_cells.dense - self.internal_cells_activity
            )

            if self.vis_server is not None:
                self._send_state()

        self.can_clear_buffers = True

    def _get_posterior(self):
        observation = np.flatnonzero(self.observation_messages)
        cells = self._get_cells_for_observation(observation)
        obs_factor = sparse_to_dense(cells, like=self.internal_messages)

        messages = self.internal_messages.reshape(self.n_hidden_vars, -1)
        obs_factor = obs_factor.reshape(self.n_hidden_vars, -1)

        messages = normalize((messages + self.posterior_noise / self.cells_per_column) * obs_factor, obs_factor)

        return messages.flatten()

    def _get_cells_for_observation(self, obs_states):
        vars_for_obs_states = obs_states // self.n_obs_states
        obs_states_per_var = obs_states - vars_for_obs_states * self.n_obs_states

        hid_vars = (
                np.tile(np.arange(self.n_hidden_vars_per_obs_var), len(vars_for_obs_states)) +
                vars_for_obs_states * self.n_hidden_vars_per_obs_var
        )
        hid_columns = (
                np.repeat(obs_states_per_var, self.n_hidden_vars_per_obs_var) +
                self.n_obs_states * hid_vars
        )

        all_vars = np.arange(self.n_hidden_vars)
        vars_without_states = all_vars[np.isin(all_vars, hid_vars, invert=True)]

        cells_for_empty_vars = self._get_cells_in_vars(vars_without_states)

        cells_in_columns = (
                (
                        hid_columns * self.cells_per_column
                ).reshape((-1, 1)) +
                np.arange(self.cells_per_column, dtype=UINT_DTYPE)
        ).flatten()

        return np.concatenate([cells_for_empty_vars, cells_in_columns])

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
            num_connected_segment >= factors.n_vars_per_factor
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

            # avoid overflow
            log_prediction_for_cells_with_factors[
                log_prediction_for_cells_with_factors < -100
            ] = -np.inf

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

        self.internal_messages = next_messages

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

        return cells_to_grow_new_segments, new_segments

    def _calculate_learning_segments(self, active_cells, next_active_cells, factors: Factors):
        # determine which segments are learning and growing
        active_cells_sdr = SDR(self.total_cells)
        active_cells_sdr.sparse = active_cells

        num_connected = factors.connections.computeActivity(
            active_cells_sdr,
            False
        )

        active_segments = np.flatnonzero(num_connected >= factors.n_vars_per_factor)

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

    @staticmethod
    def draw_messages(
            messages,
            n_vars,
            figsize=10,
            aspect_ratio=0.3,
            non_zero=True
    ):
        messages = messages.reshape(n_vars, -1)
        n_cols = max(int(np.ceil(np.sqrt(n_vars / aspect_ratio))), 1)
        n_rows = max(int(np.floor(n_cols * aspect_ratio)), 1)

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
            (self.context_forward_factors,),
            ('c',)
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
                    self.internal_messages.reshape((self.n_hidden_vars, -1)),
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

    def hist2d_segments_per_cell(self):
        # segments in use -> cells -> unique counts -> 2d hist
        cells = self.context_forward_factors.connections.mapSegmentsToCells(
            self.context_forward_factors.segments_in_use
        )

        cells, counts = np.unique(cells, return_counts=True)

        segments_per_cell = np.zeros(self.internal_cells)
        segments_per_cell[cells - self.internal_cells_range[0]] = counts
        return segments_per_cell.reshape(-1, self.cells_per_column).T

    def hist2d_new_segments_cells(self):
        cells_to_grow_segments = np.zeros(self.internal_cells)
        cells_to_grow_segments[
            self.cells_to_grow_new_context_segments_forward - self.internal_cells_range[0]
        ] = 1

        return cells_to_grow_segments.reshape(-1, self.cells_per_column).T

    def hist2d_new_segments_receptive_fields(self):
        receptive_fields = self.context_forward_factors.receptive_fields[
            self.new_context_segments_forward
        ]
        cells, counts = np.unique(receptive_fields.flatten(), return_counts=True)

        context_mask = (
                (self.context_cells_range[0] <= cells) &
                (cells < self.context_cells_range[1])
        )
        external_mask = (
                (self.external_cells_range[0] <= cells) &
                (cells < self.external_cells_range[1])
        )

        context_cells_counts = np.zeros(self.context_input_size)
        context_cells_counts[
            cells[context_mask] - self.context_cells_range[0]
        ] = counts[context_mask]
        context_cells_counts = context_cells_counts.reshape(-1, self.cells_per_column).T

        external_cells_counts = np.zeros(self.external_input_size)
        external_cells_counts[
            cells[external_mask] - self.external_cells_range[0]
        ] = counts[external_mask]
        external_cells_counts = external_cells_counts.reshape(1, -1)
        return context_cells_counts, external_cells_counts

    def connect_to_vis_server(self):
        self.vis_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        try:
            self.vis_server.connect(self.vis_server_address)
            # handshake
            self._send_json_dict({'type': 'hello'})
            data = get_data(self.vis_server)
            print(data)

            if data != 'dhtm':
                raise socket.error(
                    f'Handshake failed {self.vis_server_address}: It is not DHTM vis server!'
                )
            print(f'Connected to visualization server {self.vis_server_address}!')
        except socket.error as msg:
            self.vis_server.close()
            self.vis_server = None
            print(f'Failed to connect to the visualization server: {msg}. Proceed.')

    def _send_state(self):
        data = get_data(self.vis_server)
        if data == 'skip':
            self._send_json_dict({'type': 'skip'})
        elif data == 'close':
            self.vis_server.close()
            self.vis_server = None
            print('Server shutdown. Proceed.')
        elif data == 'step':
            data_dict = {'type': 'state'}
            data_dict['external_messages'] = self.external_messages.reshape(1, -1)
            data_dict['context_messages'] = self.context_messages.reshape(
                -1, self.cells_per_column
            ).T
            data_dict['prediction_cells'] = self.prediction_cells.reshape(
                -1, self.cells_per_column
            ).T
            data_dict['internal_messages'] = self.internal_messages.reshape(
                -1, self.cells_per_column
            ).T
            data_dict['prediction_columns'] = self.prediction_columns.reshape(
                1, -1
            )
            data_dict['observation_messages'] = self.observation_messages.reshape(
                1, -1
            )
            data_dict['segments_per_cell'] = self.hist2d_segments_per_cell()
            if len(self.new_context_segments_forward) > 0:
                data_dict['cells_for_new_segments'] = self.hist2d_new_segments_cells()
                (
                    data_dict['context_fields_for_new_segments'],
                    data_dict['external_fields_for_new_segments']
                ) = self.hist2d_new_segments_receptive_fields()

            self._send_json_dict(data_dict)

    def _send_json_dict(self, data_dict):
        send_string(json.dumps(data_dict, cls=NumpyEncoder), self.vis_server)
