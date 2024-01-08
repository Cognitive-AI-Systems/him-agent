#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import os
import sys
import yaml
import os
import sys
from scipy.special import rel_entr

import numpy as np

from hima.common.config.base import read_config, override_config
from hima.common.lazy_imports import lazy_import
from hima.common.run.argparse import parse_arg_list
from hima.modules.belief.cortial_column.cortical_column import CorticalColumn, Layer
from hima.experiments.successor_representations.runners.utils import make_decoder
from hima.common.sdr import sparse_to_dense
from hima.envs.mpg.mpg import MultiMarkovProcessGrammar
from hima.common.metrics import get_surprise

from typing import Literal

wandb = lazy_import('wandb')


class MPGTest:
    def __init__(self, logger, conf):
        
        self.seed = conf['run'].get('seed')
        self._rng = np.random.default_rng(self.seed)
        conf['mpg']['seed'] = self.seed
        conf['mpg']['initial_policy'] = 2
        self.logger = logger
        self.raw_obs_shape = conf['run']['raw_obs_shape']
        self.mpg = MultiMarkovProcessGrammar(**conf['mpg'])
        self.smf_dist = conf['run']['smf_dist']
        self.layer_type = 'dhtm'
        self.n_episodes = conf['run']['n_episodes']
        self.coders_policy = conf['run']['coders_policy']
        self.coders = conf['run']['coders']
    
        if 'reset_context_period' in conf['layer']:
            self.reset_context_period = conf['layer'].pop(
                'reset_context_period'
            )
        else:
            self.reset_context_period = 0

        self.cortical_column = self.make_cortical_column(
            conf,
            conf['run'].get('saved_model_path', None)
        )

        self.initial_context = sparse_to_dense(
            np.arange(
                self.cortical_column.layer.n_hidden_vars
            ) * self.cortical_column.layer.n_hidden_states,
            like=self.cortical_column.layer.context_messages
        )
        self.initial_external_message = None

        if self.logger is not None:
            from hima.common.metrics import ScalarMetrics
            basic_scalar_metrics = {
                'main_metrics/surprise_hidden': np.mean
            }
            basic_scalar_metrics['main_metrics/n_segments'] = np.mean

            self.scalar_metrics = ScalarMetrics(
                basic_scalar_metrics,
                self.logger
            )


    def encode_letter(self, letter, noisy_var=True):
        letter_to_numbers = self.coders[self.coders_policy]
        number_code = letter_to_numbers[letter]
        if noisy_var:
            number_code = np.append(number_code, np.random.choice([6,7,8]))
            return number_code
        else:
            return number_code
        
    def decode_letter(self, prediction):
        probs = np.zeros(self.raw_obs_shape)
        letter_to_numbers = self.coders[self.coders_policy]

        for i, (key, value) in enumerate(letter_to_numbers.items()):
            probs[i] = prediction[value[0]] * prediction[value[1]]

        return probs / np.sum(probs)

    def run(self):

        dist = np.zeros((len(self.mpg.states), len(self.mpg.alphabet) + 1))
        dist_disp = np.zeros((len(self.mpg.states), len(self.mpg.alphabet) + 1))
        true_dist = np.array([self.mpg.predict_letters(from_state=i) for i in self.mpg.states])

        norm = true_dist.sum(axis=-1)
        empty_prob = np.clip(1 - norm, 0, 1)
        true_dist = np.hstack([true_dist, empty_prob.reshape(-1, 1)])

        total_dkl = 0

        for i in range(self.n_episodes):
            self.mpg.reset()
            self.cortical_column.reset(self.initial_context, self.initial_external_message)
            steps = 0
            dkls = []

            while True:
                prev_state = self.mpg.current_state

                if (self.reset_context_period > 0) and (steps > 0):
                    if (steps % self.reset_context_period) == 0:
                        self.cortical_column.layer.set_context_messages(
                            self.initial_context
                        )
              
                letter = self.mpg.next_state()
                
                if letter is None:
                    break
                else:
                    #obs_state = np.array([self.mpg.char_to_num[letter]])
                    obs_state = self.encode_letter(letter)
                    events = np.array(obs_state)

                self.cortical_column.observe(events, None, learn=True)
                column_probs = self.decode_letter(self.cortical_column.layer.prediction_columns)

                if column_probs is None:
                    column_probs = np.ones(self.raw_obs_shape) / self.raw_obs_shape
                
                # >>> logging
                if self.logger is not None:
                    surprise = get_surprise(
                        self.cortical_column.layer.prediction_columns[:6], obs_state[:2], mode='bernoulli'
                    )
                    scalar_metrics_update = {
                        'main_metrics/surprise_hidden': surprise,
                    }
                    scalar_metrics_update['main_metrics/n_segments'] = (
                        self.cortical_column.layer.
                        context_factors.connections.numSegments()
                    )

                    self.scalar_metrics.update(
                        scalar_metrics_update
                    )
                
                column_probs = np.append(
                    column_probs, np.clip(1 - column_probs.sum(), 0, 1)
                )

                delta = column_probs - dist[prev_state]
                dist_disp[prev_state] += self.smf_dist * (
                        np.power(delta, 2) - dist_disp[prev_state])
                dist[prev_state] += self.smf_dist * delta

                if prev_state != 0:
                    dkl = min(
                            rel_entr(true_dist[prev_state], column_probs).sum(),
                            200.0
                        )
                    dkls.append(dkl)
                    total_dkl += dkl  
                # <<< logging

                steps += 1

            # >>> logging
            if self.logger is not None:
                self.scalar_metrics.log(i)
            if self.logger is not None:
                self.logger.log(
                    {
                        'main_metrics/dkl': np.array(np.abs(dkls)).mean(),
                        'main_metrics/total_dkl': total_dkl,
                    }, step=i
                )
            # <<< logging


    def make_cortical_column(self, conf=None, saved_model_path=None):
        if saved_model_path is not None:
            raise NotImplementedError
        elif conf is not None:
            layer_type = conf['run']['layer']
  
            encoder_type = conf['run']['encoder']
            encoder_conf = conf['encoder']
            layer_conf = conf['layer']
            seed = conf['run']['seed']

            if encoder_type == 'sp_ensemble':
                from hima.modules.htm.spatial_pooler import SPDecoder, SPEnsemble

                encoder_conf['seed'] = seed
                encoder_conf['inputDimensions'] = list(self.raw_obs_shape)

                encoder = SPEnsemble(**encoder_conf)
                decoder = SPDecoder(encoder)
            elif encoder_type == 'sp_grouped':
                from hima.experiments.temporal_pooling.stp.sp_ensemble import (
                    SpatialPoolerGroupedWrapper
                )
                encoder_conf['seed'] = seed
                encoder_conf['feedforward_sds'] = [self.raw_obs_shape, 0.1]

                decoder_type = conf['run'].get('decoder', None)
                decoder_conf = conf['decoder']

                encoder = SpatialPoolerGroupedWrapper(**encoder_conf)
                decoder = make_decoder(encoder, decoder_type, decoder_conf)
            else:
                raise ValueError(f'Encoder type {encoder_type} is not supported')

            layer_conf['n_obs_vars'] = encoder.n_groups
            layer_conf['n_obs_states'] = encoder.getSingleNumColumns()
            layer_conf['n_external_states'] = 0
            layer_conf['seed'] = seed

            layer_conf['n_context_states'] = (
                    encoder.getSingleNumColumns() * layer_conf['cells_per_column']
            )
            layer_conf['n_context_vars'] = encoder.n_groups
            layer_conf['n_external_vars'] = 0
            layer = Layer(**layer_conf)

            encoder = None
            decoder = None
            cortical_column = CorticalColumn(
                layer,
                encoder,
                decoder
            )
        else:
            raise ValueError

        return cortical_column
    
    
def main(config_path):
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    config = dict()

    config['run'] = read_config(config_path)

    layer_conf_path = config['run']['layer_conf']
    config['run']['layer'] = layer_conf_path.split('/')[-2]
    config['layer'] = read_config(config['run']['layer_conf'])

    if 'encoder_conf' in config['run']:
        config['encoder'] = read_config(config['run']['encoder_conf'])

    if 'decoder_conf' in config['run']:
        config['decoder'] = read_config(config['run']['decoder_conf'])

    overrides = parse_arg_list(sys.argv[2:])
    override_config(config, overrides)

    if config['run']['seed'] is None:
        config['run']['seed'] = np.random.randint(0, np.iinfo(np.int32).max)

    if config['run']['log']:
        import wandb
        logger = wandb.init(
            project=config['run']['project_name'], entity=os.environ.get('WANDB_ENTITY', None),
            config=config
        )
    else:
        logger = None

    with open(config['run']['mpg_conf'], 'r') as file:
        config['mpg'] = yaml.load(file, Loader=yaml.Loader)

    runner = MPGTest(logger, config)
    runner.run()


if __name__ == '__main__':
    default_config = 'hima/experiments/sequence/configs/runner/dhtm/mpg_dhtm.yaml'
    main(os.environ.get('RUN_CONF', default_config))