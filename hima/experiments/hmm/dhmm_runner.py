#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from hima.modules.htm.dchmm import DCHMM
from hima.envs.mpg import MarkovProcessGrammar, MultiMarkovProcessGrammar
import numpy as np
from scipy.special import rel_entr
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import yaml
import os
import sys
import ast
import pickle


class MPGTest:
    def __init__(self, logger, conf):
        self.seed = conf['run']['seed']

        conf['hmm']['seed'] = self.seed
        conf['mpg']['seed'] = self.seed

        self.mpg = MarkovProcessGrammar(**conf['mpg'])

        conf['hmm']['n_obs_states'] = len(self.mpg.alphabet)
        self.hmm = DCHMM(**conf['hmm'])

        self.n_episodes = conf['run']['n_episodes']
        self.smf_dist = conf['run']['smf_dist']
        self.log_update_rate = conf['run']['update_rate']
        self.save_model = conf['run']['save_model']
        self.logger = logger

    def run(self):
        dist = np.zeros((len(self.mpg.states), len(self.mpg.alphabet)))
        dist_disp = np.zeros((len(self.mpg.states), len(self.mpg.alphabet)))
        true_dist = np.array([self.mpg.predict_letters(from_state=i) for i in self.mpg.states])

        total_surprise = 0
        total_dkl = 0
        for i in range(self.n_episodes):
            self.mpg.reset()
            self.hmm.reset()

            dkls = []
            surprises = []

            while True:
                prev_state = self.mpg.current_state

                letter = self.mpg.next_state()

                if letter is None:
                    break
                else:
                    obs_state = np.array(
                        [self.mpg.char_to_num[letter]]
                    )

                self.hmm.predict_cells()
                column_probs = self.hmm.predict_columns()
                self.hmm.observe(obs_state, learn=True)

                # metrics
                # 1. surprise
                active_columns = np.arange(self.hmm.n_columns) == obs_state
                surprise = - np.sum(np.log(column_probs[active_columns]))
                surprise += - np.sum(np.log(1 - column_probs[~active_columns]))

                surprises.append(surprise)
                total_surprise += surprise

                # 2. distribution
                delta = column_probs - dist[prev_state]
                dist_disp[prev_state] += self.smf_dist * (
                        np.power(delta, 2) - dist_disp[prev_state])
                dist[prev_state] += self.smf_dist * delta

                # 3. Kl distance
                dkl = min(
                        rel_entr(true_dist[prev_state], column_probs).sum(),
                        200.0
                    )
                dkls.append(dkl)
                total_dkl += dkl

            if self.logger is not None:
                self.logger.log(
                    {
                        'main_metrics/surprise': np.array(surprises).mean(),
                        'main_metrics/dkl': np.array(np.abs(dkls)).mean(),
                        'main_metrics/total_surprise': total_surprise,
                        'main_metrics/total_dkl': total_dkl,
                        'connections/n_segments': self.hmm.connections.numSegments()
                    }, step=i
                )

                if (self.log_update_rate is not None) and (i % self.log_update_rate == 0):
                    kl_divs = rel_entr(true_dist, dist).sum(axis=-1)

                    n_states = len(self.mpg.states)
                    k = int(np.ceil(np.sqrt(n_states)))
                    fig, axs = plt.subplots(k, k)
                    fig.tight_layout(pad=3.0)

                    for n in range(n_states):
                        ax = axs[n // k][n % k]
                        ax.grid()
                        ax.set_ylim(0, 1)
                        ax.set_title(
                            f's: {n}; ' + '$D_{KL}$: ' + f'{np.round(kl_divs[n], 2)}'
                        )
                        ax.bar(
                            np.arange(dist[n].shape[0]),
                            dist[n],
                            tick_label=self.mpg.alphabet,
                            label='TM',
                            color=(0.7, 1.0, 0.3),
                            capsize=4,
                            ecolor='#2b4162',
                            yerr=np.sqrt(dist_disp[n])
                        )
                        ax.bar(
                            np.arange(dist[n].shape[0]),
                            true_dist[n],
                            tick_label=self.mpg.alphabet,
                            color='#8F754F',
                            alpha=0.6,
                            label='True'
                        )

                        fig.legend(['Predicted', 'True'], loc=8)

                        self.logger.log(
                            {'density/letter_predictions': wandb.Image(fig)}, step=i
                        )

                        plt.close(fig)

        if self.logger is not None and self.save_model:
            name = self.logger.name
            with open(f"logs/model_{name}.pkl", 'wb') as file:
                pickle.dump((self.mpg, self.hmm), file)


class MMPGTest:
    def __init__(self, logger, conf):
        self.seed = conf['run']['seed']

        conf['hmm']['seed'] = self.seed
        conf['mpg']['seed'] = self.seed

        self.mpg = MultiMarkovProcessGrammar(**conf['mpg'])

        self.n_policies = self.mpg.policy_transition_probs.shape[0]
        self.n_obs_states = len(self.mpg.alphabet)
        conf['hmm']['n_obs_states'] = max(self.n_obs_states, self.n_policies)

        self.hmm = DCHMM(**conf['hmm'])

        self.n_episodes = conf['run']['n_episodes']
        self.smf_dist = conf['run']['smf_dist']
        self.log_update_rate = conf['run']['update_rate']
        self.save_model = conf['run']['save_model']
        self.logger = logger

    def run(self):
        dist = np.zeros((self.n_policies, len(self.mpg.states), len(self.mpg.alphabet)))
        dist_disp = np.zeros((self.n_policies, len(self.mpg.states), len(self.mpg.alphabet)))

        true_dist = np.zeros((self.n_policies, len(self.mpg.states), len(self.mpg.alphabet)))
        for pol in range(self.n_policies):
            self.mpg.set_policy(pol)
            true_dist[pol] = np.array(
                [self.mpg.predict_letters(from_state=i) for i in self.mpg.states]
            )

        total_surprise = 0
        total_dkl = 0
        for i in range(self.n_episodes):
            self.mpg.reset()
            self.hmm.reset()

            dkls = []
            surprises = []

            policy = self.mpg.rng.integers(self.n_policies)
            self.mpg.set_policy(policy)

            while True:
                prev_state = self.mpg.current_state

                letter = self.mpg.next_state()

                if letter is None:
                    break
                else:
                    obs_state = np.array(
                        [
                            self.mpg.char_to_num[letter],
                            policy + self.hmm.n_obs_states
                        ]
                    )

                self.hmm.predict_cells()
                column_probs = self.hmm.predict_columns()[:self.n_obs_states]
                self.hmm.observe(obs_state, learn=True)

                # metrics
                # 1. surprise
                active_columns = np.arange(self.hmm.n_obs_states) == obs_state[0]
                surprise = - np.sum(np.log(column_probs[active_columns]))
                surprise += - np.sum(np.log(1 - column_probs[~active_columns]))

                surprises.append(surprise)
                total_surprise += surprise

                # 2. distribution
                delta = column_probs - dist[policy][prev_state]
                dist_disp[policy][prev_state] += self.smf_dist * (
                        np.power(delta, 2) - dist_disp[policy][prev_state])
                dist[policy][prev_state] += self.smf_dist * delta

                # 3. Kl distance
                dkl = min(
                        rel_entr(true_dist[policy][prev_state], column_probs).sum(),
                        200.0
                    )
                dkls.append(dkl)
                total_dkl += dkl

            if self.logger is not None:
                self.logger.log(
                    {
                        'main_metrics/surprise': np.array(surprises).mean(),
                        'main_metrics/dkl': np.array(np.abs(dkls)).mean(),
                        'main_metrics/total_surprise': total_surprise,
                        'main_metrics/total_dkl': total_dkl,
                        'connections/n_segments': self.hmm.connections.numSegments()
                    }, step=i
                )

                if (self.log_update_rate is not None) and (i % self.log_update_rate == 0):
                    kl_divs = rel_entr(true_dist, dist).sum(axis=-1)

                    n_states = len(self.mpg.states)
                    k = int(np.ceil(np.sqrt(n_states)))

                    for pol in range(self.n_policies):
                        fig, axs = plt.subplots(k, k)
                        fig.tight_layout(pad=3.0)

                        for n in range(n_states):
                            ax = axs[n // k][n % k]
                            ax.grid()
                            ax.set_ylim(0, 1)
                            ax.set_title(
                                f's: {n}; ' + '$D_{KL}$: ' + f'{np.round(kl_divs[pol][n], 2)}'
                            )
                            ax.bar(
                                np.arange(dist[pol][n].shape[0]),
                                dist[pol][n],
                                tick_label=self.mpg.alphabet,
                                label='TM',
                                color=(0.7, 1.0, 0.3),
                                capsize=4,
                                ecolor='#2b4162',
                                yerr=np.sqrt(dist_disp[pol][n])
                            )
                            ax.bar(
                                np.arange(dist[pol][n].shape[0]),
                                true_dist[pol][n],
                                tick_label=self.mpg.alphabet,
                                color='#8F754F',
                                alpha=0.6,
                                label='True'
                            )

                            fig.legend(['Predicted', 'True'], loc=8)

                            self.logger.log(
                                {f'density/letter_predictions_policy_{pol}': wandb.Image(fig)},
                                step=i
                            )

                            plt.close(fig)

        if self.logger is not None and self.save_model:
            name = self.logger.name
            with open(f"logs/model_{name}.pkl", 'wb') as file:
                pickle.dump((self.mpg, self.hmm), file)


class NStepTest:
    def __init__(self, logger, conf, model):
        self.mpg, self.hmm = model


def main(config_path):
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    config = dict()

    with open(config_path, 'r') as file:
        config['run'] = yaml.load(file, Loader=yaml.Loader)

    with open(config['run']['hmm_conf'], 'r') as file:
        config['hmm'] = yaml.load(file, Loader=yaml.Loader)

    with open(config['run']['mpg_conf'], 'r') as file:
        config['mpg'] = yaml.load(file, Loader=yaml.Loader)

    for arg in sys.argv[2:]:
        key, value = arg.split('=')

        try:
            value = ast.literal_eval(value)
        except ValueError:
            ...

        key = key.lstrip('-')
        if key.endswith('.'):
            # a trick that allow distinguishing sweep params from config params
            # by adding a suffix `.` to sweep param - now we should ignore it
            key = key[:-1]
        tokens = key.split('.')
        c = config
        for k in tokens[:-1]:
            if not k:
                # a trick that allow distinguishing sweep params from config params
                # by inserting additional dots `.` to sweep param - we just ignore it
                continue
            if 0 in c:
                k = int(k)
            c = c[k]
        c[tokens[-1]] = value

    if config['run']['log']:
        logger = wandb.init(
            project=config['run']['project_name'], entity=os.environ['WANDB_ENTITY'],
            config=config
        )
    else:
        logger = None

    experiment = config['run']['experiment']

    if experiment == 'mpg':
        runner = MPGTest(logger, config)
    elif experiment == 'mmpg':
        runner = MMPGTest(logger, config)
    else:
        raise ValueError

    runner.run()


if __name__ == '__main__':
    main('configs/dhmm_runner.yaml')
