#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from hima.common.config.base import TConfig
from hima.common.config.global_config import GlobalConfig
from hima.common.run.wandb import get_logger
from hima.common.timer import timer, print_with_timestamp
from hima.common.utils import timed
from hima.experiments.temporal_pooling.data.synthetic_sequences import Sequence
from hima.experiments.temporal_pooling.graph.model import Model
from hima.experiments.temporal_pooling.graph.model_compiler import ModelCompiler
from hima.experiments.temporal_pooling.iteration import IterationConfig
from hima.experiments.temporal_pooling.resolvers.type_resolver import StpLazyTypeResolver
from hima.experiments.temporal_pooling.run_progress import RunProgress
from hima.experiments.temporal_pooling.utils import resolve_random_seed, scheduled

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run


class StpExperiment:
    config: GlobalConfig
    logger: Run | None
    init_time: float

    seed: int

    model: Model
    iterate: IterationConfig

    def __init__(
            self, config: TConfig, config_path: Path,
            log: bool, seed: int,
            iterate: TConfig, data: TConfig,
            model: TConfig,
            # track_streams: TConfig, stats_and_metrics: TConfig, diff_stats: TConfig,
            log_schedule: TConfig,
            project: str = None,
            **_
    ):
        self.init_time = timer()
        self.print_with_timestamp('==> Init')

        self.config = GlobalConfig(
            config=config, config_path=config_path, type_resolver=StpLazyTypeResolver()
        )
        self.logger = get_logger(config, log=log, project=project)
        self.seed = resolve_random_seed(seed)

        self.iterate = self.config.resolve_object(iterate, object_type_or_factory=IterationConfig)
        self.data = self.config.resolve_object(
            data,
            n_sequences=self.iterate.sequences,
            sequence_length=self.iterate.elements
        )
        model_compiler = ModelCompiler(self.config)
        self.model = self.config.resolve_object(model, object_type_or_factory=model_compiler.parse)

        # propagate data SDS to the model graph
        self.model.streams['input.sdr'].set_sds(self.data.sds)
        model_compiler.compile(self.model)
        print(self.model)
        print()

        self.progress = RunProgress()
        # stats_and_metrics = self.config.resolve_object(
        #     stats_and_metrics, object_type_or_factory=StatsMetricsConfig
        # )
        # self.stats = ExperimentStats(
        #     n_sequences=self.iterate.sequences, progress=self.progress, logger=self.logger,
        #     blocks=self.model.blocks, track_streams=track_streams, stats_config=stats_and_metrics,
        #     diff_stats=diff_stats
        # )
        self.log_schedule = log_schedule

    def run(self):
        self.print_with_timestamp('==> Run')
        # self.stats.define_metrics()

        for epoch in range(self.iterate.epochs):
            _, elapsed_time = self.train_epoch()
            self.print_with_timestamp(f'Epoch {epoch}')
        self.print_with_timestamp('<==')

    @timed
    def train_epoch(self):
        self.progress.next_epoch()
        # self.stats.on_epoch_started()

        # noinspection PyTypeChecker
        for sequence in self.data:
            for i_repeat in range(self.iterate.sequence_repeats):
                self.run_sequence(sequence, i_repeat, learn=True)
            # self.stats.on_sequence_finished()

        epoch_final_log_scheduled = scheduled(
            i=self.progress.epoch, schedule=self.log_schedule['epoch'],
            always_report_first=True, always_report_last=True, i_max=self.iterate.epochs
        )
        # self.stats.on_epoch_finished(epoch_final_log_scheduled)

        # blocks = self.pipeline.blocks
        # sp = blocks['sp2'].sp if 'sp2' in blocks else blocks['sp1']
        # print(f'{round(sp.n_computes / sp.run_time / 1000, 2)} kcps')
        # print(.sp.activation_entropy())
        # print('_____')

    def run_sequence(self, sequence: Sequence, i_repeat: int = 0, learn=True):
        self.reset_blocks('temporal_memory', 'temporal_pooler')

        log_scheduled = scheduled(
            i=i_repeat, schedule=self.log_schedule['repeat'],
            always_report_first=True, always_report_last=True, i_max=self.iterate.sequence_repeats
        )
        # self.stats.on_sequence_started(sequence.id, log_scheduled)

        for _, input_sdr in enumerate(sequence):
            self.reset_blocks('spatial_pooler', 'custom_sp')
            for _ in range(self.iterate.element_repeats):
                self.progress.next_step()
                self.model.streams['input.sdr'].set(input_sdr)
                self.model.forward()
                # self.stats.on_step()

    def reset_blocks(self, *blocks_family):
        blocks_family = set(blocks_family)
        for name in self.model.blocks:
            block = self.model.blocks[name]
            if block.family in blocks_family:
                block.reset()

    def print_with_timestamp(self, text: str):
        print_with_timestamp(text, self.init_time)
