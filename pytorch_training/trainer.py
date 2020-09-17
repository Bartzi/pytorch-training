import math
from functools import partial
from typing import Iterable

from tqdm import trange

from pytorch_training.distributed import synchronize, get_rank, get_world_size
from pytorch_training.extension import Extension
from pytorch_training.reporter import get_current_reporter
from pytorch_training.triggers import IntervalTrigger
from pytorch_training.updater import Updater


class Trainer:

    def __init__(self, updater: Updater, stop_trigger: IntervalTrigger) -> None:
        self.updater = updater
        self.stop_trigger = stop_trigger
        self.extensions = []

    def extend(self, extension: Extension) -> None:
        assert isinstance(extension, Extension), "A Trainer extension must be a subclass of Extension!"
        self.extensions.append(extension)

    def init_extensions(self) -> None:
        for extension in self.extensions:
            extension.initialize(self)

    def run_extensions(self) -> None:
        for extension in self.extensions:
            extension(self)

    def finalize_extensions(self) -> None:
        for extension in self.extensions:
            extension.finalize(self)

    def get_progressbar(self, start: int, end: int = None, step: int = 1, **kwargs) -> Iterable:
        if end is None:
            end = start
            start = 0
        return trange(start, end, step, **kwargs)

    @property
    def num_epochs(self) -> int:
        num_remaining_epochs = self.stop_trigger.period - self.updater.current_epoch
        if self.stop_trigger.unit == 'epoch':
            return num_remaining_epochs
        else:
            return max(math.ceil(num_remaining_epochs / self.updater.epoch_length), 1)

    @property
    def iterations_per_epoch(self) -> int:
        num_iterations_in_epoch = self.updater.epoch_length - self.updater.iteration_in_epoch
        if self.stop_trigger.unit == 'epoch':
            return num_iterations_in_epoch
        else:
            return min(self.stop_trigger.period - self.updater.iteration, self.updater.epoch_length)

    def run_training(self):
        reporter = get_current_reporter()

        for _ in self.get_progressbar(self.num_epochs, desc='epoch'):
            self.updater.reset()
            for __ in self.get_progressbar(self.iterations_per_epoch, leave=False, desc='iteration'):
                with reporter:
                    self.updater.update()

                self.run_extensions()

                if self.stop_trigger(self):
                    return

    def train(self) -> None:
        self.init_extensions()
        self.run_training()
        self.finalize_extensions()


class DistributedTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank = get_rank()
        self.world_size = get_world_size()

    def init_extensions(self) -> None:
        super().init_extensions()
        synchronize()

    def run_extensions(self) -> None:
        super().run_extensions()
        synchronize()

    def finalize_extensions(self) -> None:
        super().finalize_extensions()
        synchronize()

    def get_progressbar(self, start: int, end: int = None, step: int = 1, **kwargs) -> Iterable:
        if self.rank == 0:
            range_fun = partial(trange, **kwargs)
        else:
            range_fun = range

        if end is None:
            end = start
            start = 0

        return range_fun(start, end, step)
