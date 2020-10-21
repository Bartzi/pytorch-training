from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytorch_training import Trainer, Updater
from pytorch_training.trigger import Trigger


class IntervalTrigger(Trigger):

    def __init__(self, period: int, unit: str):
        if unit not in ['iteration', 'epoch']:
            raise ValueError("Unit for Interval Trigger can only supplied as 'iteration', or 'epoch'")

        self.period = period
        self.unit = unit

        self.last_iteration = 0
        self.last_epoch_detail = 0

    def decide_for_epoch(self, updater: 'Updater') -> bool:
        epoch_detail = updater.epoch_detail
        last_epoch_detail = self.last_epoch_detail

        fire = last_epoch_detail // self.period != \
            epoch_detail // self.period

        self.last_epoch_detail = epoch_detail
        return fire

    def decide_for_iteration(self, updater: 'Updater') -> bool:
        iteration = updater.iteration
        last_iteration = self.last_iteration

        # if previous_iteration is invalid value,
        # guess it from current iteration.
        if last_iteration < 0:
            last_iteration = iteration - 1

        fire = last_iteration // self.period != \
            iteration // self.period

        self.last_iteration = iteration
        return fire

    def __call__(self, trainer: 'Trainer') -> bool:
        updater = trainer.updater
        if self.unit == 'epoch':
            fire = self.decide_for_epoch(updater)
        else:
            fire = self.decide_for_iteration(updater)
        return fire
