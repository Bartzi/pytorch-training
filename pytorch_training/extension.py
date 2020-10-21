from typing import TYPE_CHECKING

from pytorch_training.triggers import get_trigger

if TYPE_CHECKING:
    from pytorch_training.trainer import Trainer
    from pytorch_training.trigger import Trigger


class Extension:

    def __init__(self, trigger: 'Trigger' = (1, 'epoch')):
        self.trigger = get_trigger(trigger)

    def __call__(self, trainer: 'Trainer'):
        if self.trigger(trainer):
            self.run(trainer)

    def initialize(self, trainer: 'Trainer'):
        pass

    def finalize(self, trainer: 'Trainer'):
        pass

    def run(self, trainer: 'Trainer'):
        raise NotImplementedError
