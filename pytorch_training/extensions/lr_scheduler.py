from typing import Dict

from torch.optim.lr_scheduler import _LRScheduler

from pytorch_training import Extension, Trainer
from pytorch_training.reporter import get_current_reporter


class LRScheduler(Extension):

    def __init__(self, schedulers: Dict[str, _LRScheduler], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.schedulers = schedulers

    def initialize(self, trainer: Trainer):
        for name, scheduler in self.schedulers.items():
            self.log_lr(name, scheduler)

    def log_lr(self, scheduler_name: str, scheduler: _LRScheduler):
        for i, param_group in enumerate(scheduler.optimizer.param_groups):
            lr = param_group['lr']
            suffix = f"/{i}" if len(scheduler.optimizer.param_groups) > 1 else ""
            with get_current_reporter() as reporter:
                reporter.add_observation({f"lr/{scheduler_name}{suffix}": lr}, prefix='metrics')

    def run(self, trainer: Trainer):
        for name, scheduler in self.schedulers.items():
            scheduler.step()
            self.log_lr(name, scheduler)

