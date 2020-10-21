import math
import warnings
from typing import Callable

import torch


class LambdaLRWithRamp(torch.optim.lr_scheduler.LambdaLR):

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        return [lmbda(self.last_epoch, base_lr) for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)]

    @staticmethod
    def get_lr_with_ramp(num_steps: int, rampdown: float = 0.25, rampup: float = 0.05) -> Callable[[int, float], float]:
        def schedule_lr(iteration, initial_lr):
            t = iteration / num_steps
            lr_ramp = min(1, (1 - t) / rampdown) if rampdown != 0 else 1
            lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
            lr_ramp = lr_ramp * (min(1, t / rampup) if rampup != 0 else 1)

            return initial_lr * lr_ramp
        return schedule_lr
