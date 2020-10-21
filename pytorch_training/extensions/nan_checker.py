import sys

import torch

from pytorch_training.trainer import Trainer
from pytorch_training.extension import Extension


class NaNChecker(Extension):

    def __init__(self, network: torch.nn.Module, network_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.network = network
        self.network_name = network_name

    def run(self, trainer: Trainer):
        nan_detected = False
        for name, param in self.network.named_parameters(prefix=self.network_name):
            if (param.data != param.data).any():
                nan_detected = True
                print(f"param {name} contains NaN at iteration {trainer.updater.iteration}!")
            if param.grad is not None and (param.grad.data != param.grad.data).any():
                nan_detected = True
                print(f"grad of param {name} contains NaN at iteration {trainer.updater.iteration}!")

        if nan_detected:
            sys.exit("NaN detected")
