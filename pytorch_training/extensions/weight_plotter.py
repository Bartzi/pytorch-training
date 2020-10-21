import torch

from pytorch_training.extension import Extension
from pytorch_training.extensions.logger import Logger
from pytorch_training.trainer import Trainer


class WeightPlotter(Extension):

    def __init__(self, network: torch.nn.Module, logger: Logger, prefix: str = '', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.network = network
        self.prefix = prefix
        self.logger = logger

    def run(self, trainer: Trainer):
        for name, param in self.network.named_parameters():
            self.logger.save_histograms(
                trainer,
                [{
                    "iteration": trainer.updater.iteration,
                    '/'.join([self.prefix, name]): param.data.cpu().numpy()
                }]
            )
            if param.grad is not None:
                self.logger.save_histograms(
                    trainer,
                    [{
                        "iteration": trainer.updater.iteration,
                        '/'.join([self.prefix, 'grad', name]): param.grad.data.cpu().numpy()
                    }]
                )
