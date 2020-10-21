import time

from torch.utils.data import DataLoader

from pytorch_training.trainer import Trainer
from pytorch_training.extension import Extension


class SeedSetter(Extension):

    def __init__(self, dataloader: DataLoader, *args, **kwargs):
        self.dataloader = dataloader
        super().__init__(*args, **kwargs)

    def run(self, trainer: Trainer):
        if hasattr(self.dataloader.sampler, 'set_epoch'):
            seed = time.time_ns()
            self.dataloader.sampler.set_epoch(seed)
