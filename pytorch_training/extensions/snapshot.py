import os

import torch

from pytorch_training.trainer import Trainer
from pytorch_training.extension import Extension


class Snapshotter(Extension):

    def __init__(self, modules_to_save: dict, log_dir: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.modules_to_save = modules_to_save
        self.log_dir = os.path.join(log_dir, 'checkpoints')
        os.makedirs(self.log_dir, exist_ok=True)

    def run(self, trainer: Trainer):
        torch.save(
            {key: value.state_dict() for key, value in self.modules_to_save.items()},
            os.path.join(self.log_dir, f"{str(trainer.updater.iteration).zfill(6)}.pt")
        )

    def finalize(self, trainer: 'Trainer'):
        self.run(trainer)
