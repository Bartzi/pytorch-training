import os

import torchvision

from pytorch_training.trainer import Trainer
from pytorch_training.extension import Extension


class ImageDebugger(Extension):

    def __init__(self, log_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_dir = os.path.join(log_dir, 'debug_images')
        os.makedirs(self.log_dir, exist_ok=True)

    def run(self, trainer: Trainer):
        current_input_image = trainer.updater.current_image
        iteration = trainer.updater.iteration
        torchvision.utils.save_image(current_input_image, os.path.join(self.log_dir, f"{iteration}.png"), normalize=True)
