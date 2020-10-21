import os
from typing import List

import torch
import torchvision.utils
from PIL import Image

from pytorch_training.trainer import Trainer
from pytorch_training.extension import Extension
from pytorch_training.reporter import get_current_reporter
from pytorch_training.images.utils import make_image


class ImagePlotter(Extension):

    def __init__(self, input_images: list, networks: list, log_dir: str, *args, plot_to_logger: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_images = torch.stack(input_images).cuda()
        self.networks = networks
        self.image_dir = os.path.join(log_dir, 'images')
        self.log_to_logger = plot_to_logger
        os.makedirs(self.image_dir, exist_ok=True)

    def initialize(self, trainer: Trainer):
        self.run(trainer)

    def get_predictions(self) -> List[torch.Tensor]:
        predictions = [self.input_images]
        for network in self.networks:
            predictions.append(network(predictions[-1]))
        return predictions

    def run(self, trainer: Trainer):
        try:
            for network in self.networks:
                network.eval()
            with torch.no_grad():
                predictions = self.get_predictions()
        finally:
            for network in self.networks:
                network.train()

        display_images = torch.cat(predictions, dim=0)

        image_grid = torchvision.utils.make_grid(display_images, nrow=self.input_images.shape[0])

        dest_file_name = os.path.join(self.image_dir, f"{trainer.updater.iteration:08d}.png")
        dest_image = make_image(image_grid)
        Image.fromarray(dest_image).save(dest_file_name)

        if self.log_to_logger:
            with get_current_reporter() as reporter:
                reporter.add_image({"image_plotter": dest_image}, trainer.updater.iteration)

        del display_images
        torch.cuda.empty_cache()
