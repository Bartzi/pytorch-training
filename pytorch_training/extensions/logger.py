import json
import multiprocessing
import os
import queue
import shutil
from argparse import Namespace
from typing import List, Union

import numpy
import wandb
from torch.utils.tensorboard import SummaryWriter

from pytorch_training.trainer import Trainer
from pytorch_training.extension import Extension
from pytorch_training.reporter import Reporter


class Logger(Extension):

    def __init__(self, log_dir, train_args, train_config, backup_base_path, *args, exclusion_filters=(r'*logs*',), master=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.reporter = Reporter()
        self.reporter.set_as_active_reporter()
        self.log_dir = log_dir
        self.master = master
        self.observation_queue = multiprocessing.Queue()
        self.histogram_queue = multiprocessing.Queue()
        self.image_queue = multiprocessing.Queue()

        if self.master:
            self.backup_code(backup_base_path, log_dir, exclusion_filters)
            self.log_args(train_args, 'args.json')
            self.log_args(train_config, 'config.json')

    def log_scalar(self, *args):
        if self.master:
            self.log_scalar_impl(*args)

    def log_scalar_impl(self, key: str, scalar: Union[float, int], iteration: int):
        raise NotImplementedError

    def log_histogram(self, *args):
        if self.master:
            self.log_histogram_impl(*args)

    def log_histogram_impl(self, key: str, histogram: numpy.ndarray, iteration: int):
        raise NotImplementedError

    def log_image(self, *args):
        if self.master:
            self.log_image_impl(*args)

    def log_image_impl(self, key: str, image: numpy.ndarray, iteration: int):
        assert image.dtype == 'uint8', "supplied image has a not supported dtype"
        assert image.shape[-1] in [1, 3], "supplied image should already be transposed and have either 1 or 3 channels"
        assert len(image.shape) == 3, "only one image is supposed to be supplied"

    def save_observations(self, trainer: Trainer, observations: dict):
        self.observation_queue.put((trainer.updater.iteration, observations))

    def save_histograms(self, trainer: Trainer, histograms: List[dict]):
        for iteration_histograms in histograms:
            iteration = iteration_histograms.pop('iteration')
            self.histogram_queue.put((iteration, iteration_histograms))

    def save_images(self, trainer: Trainer, images: List[dict]):
        for iteration_image in images:
            iteration = iteration_image.pop('iteration')
            self.image_queue.put((iteration, iteration_image))

    def log_observations(self):
        try:
            while not self.observation_queue.empty():
                queue_data = self.observation_queue.get_nowait()
                for key, value in queue_data[1].items():
                    self.log_scalar(key, value, queue_data[0])
        except queue.Empty:
            pass

    def log_histograms(self):
        try:
            while not self.histogram_queue.empty():
                queue_data = self.histogram_queue.get_nowait()
                for key, value in queue_data[1].items():
                    self.log_histogram(key, value, queue_data[0])
        except queue.Empty:
            pass

    def log_images(self):
        try:
            while not self.image_queue.empty():
                queue_data = self.image_queue.get_nowait()
                for key, value in queue_data[1].items():
                    self.log_image(key, value, queue_data[0])
        except queue.Empty:
            pass

    def log_args(self, args, file_name):
        if isinstance(args, Namespace):
            args = vars(args)

        log_dir = os.path.join(self.log_dir, 'config')
        os.makedirs(log_dir, exist_ok=True)

        args = {k: v for k, v in args.items() if not k.startswith('_')}
        with open(os.path.join(log_dir, file_name), 'w') as handle:
            json.dump(args, handle, indent='\t')

    def run(self, trainer):
        self.save_observations(trainer, self.reporter.get_mean_observation())
        self.save_histograms(trainer, self.reporter.get_histograms())
        self.save_images(trainer, self.reporter.get_images())
        if self.master:
            self.log_observations()
            self.log_histograms()
            self.log_images()
        self.reporter.reset()

    def finalize(self, trainer: 'Trainer'):
        self.run(trainer)

    def backup_code(self, backup_base_path, log_dir, exclusion_filters):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        ignore_patterns = shutil.ignore_patterns(*exclusion_filters)
        shutil.copytree(backup_base_path, os.path.join(log_dir, 'code'), ignore=ignore_patterns)


class TensorboardLogger(Logger):

    def log_image_impl(self, key, image, iteration):
        super().log_image_impl(key, image, iteration)
        self.tensorboard_handle.add_image(key, image, global_step=iteration, dataformats='HWC')

    def log_scalar_impl(self, key, scalar, iteration):
        self.tensorboard_handle.add_scalar(key, scalar, global_step=iteration)

    def log_histogram_impl(self, key, histogram, iteration):
        self.tensorboard_handle.add_histogram(key, histogram, global_step=iteration)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.master:
            self.tensorboard_handle = SummaryWriter(self.log_dir)


class WandBLogger(Logger):

    def log_image_impl(self, key: str, image: numpy.ndarray, iteration: int):
        super().log_image_impl(key, image, iteration)
        self.wandb_handle.log({key: [wandb.Image(image)]})

    def log_scalar_impl(self, key, scalar, iteration):
        self.wandb_handle.log({key: scalar}, step=iteration)

    def log_histogram_impl(self, key, histogram, iteration):
        self.wandb_handle.log({key: histogram}, step=iteration)

    def __init__(self, *args, run_name=None, project_name=None,  **kwargs):
        kwargs['exclusion_filters'] = (r'*logs*', r'*wandb')
        if kwargs.get('master', True):
            assert run_name is not None, "You must supply a name for the current run"
            assert project_name is not None, "You must supply a name for the project that is to be logged on WandB"
            self.wandb_handle = wandb.init(name=run_name, project=project_name, force=True)
        super().__init__(*args, **kwargs)

    def log_args(self, args, file_name):
        wandb.config.update(args, allow_val_change=True)
        super().log_args(args, file_name)
