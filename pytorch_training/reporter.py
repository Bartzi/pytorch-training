from __future__ import annotations

import statistics
import threading
from collections import defaultdict
from typing import Dict, Union, List

import numpy
import torch

_thread_local = threading.local()


class Reporter:

    def __init__(self):
        self.observations = []
        self.current_observation = {}
        self.histograms = []
        self.current_histogram = {}
        self.images = []
        self.current_image = {}

    def reset(self):
        self.observations.clear()
        self.histograms.clear()
        self.images.clear()

    def get_mean_observation(self) -> dict:
        all_observations = self.get_observations()

        mean_observation = {key: statistics.mean(value) for key, value in all_observations.items()}
        return mean_observation

    def get_observations(self) -> dict:
        all_observations = defaultdict(list)
        for observation in self.observations:
            for key, value in observation.items():
                all_observations[key].append(value)

        return all_observations

    def get_histograms(self) -> list:
        return self.histograms

    def get_images(self) -> list:
        return self.images

    def add_observation(self, observation: Dict[str, Union[torch.Tensor, float]], prefix: str = None):
        if prefix is not None:
            observation = {f"{prefix}/{key}": value for key, value in observation.items()}

        observation = {key: float(value) for key, value in observation.items()}
        self.current_observation.update(observation)

    def add_histogram(self, histogram: Dict[str, Union[numpy.ndarray, int]], iteration: int, prefix: str = None):
        if prefix is not None:
            histogram = {f"{prefix}/{key}": value for key, value in histogram.items()}

        histogram['iteration'] = iteration
        self.current_histogram.update(histogram)

    def add_image(self, image: Dict[str, Union[numpy.ndarray, int]], iteration: int, prefix: str = None):
        if prefix is not None:
            image = {f"{prefix}/{key}": value for key, value in image.items()}

        image['iteration'] = iteration
        self.current_image.update(image)

    def set_as_active_reporter(self):
        _get_reporters().append(self)

    def __enter__(self) -> Reporter:
        _get_reporters().append(self)
        self.current_observation = {}
        self.current_histogram = {}
        self.current_image = {}
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False

        if len(self.current_observation) > 0:
            self.observations.append(self.current_observation)
        if len(self.current_histogram) > 0:
            self.histograms.append(self.current_histogram)
        if len(self.current_image) > 0:
            self.images.append(self.current_image)
        _get_reporters().pop()


def _get_reporters() -> List[Reporter]:
    try:
        reporters = _thread_local.reporters
    except AttributeError:
        reporters = _thread_local.reporters = []
    return reporters


def get_current_reporter() -> Reporter:
    return _get_reporters()[-1]
