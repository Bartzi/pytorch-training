from typing import Iterable

from torch import nn
from torch.optim.optimizer import Optimizer


class Updater:

    def __init__(self, iterators: dict, networks: dict, optimizers: dict, loss_weights: dict = None, device: str = 'cuda', copy_to_device: bool = True):
        self.data_loaders = iterators
        self.iterators = None
        self.networks = networks
        self.optimizers = optimizers
        self.loss_weights = loss_weights
        self.device = device

        if copy_to_device:
            for network in self.networks.values():
                network.to(device)

        self.iteration = 0
        self.reset()

    @property
    def epoch_length(self) -> int:
        return min(len(iterator) for iterator in self.data_loaders.values())

    @property
    def epoch_detail(self) -> float:
        return self.current_epoch + self.iteration_in_epoch / self.epoch_length

    @property
    def current_epoch(self):
        return self.iteration // self.epoch_length

    @property
    def iteration_in_epoch(self):
        return self.iteration % self.epoch_length

    def update(self):
        self.update_core()
        self.iteration += 1

    def reset(self):
        self.iterators = {key: iter(data_loader) for key, data_loader in self.data_loaders.items()}

    def update_core(self):
        raise NotImplementedError


class UpdateDisabler:

    def __init__(self, network: nn.Module):
        self.network = network

    def set_requires_grad(self, flag: bool):
        for parameter in self.network.parameters():
            parameter.requires_grad = flag

    def __enter__(self):
        self.set_requires_grad(False)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False

        self.set_requires_grad(True)


class GradientApplier:

    def __init__(self, networks: Iterable[nn.Module], optimizers: Iterable[Optimizer]):
        self.networks = networks
        self.optimizers = optimizers

    def __enter__(self):
        for network in self.networks:
            network.zero_grad()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False

        for optimizer in self.optimizers:
            optimizer.step()
