from typing import Callable, Union

import torch
import torch.distributed
from torch.utils.data import DataLoader
from tqdm import tqdm

from pytorch_training.distributed import synchronize, get_rank, get_world_size
from pytorch_training.extension import Extension
from pytorch_training.extensions.logger import Logger
from pytorch_training.reporter import Reporter
from pytorch_training.trainer import Trainer


class Evaluator(Extension):

    def __init__(self, data_loader: DataLoader, logger: Logger, eval_func: Callable[..., None], device: Union[int, str], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_loader = data_loader
        self.logger = logger
        self.eval_func = eval_func
        self.device = device

    def initialize(self, trainer: 'Trainer'):
        self.run(trainer)

    def finalize(self, trainer: 'Trainer'):
        self.run(trainer)

    def synchronize_observation(self, observation: torch.Tensor) -> torch.Tensor:
        if get_world_size() > 1:
            if torch.distributed.get_backend() == 'nccl':
                # we are running the nccl backend, so all of our tensors have to be on gpu!
                observation = observation.to('cuda')

            gathered_observations = [torch.empty(observation.shape, device=observation.device) for _ in range(get_world_size())]
            torch.distributed.all_gather(gathered_observations, observation)
            observation = torch.stack(gathered_observations, dim=0)
        return observation.mean().cpu()

    def run(self, trainer: Trainer):
        reporter = Reporter()

        self.evaluate(reporter)

        synchronize()
        observation = reporter.get_observations()
        observation = {k: float(self.synchronize_observation(torch.tensor(obs))) for k, obs in observation.items()}

        if get_rank() == 0:
            self.logger.save_observations(trainer, observation)
            self.logger.log_observations()

    def progress_bar(self):
        if get_rank() == 0:
            iterator = tqdm(self.data_loader, leave=False)
            iterator.set_description("Evaluation")
        else:
            iterator = self.data_loader
        return iterator

    def evaluate(self, reporter: Reporter):
        for batch in self.progress_bar():
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with reporter:
                self.eval_func(batch)

        torch.cuda.empty_cache()
