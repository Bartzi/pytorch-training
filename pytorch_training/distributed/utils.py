from torch import distributed
from torch.nn.parallel import DistributedDataParallel as DDP

def get_rank():
    if not distributed.is_available():
        return 0

    if not distributed.is_initialized():
        return 0

    return distributed.get_rank()


def synchronize():
    if not distributed.is_available():
        return

    if not distributed.is_initialized():
        return

    world_size = distributed.get_world_size()

    if world_size == 1:
        return

    distributed.barrier()


def get_world_size():
    if not distributed.is_available():
        return 1

    if not distributed.is_initialized():
        return 1

    return distributed.get_world_size()


def strip_parallel_module(module):
    return module if not isinstance(module, DDP) else module.module
