from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytorch_training import Trainer


class Trigger:

    def __call__(self, trainer: 'Trainer') -> bool:
        raise NotImplementedError
