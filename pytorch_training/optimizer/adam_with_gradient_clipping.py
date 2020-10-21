from typing import Callable

from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam


class GradientClipAdam(Adam):

    def __init__(self, *args, max_norm: float = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def apply_gradient_clipping(self):
        for group in self.param_groups:
            clip_grad_norm_(group['params'], self.max_norm)

    def step(self, closure: Callable = None) -> None:
        self.apply_gradient_clipping()
        super().step(closure)
