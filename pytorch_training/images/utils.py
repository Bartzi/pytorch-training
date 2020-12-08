from pathlib import Path
from typing import Union, Callable

import numpy
import torch
from PIL import Image

from pytorch_training.data import Compose


Image.init()


def is_image(file_name: Union[str, Path]) -> bool:
    if not isinstance(file_name, Path):
         file_name = Path(file_name)
    return file_name.suffix.lower() in Image.EXTENSION.keys()


def load_and_prepare_image(file_name: Union[str, Path], transforms: Compose, add_batch_dim=True) -> torch.Tensor:
    with Image.open(file_name) as image:
        image = image.convert("RGB")
        tensor = transforms(image)

    if len(tensor.shape) == 3 and add_batch_dim:
        tensor = tensor.unsqueeze(0)

    return tensor


def clamp_and_unnormalize(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.clamp_(min=-1, max=1).add(1).div_(2)


def make_image(tensor: torch.Tensor, normalize_func: Callable[[torch.Tensor], torch.Tensor] = clamp_and_unnormalize) -> numpy.ndarray:
    tensor = tensor.detach()
    tensor = normalize_func(tensor)
    tensor = tensor.mul(255).type(torch.uint8)
    if len(tensor.shape) == 4:
        permute_indices = (0, 2, 3, 1)
    else:
        permute_indices = (1, 2, 0)
    tensor = tensor.permute(*permute_indices).to('cpu').numpy()
    if tensor.shape[-1] == 1:
        # grayscale image
        tensor = tensor.squeeze(-1)
    return tensor
