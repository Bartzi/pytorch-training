import json
from collections import Callable
from pathlib import Path
from typing import Dict, Union

import torch
from torch.utils import data

from pytorch_training.data.utils import default_loader
from pytorch_training.images import is_image


class JSONDataset(data.Dataset):

    def __init__(self, json_file: Union[str, Path], root: Union[str, Path] = None, transforms: Callable = None, loader: Callable = default_loader):
        with Path(json_file).open() as f:
            self.load_json_data(json.load(f))

        self.root = Path(root)
        self.transforms = transforms
        self.loader = loader

    def load_json_data(self, json_data: Union[dict, list]):
        self.image_data = json_data
        self.image_data = [file_path for file_path in self.image_data if is_image(file_path)]

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        path = self.image_data[index]
        if self.root is not None:
            path = self.root / path

        image = self.loader(path)

        if self.transforms is not None:
            image = self.transforms(image)

        return {'image': image}
