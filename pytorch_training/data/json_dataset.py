import json
import os
from collections import Callable

import numpy
from torch.utils import data
from typing import Dict

from pytorch_training.data.utils import default_loader, is_image_file
from pytorch_training.images import is_image


class JSONDataset(data.Dataset):

    def __init__(self, json_file: str, root: str = None, transforms: Callable = None, loader: Callable = default_loader):
        with open(json_file) as f:
            self.image_data = json.load(f)
            self.image_data = [file_path for file_path in self.image_data if is_image(file_path)]

        self.root = root
        self.transforms = transforms
        self.loader = loader

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, index: int) -> Dict[str, numpy.ndarray]:
        path = self.image_data[index]
        if self.root is not None:
            path = os.path.join(self.root, path)

        image = self.loader(path)

        if self.transforms is not None:
            image = self.transforms(image)

        return {'image': image}
