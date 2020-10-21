from __future__ import annotations

from torchvision.transforms import Compose as TorchCompose


class Compose(TorchCompose):

    def add(self, transforms: list) -> Compose:
        self.transforms.extend(transforms)
        return self
