import tempfile
from pathlib import Path

import numpy
import pytest
import torch
from PIL import Image
from torchvision import transforms

from pytorch_training.data.transforms import Compose
from pytorch_training.images.utils import load_and_prepare_image, clamp_and_unnormalize, make_image


class TestImageUtils:

    @pytest.fixture(params=[(1, 3, 256, 256), (3, 256, 256)])
    def tensor(self, request):
        tensor = torch.rand(request.param)
        tensor[0, 0] = -1
        tensor[-1, -1] = 1

        return tensor

    def test_clamp_and_unnormalize(self, tensor):
        normalized_tensor = tensor * 2 - 1
        unnormalized_tensor = clamp_and_unnormalize(normalized_tensor)

        tensor[0, 0] = 0

        assert torch.allclose(tensor, unnormalized_tensor)

    def test_make_image(self, tensor):
        normalized_tensor = tensor * 2 - 1
        image = make_image(normalized_tensor)

        assert isinstance(image, numpy.ndarray)
        assert image.max() == 255
        assert image.min() == 0

    def test_load_and_prepare_image(self, tensor):
        test_image = Image.new('RGB', (512, 512), 'black')

        transform = Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            file_name = temp_dir / "image.png"
            test_image.save(file_name)

            loaded_image = load_and_prepare_image(file_name, transform)

            assert loaded_image.shape == (1, 3, 256, 256)
            assert loaded_image.min() == -1
            assert loaded_image.max() == -1
