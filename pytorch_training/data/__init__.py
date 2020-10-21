from torch.utils.data import DataLoader

from pytorch_training.data.folder_dataset import ImageFolder
from pytorch_training.data.transforms import Compose


def get_data_loader(image_dir, config, transforms) -> DataLoader:
    transform = transforms.Compose(transforms)
    dataset = ImageFolder(image_dir, transform)

    loader = DataLoader(
        dataset=dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        drop_last=True,
        num_workers=config['num_workers']
    )

    return loader
