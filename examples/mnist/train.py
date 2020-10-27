import argparse
from datetime import datetime
from pathlib import Path

from torch.optim import Adadelta
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms

from pytorch_training import Trainer
from pytorch_training.extensions import Snapshotter, Evaluator
from pytorch_training.extensions.logger import TensorboardLogger
from pytorch_training.extensions.lr_scheduler import LRScheduler
from pytorch_training.triggers import get_trigger

from dataset import MNIST
from evaluation import MNISTEvaluator
from network import Net
from updater import MNISTUpdater


def main(args):
    # first, we define some pre-processing
    data_transforms = transforms.Compose([
        # extra augmentations
        # transforms.ColorJitter(brightness=0.3),
        # necessary transformations
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # build our data loaders
    train_dataset = MNIST('../data', train=True, download=True, transform=data_transforms)
    test_dataset = MNIST('../data', train=False, transform=data_transforms)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

    # create the model
    model = Net()

    # build an optimizer for optimizing the parameters of our model
    optimizer = Adadelta(model.parameters(), lr=args.lr)

    # if we want to use cuda, we have to copy all parameters to the GPU
    model.to(args.device)

    # build object that handles updating routine
    updater = MNISTUpdater(
        iterators={'images': train_dataloader},
        networks={'net': model},
        optimizers={'main': optimizer},
        device=args.device,
        copy_to_device=True,
    )

    # build the trainer
    trainer = Trainer(
        updater,
        stop_trigger=get_trigger((args.epochs, 'epoch'))
    )

    # prepare logging
    logger = TensorboardLogger(
        args.log_dir,
        args,
        {},
        Path(__file__).resolve().parent,
        trigger=get_trigger((100, 'iteration'))
    )

    # make sure we are evaluating
    trainer.extend(Evaluator(
        test_dataloader,
        logger,
        MNISTEvaluator(model),
        args.device,
    ))

    # make sure we are saving the trained models to disk, including the optimizer. This allows us to resume training.
    snapshotter = Snapshotter(
        {
            'network': model,
            'optimizer': optimizer
        },
        args.log_dir
    )
    trainer.extend(snapshotter)

    # add learning rate scheduling, in this case Cosine Annealing
    schedulers = {
        "encoder": CosineAnnealingLR(optimizer, trainer.num_epochs * trainer.iterations_per_epoch, eta_min=1e-8)
    }
    lr_scheduler = LRScheduler(schedulers, trigger=get_trigger((1, 'iteration')))
    trainer.extend(lr_scheduler)

    trainer.extend(logger)

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on MNIST", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--device", default='cuda', help="Device to use for training")
    parser.add_argument("-b", "--batch-size", default=64, type=int, help="batch size for training")
    parser.add_argument('--epochs', type=int, default=14, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1.0, help='learning rate')
    parser.add_argument('-l', '--log-dir', default='logs', help='where to log the train results')

    parsed_args = parser.parse_args()

    # c
    parsed_args.log_dir = str(Path(parsed_args.log_dir) / datetime.utcnow().isoformat())
    main(parsed_args)
