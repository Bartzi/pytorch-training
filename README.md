# pytorch-training

Repository to keep pytorch training scripts.

# Installation

run `pip install -e .` in the root of the repository.

# Usage

This repo contains an extensible Training Loop abstraction.

For training, you'll need to build your own `Updater` based on `pytorch_training.updater.Updater`.
You can then use this `updater` to build a `Trainer` object that handles the training.
