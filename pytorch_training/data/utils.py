import os
import tempfile
import warnings
from pathlib import Path

import filelock as filelock
import shutil
from PIL import Image
from typing import Callable, Any

Image.init()


_dataset_root = Path(os.environ.get(
    'PYTORCH_TRAINING_DATASET_ROOT',
    os.path.join(os.path.expanduser('~'), '.pytorch_training', 'datasets')))


def default_loader(path):
    return Image.open(path).convert('RGB')


def is_image_file(file_name):
    msg = ("pytorch_training.data.utils.is_image_file should be replaced by "
           "pytorch_training.images.utils.is_image.")
    warnings.warn(msg, DeprecationWarning)
    return Path(file_name).suffix.lower() in Image.EXTENSION.keys()


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

# Borrowed from Chainer =)
def cache_or_load_file(path: str, creator: Callable, loader: Callable) -> Any:
    """Caches a file if it does not exist, or loads it otherwise.
    This is a utility function used in dataset loading routines. The
    ``creator`` creates the file to given path, and returns the content. If the
    file already exists, the ``loader`` is called instead, and it loads the
    file and returns the content.
    Note that the path passed to the creator is temporary one, and not same as
    the path given to this function. This function safely renames the file
    created by the creator to a given path, even if this function is called
    simultaneously by multiple threads or processes.
    Args:
        path (str): Path to save the cached file.
        creator: Function to create the file and returns the content. It takes
            a path to temporary place as the argument. Before calling the
            creator, there is no file at the temporary path.
        loader: Function to load the cached file and returns the content.
    Returns:
        It returns the returned values by the creator or the loader.
    """
    path = Path(path)
    if path.exists():
        return loader(path)

    _dataset_root.mkdir(parents=True, exist_ok=True)

    lock_path = _dataset_root / '_create_lock'

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        file_name = path.name
        temp_path = temp_dir / file_name
        content = creator(temp_path)
        with filelock.FileLock(lock_path):
            if not os.path.exists(path):
                shutil.move(temp_path, path)

    return content
