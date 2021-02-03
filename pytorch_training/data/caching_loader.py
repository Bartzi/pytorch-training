from pathlib import Path
from typing import Union, Callable, Dict

from PIL import Image
from tqdm import tqdm

from pytorch_training.data.utils import default_loader
from pytorch_training.images import is_image

DEFAULT_CACHE_ROOT = Path().cwd() / 'image_cache'

class CachingLoader:

    def __init__(self, load_root: Union[str, Path], cache_root: Path = DEFAULT_CACHE_ROOT, base_loader: Callable = default_loader):
        self.load_root = load_root
        self.base_loader = base_loader
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.cache: Dict[str, str] = {}
        self.build_cache()

    def build_cache(self):
        for path in tqdm(self.cache_root.glob(r'**/*'), desc="Building local image cache", leave=False):
            if is_image(path):
                self.cache[str(path.relative_to(self.cache_root))] = str(path)

    def __call__(self, path: Union[str, Path]) -> Image:
        if not isinstance(path, Path):
            path = Path(path)

        cache_key = str(path.relative_to(self.load_root))
        local_image_path = self.cache.get(cache_key, None)

        if local_image_path is not None:
            # we have the image already in our cache, so no need to draw it from the remote anymore
            return self.base_loader(local_image_path)

        # we need to get the image from the remote file server and cache it locally
        image = self.base_loader(path)
        cache_path = self.cache_root / cache_key
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(cache_path)
        self.cache[cache_key] = str(cache_path)
        return image
