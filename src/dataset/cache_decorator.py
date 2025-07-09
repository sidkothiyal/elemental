import hashlib
from functools import wraps
from pathlib import Path

import dill
import lz4.frame


def cache_getitem(cache_dir):
    """
    Decorator to cache __getitem__ results to disk using dill + lz4.
    Each sample is saved as an individual file in cache_dir.
    Usage:
        @cache_getitem(cache_dir="/tmp/mycache")
        def __getitem__(self, idx):
            ...
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    def decorator(getitem_func):
        @wraps(getitem_func)
        def wrapper(self, idx):
            # Build a unique cache key based on class, idx, and key dataset params
            key_parts = [self.__class__.__name__, str(idx)]
            # Optionally add common dataset attributes if present
            for attr in [
                "split",
                "seed",
                "max_length",
                "pretrained_tokenizer",
                "segmentation_enabled",
                "depth_enabled",
                "normalize",
                "classes",
                "root_dir",
                "dataset_dir",
            ]:
                if hasattr(self, attr):
                    value = getattr(self, attr)
                    key_parts.append(f"{attr}:{value}")
            key_str = "|".join(map(str, key_parts))
            cache_key = hashlib.md5(key_str.encode()).hexdigest()
            cache_path = cache_dir / f"getitem_{cache_key}.dill.lz4"

            if cache_path.exists():
                try:
                    with open(cache_path, "rb") as f:
                        compressed = f.read()
                    data = dill.loads(lz4.frame.decompress(compressed))
                    return data
                except Exception as e:
                    print(f"[cache_getitem] Failed to load cache: {e}. Recomputing.")

            # Cache miss or error: compute and save
            data = getitem_func(self, idx)
            try:
                compressed = lz4.frame.compress(dill.dumps(data))
                with open(cache_path, "wb") as f:
                    f.write(compressed)
            except Exception as e:
                print(f"[cache_getitem] Failed to save cache: {e}")
            return data

        return wrapper

    return decorator
