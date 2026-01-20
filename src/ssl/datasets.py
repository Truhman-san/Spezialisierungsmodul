from __future__ import annotations
from pathlib import Path
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE

def _load_png_gray(path: tf.Tensor, image_size: tuple[int,int]):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.resize(img, image_size, method="bilinear")
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img  # (H,W,1)

def build_unlabeled_ds(images_dir: str, image_size: tuple[int,int], batch_size: int,
                       shuffle: int = 1024, seed: int = 42):
    p = Path(images_dir)
    files = sorted([str(x) for x in p.glob("*.png")])
    if not files:
        raise FileNotFoundError(f"No PNGs found in {images_dir}")
    ds = tf.data.Dataset.from_tensor_slices(files)
    ds = ds.shuffle(min(shuffle, len(files)), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda f: _load_png_gray(f, image_size), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds
