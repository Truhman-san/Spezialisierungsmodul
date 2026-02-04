from __future__ import annotations
from typing import Tuple
import numpy as np
import cv2
import random
import math

from .config import BASE_TILE


def make_repeated_tile(
    canvas_size: Tuple[int, int],
    base_tile: np.ndarray | None = None,
) -> np.ndarray:
    if base_tile is None:
        base_tile = BASE_TILE

    H, W = int(canvas_size[0]), int(canvas_size[1])
    tile_h, tile_w = base_tile.shape[:2]

    reps_h = math.ceil(H / tile_h)
    reps_w = math.ceil(W / tile_w)

    tile_big = np.tile(base_tile, (reps_h, reps_w))
    canvas = tile_big[:H, :W]
    return canvas.astype(np.uint8)


