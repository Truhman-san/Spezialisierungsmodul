from __future__ import annotations
from typing import Tuple
import numpy as np
import cv2
import random

from .config import BASE_TILE, tile_repetitions_for_canvas


def make_repeated_tile(
    canvas_size: Tuple[int, int],
    base_tile: np.ndarray | None = None,
) -> np.ndarray:
    if base_tile is None:
        base_tile = BASE_TILE

    H, W = int(canvas_size[0]), int(canvas_size[1])
    reps_h, reps_w = tile_repetitions_for_canvas((H, W))

    tile_big = np.tile(base_tile, (reps_h, reps_w)).astype(np.float32)
    canvas = tile_big[:H, :W]
    return canvas.astype(np.uint8)


def build_dimer_canvas(
    canvas_size: Tuple[int, int],
    rotation_deg_range: Tuple[float, float] = (-8.0, 8.0),
) -> np.ndarray:
    H, W = int(canvas_size[0]), int(canvas_size[1])

    base = make_repeated_tile((H, W), BASE_TILE).astype(np.float32)

    rot_deg = random.uniform(*rotation_deg_range)
    M = cv2.getRotationMatrix2D((W // 2, H // 2), rot_deg, 1.0)
    rotated = cv2.warpAffine(
        base,
        M,
        (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return np.clip(rotated, 0, 255).astype(np.uint8)
