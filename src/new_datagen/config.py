from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Sequence
import numpy as np

# ---------------------------
# Dimensionen / Canvas
# ---------------------------

TILE_HEIGHT: int = 9
TILE_WIDTH: int = 18

CANVAS_SIZE: Tuple[int, int] = (1800, 1800)  


# ---------------------------
# Basis-Kachel 
# ---------------------------

BASE_TILE_DATA: Tuple[Tuple[float, ...], ...] = (
    (
        0.00392, 0.00235, 0.01647, 0.03922, 0.06588, 0.07843,
        0.07882, 0.07451, 0.07373, 0.07373, 0.07451, 0.07882,
        0.07843, 0.06588, 0.03961, 0.01647, 0.00235, 0.00392,
    ),
    (
        0.00392, 0.00510, 0.02196, 0.04431, 0.07137, 0.08196,
        0.08078, 0.07843, 0.08235, 0.08196, 0.07804, 0.08078,
        0.08196, 0.07137, 0.04471, 0.02196, 0.00510, 0.00392,
    ),
    (
        0.00863, 0.01255, 0.03020, 0.05569, 0.07843, 0.08628,
        0.08588, 0.08471, 0.08549, 0.08549, 0.08471, 0.08588,
        0.08667, 0.07843, 0.05569, 0.03098, 0.01294, 0.00863,
    ),
    (
        0.01412, 0.01529, 0.03177, 0.06039, 0.08353, 0.09294,
        0.09529, 0.09098, 0.08863, 0.08824, 0.09098, 0.09490,
        0.09294, 0.08353, 0.06078, 0.03177, 0.01529, 0.01373,
    ),
    (
        0.01529, 0.01529, 0.03608, 0.06235, 0.08353, 0.09765,
        0.09882, 0.09451, 0.08902, 0.08863, 0.09412, 0.09882,
        0.09765, 0.08314, 0.06235, 0.03608, 0.01529, 0.01529,
    ),
    (
        0.01412, 0.01529, 0.03216, 0.06078, 0.08392, 0.09294,
        0.09529, 0.09059, 0.08824, 0.08824, 0.09059, 0.09529,
        0.09294, 0.08353, 0.06039, 0.03216, 0.01529, 0.01412,
    ),
    (
        0.00863, 0.01294, 0.03098, 0.05608, 0.07804, 0.08667,
        0.08588, 0.08431, 0.08549, 0.08549, 0.08431, 0.08588,
        0.08667, 0.07843, 0.05569, 0.03020, 0.01255, 0.00863,
    ),
    (
        0.00353, 0.00510, 0.02196, 0.04471, 0.07098, 0.08235,
        0.08078, 0.07804, 0.08235, 0.08235, 0.07804, 0.08118,
        0.08235, 0.07137, 0.04392, 0.02157, 0.00471, 0.00353,
    ),
    (
        0.00353, 0.00235, 0.01647, 0.03961, 0.06588, 0.07804,
        0.07843, 0.07451, 0.07373, 0.07373, 0.07451, 0.07882,
        0.07804, 0.06588, 0.03922, 0.01647, 0.00235, 0.00353,
    ),
)


@dataclass(frozen=True)
class DtypeSpec:
    gray_dtype: np.dtype = np.uint8      
    float_dtype: np.dtype = np.float64  
    gray_levels: int = 256               


DTYPE = DtypeSpec()


def build_base_tile(scale_to_uint8: bool = True) -> np.ndarray:
    arr = np.asarray(BASE_TILE_DATA, dtype=DTYPE.float_dtype)

    if arr.shape != (TILE_HEIGHT, TILE_WIDTH):
        raise ValueError(
            f"BASE_TILE_DATA hat Form {arr.shape}, erwartet {(TILE_HEIGHT, TILE_WIDTH)}."
        )

    if scale_to_uint8:
        max_val = float(np.max(arr))
        if max_val <= 0.0:
            raise ValueError("BASE_TILE_DATA enthält keine positiven Werte – Skalierung nicht möglich.")
        arr = (arr / max_val) * (DTYPE.gray_levels - 1)
        return arr.astype(DTYPE.gray_dtype, copy=False)
    else:
        return arr


# Beim Import einmal erzeugen
BASE_TILE: np.ndarray = build_base_tile(scale_to_uint8=True)
BASE_TILE_FLOAT: np.ndarray = np.asarray(BASE_TILE_DATA, dtype=DTYPE.float_dtype)

# Checks
TILE_SHAPE: Tuple[int, int] = (TILE_HEIGHT, TILE_WIDTH)
TILE_AREA: int = TILE_HEIGHT * TILE_WIDTH

assert BASE_TILE.shape == TILE_SHAPE
assert BASE_TILE.dtype == DTYPE.gray_dtype
assert 0 <= int(BASE_TILE.min()) and int(BASE_TILE.max()) <= 255

# Dimer-Maske 
DIMER_THRESHOLD: float = float(np.percentile(BASE_TILE_FLOAT, 75.0))
BASE_TILE_DIMER_MASK: np.ndarray = (BASE_TILE_FLOAT >= DIMER_THRESHOLD).astype(np.uint8)
assert BASE_TILE_DIMER_MASK.shape == TILE_SHAPE


def tile_repetitions_for_canvas(canvas_size: Tuple[int, int] | Sequence[int]) -> Tuple[int, int]:
    H, W = int(canvas_size[0]), int(canvas_size[1])
    reps_h = int(np.ceil(H / TILE_HEIGHT))
    reps_w = int(np.ceil(W / TILE_WIDTH))
    return reps_h, reps_w
