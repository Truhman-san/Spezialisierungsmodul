from __future__ import annotations
import numpy as np
import random

from .config import BASE_TILE_FLOAT, DTYPE, TILE_WIDTH


def _z_factor_x(x: np.ndarray, A: float, B: float) -> np.ndarray:
    """
    Z_factor(X) = A * (1/(7*sqrt(pi/2))*exp(-2*((X-5-B)/7)^2)
                      + 1/(7*sqrt(pi/2))*exp(-2*((X-14+B)/7)^2)
                      - 0.04140)
    x: 1D-Array der Dimer-Indizes (1..18)
    """
    norm = 1.0 / (7.0 * np.sqrt(np.pi / 2.0))

    term1 = norm * np.exp(-2.0 * ((x - 5.0 - B) / 7.0) ** 2)
    term2 = norm * np.exp(-2.0 * ((x - 14.0 + B) / 7.0) ** 2)

    zf = A * (term1 + term2 - 0.04140)
    return zf


def build_physical_dimer_tile(
    A: float | None = None,
    B: float | None = None,
) -> np.ndarray:
    """
    Erzeugt ein physikalisch modifiziertes 9x18-Dimer-Template auf Basis von BASE_TILE_FLOAT.

    Height(X) = Z_raw(X) / (1 + Z_factor(X))
    - A: 0 < A < 30
    - B: 0 < B < 1
    - Rückgabe: uint8[9,18] mit 0..255
    """
    if A is None:
        A = random.uniform(0.0, 30.0)
    if B is None:
        B = random.uniform(0.00, 1.0)

    x_idx = np.arange(1, TILE_WIDTH + 1, dtype=np.float64)
    zf = _z_factor_x(x_idx, A=A, B=B)  # (18,)

    # 9x18: gleiche Modulation für alle Y-Zeilen
    denom = 1.0 + zf[None, :]          
    z_raw = np.asarray(BASE_TILE_FLOAT, dtype=np.float64)
    z_mod = z_raw / denom

    z_min = float(z_mod.min())
    z_max = float(z_mod.max())
    if z_max > z_min:
        z_norm = (z_mod - z_min) / (z_max - z_min)
    else:
        z_norm = np.zeros_like(z_mod, dtype=np.float64)

    tile_uint8 = (z_norm * (DTYPE.gray_levels - 1)).astype(DTYPE.gray_dtype, copy=False)
    return tile_uint8
