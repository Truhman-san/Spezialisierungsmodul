from __future__ import annotations
import numpy as np
import random 
from .config import TILE_HEIGHT, TILE_WIDTH


def _update_mask_from_factor(
    mask: np.ndarray,
    x0: int,
    y0: int,
    H_DEFECT: int,
    W_DEFECT: int,
    factor: np.ndarray,
    class_id: int,
    rel_thresh: float = 0.2,
    abs_thresh: float = 0.02,
) -> np.ndarray:
    delta = np.abs(factor - 1.0)
    max_delta = float(delta.max()) if delta.size > 0 else 0.0

    if max_delta <= 0.0 or not np.isfinite(max_delta):
        return mask  

    thr = max(abs_thresh, rel_thresh * max_delta)
    local_mask = delta > thr

    sub = mask[y0:y0 + H_DEFECT, x0:x0 + W_DEFECT]
    sub[local_mask] = class_id
    mask[y0:y0 + H_DEFECT, x0:x0 + W_DEFECT] = sub
    return mask


def apply_defect_overlay(
    canvas: np.ndarray,
    mask: np.ndarray,
    x0: int,
    y0: int,
    y_offset: int = 0,
    class_id: int = 1,   # 1-Dimer-Signatur
):
    """
    1-Dimer-Signatur (einzelner Defekt-Spot).
    """
    H1, W1 = TILE_HEIGHT, TILE_WIDTH
    H_DEFECT, W_DEFECT = 3 * H1, W1
    GAUSS_NORM = np.sqrt(np.pi / 2)
    GAUSS_SCALE, GAUSS_WIDTH, GAUSS_CENTER1, GAUSS_CENTER2 = 28, 7, 4, 14
    GAUSS_SCALE_ALT, GAUSS_WIDTH_ALT, GAUSS_CENTER_ALT = 7, 5, 9

    H, W = canvas.shape[:2]
    if x0 < 0 or y0 < 0 or x0 + W_DEFECT > W or y0 + H_DEFECT > H:
        return canvas, mask

    y_coords = np.arange(y0, y0 + H_DEFECT)
    x_coords = np.arange(x0, x0 + W_DEFECT)
    X, Y = np.meshgrid(x_coords, y_coords)

    A = random.uniform(4, 7)
    T = random.uniform(0.7, 1.3)
    local_X = X - x0

    factor1 = T * GAUSS_SCALE / (GAUSS_WIDTH * GAUSS_NORM)
    factor_alt = T * GAUSS_SCALE_ALT / (GAUSS_WIDTH_ALT * GAUSS_NORM)
    term1 = 1 + factor1 * np.exp(-2 * ((local_X - GAUSS_CENTER1) / GAUSS_WIDTH) ** 2)
    term2 = factor1 * np.exp(-2 * ((local_X - GAUSS_CENTER2) / GAUSS_WIDTH) ** 2)
    term3 = factor_alt * np.exp(
        -2 * ((local_X - GAUSS_CENTER_ALT) / GAUSS_WIDTH_ALT) ** 2
    )
    const1 = 1 + factor1 * np.exp(-2 * ((1 - GAUSS_CENTER1) / GAUSS_WIDTH) ** 2)
    const2 = factor1 * np.exp(-2 * ((1 - GAUSS_CENTER2) / GAUSS_WIDTH) ** 2)
    const3 = factor_alt * np.exp(-2 * ((1 - GAUSS_CENTER_ALT) / GAUSS_WIDTH_ALT) ** 2)
    fx = term1 + term2 + term3 - (const1 + const2 + const3)

    y_center = (H_DEFECT - 1) / 2 + y_offset
    fy = (-7 / (A * GAUSS_NORM)) * np.exp(-2 * ((Y - y0 - y_center) / A) ** 2)
    factor = 1 + fx * fy

    region = canvas[y0 : y0 + H_DEFECT, x0 : x0 + W_DEFECT].astype(np.float64)
    canvas[y0 : y0 + H_DEFECT, x0 : x0 + W_DEFECT] = (
        (region * factor).clip(0, 255).astype(np.uint8)
    )
    mask = _update_mask_from_factor(
        mask, x0, y0, H_DEFECT, W_DEFECT, factor, class_id
    )
    return canvas, mask


def apply_defect_overlay_v1(
    canvas,
    mask,
    x0,
    y0,
    y_offset=0,
    contrast=1.0,
    class_id: int = 2,   # 2-Dimer-Signatur
):
    H1, W1 = TILE_HEIGHT, TILE_WIDTH
    H_DEFECT, W_DEFECT = 4 * H1, W1
    GAUSS_NORM = np.sqrt(np.pi / 2.0)

    H, W = canvas.shape[:2]
    if x0 < 0 or y0 < 0 or x0 + W_DEFECT > W or y0 + H_DEFECT > H:
        return canvas, mask

    y_coords = np.arange(y0, y0 + H_DEFECT)
    x_coords = np.arange(x0, x0 + W_DEFECT)
    X, Y = np.meshgrid(x_coords, y_coords)
    local_X = X - x0
    local_Y = Y - y0

    H_param = random.uniform(0.0, 2.0)
    D = random.uniform(7.0, 13.0)
    E = random.uniform(7.0, 13.0)
    A = random.uniform(0.0, 1.5)
    B = random.uniform(0.0, 1.5)

    Zx = (
        (-3.0 / (4.0 * GAUSS_NORM)) * np.exp(-1.0 * ((local_X - 12.0) / 4.0) ** 2)
        + (-1.0 / (2.0 * GAUSS_NORM)) * np.exp(-1.0 * ((local_X - 14.0) / 2.0) ** 2)
        + (H_param / (2.5 * GAUSS_NORM)) * np.exp(-2.0 * ((local_X - 5.0) / 2.5) ** 2)
    )

    Zy_inner = (
        1.0
        - (-D / (7.0 * GAUSS_NORM)) * np.exp(-2.0 * ((local_Y - 13.0) / 7.0) ** 2)
        + (E  / (7.0 * GAUSS_NORM)) * np.exp(-2.0 * ((local_Y - 23.0) / 7.0) ** 2)
        + (2.0 / (5.0 * GAUSS_NORM)) * np.exp(-2.0 * ((local_Y - 18.0) / 5.0) ** 2)
        + (-A / (2.0 * GAUSS_NORM)) * np.exp(-2.0 * ((local_Y - 5.0)  / 2.0) ** 2)
        + (-B / (2.0 * GAUSS_NORM)) * np.exp(-2.0 * ((local_Y - 31.0) / 2.0) ** 2)
        - 1.00319
    )
    Zy = -Zy_inner

    factor = (1.0 - (Zx * Zy)) * float(contrast)

    if y_offset != 0:
        h = H_DEFECT
        wy = np.linspace(-1, 1, h)[None, :]
        window = np.exp(-((wy - (y_offset / max(1, h))) ** 2) * 2.0).T
        factor *= window

    region = canvas[y0:y0 + H_DEFECT, x0:x0 + W_DEFECT].astype(np.float64)
    region = (region * factor).clip(0, 255)
    canvas[y0:y0 + H_DEFECT, x0:x0 + W_DEFECT] = region.astype(np.uint8)

    mask = _update_mask_from_factor(
        mask, x0, y0, H_DEFECT, W_DEFECT, factor, class_id, rel_thresh=0.35, abs_thresh=0.04,
    )
    return canvas, mask


def apply_defect_overlay_v2(
    canvas,
    mask,
    x0,
    y0,
    y_offset=0,
    contrast=1.0,
    class_id: int = 3,   # Double-Dimer-Signatur
):
    H1, W1_tile = TILE_HEIGHT, TILE_WIDTH
    H_DEFECT, W_DEFECT = 4 * H1, W1_tile
    GAUSS_NORM = np.sqrt(np.pi / 2.0)

    H, W = canvas.shape[:2]
    if x0 < 0 or y0 < 0 or x0 + W_DEFECT > W or y0 + H_DEFECT > H:
        return canvas, mask

    y_coords = np.arange(y0, y0 + H_DEFECT)
    x_coords = np.arange(x0, x0 + W_DEFECT)
    X, Y = np.meshgrid(x_coords, y_coords)
    local_X = X - x0
    local_Y = Y - y0

    S = random.uniform(1.0, 2.0)
    T = random.uniform(1.0, 3.0)
    W1 = random.uniform(11.0, 14.5)
    W2 = random.uniform(21.5, 25.0)

    Zx = (
        (-S * T / (4.0 * GAUSS_NORM)) * np.exp(-T   * ((local_X -  9.5) / 4.0) ** 2)
        + (-T     / (2.0 * GAUSS_NORM)) * np.exp(-0.5 * ((local_X - 14.0) / 2.0) ** 2)
        + (-T   / (2.0 * GAUSS_NORM)) * np.exp(-0.5 * ((local_X -  5.0) / 2.0) ** 2)
    )

    Zy_inner = (
        1.0
        - (-10.0 / (7.0 * GAUSS_NORM)) * np.exp(-2.0 * ((local_Y - W1) / 7.0) ** 2)
        + ( 10.0 / (7.0 * GAUSS_NORM)) * np.exp(-2.0 * ((local_Y - W2) / 7.0) ** 2)
        + ( 4.0 / (5.0 * GAUSS_NORM)) * np.exp(-2.0 * ((local_Y - 18.0) / 5.0) ** 2)
        - 1.00319
    )
    Zy = -Zy_inner

    factor = (1.0 - (Zx * Zy)) * float(contrast)

    if y_offset != 0:
        h = H_DEFECT
        wy = np.linspace(-1, 1, h)[None, :]
        window = np.exp(-((wy - (y_offset / max(1, h))) ** 2) * 2.0).T
        factor *= window

    region = canvas[y0:y0 + H_DEFECT, x0:x0 + W_DEFECT].astype(np.float64)
    region = (region * factor).clip(0, 255)
    canvas[y0:y0 + H_DEFECT, x0:x0 + W_DEFECT] = region.astype(np.uint8)

    mask = _update_mask_from_factor(
        mask, x0, y0, H_DEFECT, W_DEFECT, factor, class_id
    )
    return canvas, mask

