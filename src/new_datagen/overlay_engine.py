from __future__ import annotations
import random
import numpy as np
from .config import TILE_HEIGHT, TILE_WIDTH
from .overlays import (
    apply_defect_overlay as gaussian_one_pair,
    apply_defect_overlay_v1,
    apply_defect_overlay_v2,
)

# --- Welche Signatur gehört wohin ---
_APPLIERS = {
    "gaussian": gaussian_one_pair,      # 1-Dimer
    "v1": apply_defect_overlay_v1,         # 2-Dimer
    "v2": apply_defect_overlay_v2,         # Double-Dimer
}

# --- Größe der Signatur-Box ---
_MODE_BBOX = {
    "gaussian": (3 * TILE_HEIGHT, TILE_WIDTH),
    "v1": (4 * TILE_HEIGHT, TILE_WIDTH),
    "v2": (4 * TILE_HEIGHT, TILE_WIDTH),
}

# --- Snap-Raster pro Defekt ---
_MODE_SNAP = {
    "gaussian": (TILE_WIDTH, 3 * TILE_HEIGHT),
    "v1":       (TILE_WIDTH, 4 * TILE_HEIGHT),
    "v2":       (TILE_WIDTH, 4 * TILE_HEIGHT),
}

# --- Optionale Tile-Phasen (falls Base-Tile nicht bei (0,0) beginnt) ---
_LATTICE_PHASE_X = 0
_LATTICE_PHASE_Y = 0


def _snap_to_grid(x0: int, y0: int, mode: str) -> tuple[int, int]:
    dx, dy = _MODE_SNAP[mode]
    sx = ((x0 - _LATTICE_PHASE_X) // dx) * dx + _LATTICE_PHASE_X
    sy = ((y0 - _LATTICE_PHASE_Y) // dy) * dy + _LATTICE_PHASE_Y
    return int(sx), int(sy)


def _clamp_to_fit(H: int, W: int, x0: int, y0: int, w: int, h: int):
    x0 = min(max(0, x0), max(0, W - w))
    y0 = min(max(0, y0), max(0, H - h))
    return x0, y0


def place_single_defect(canvas, mask, mode: str) -> tuple[np.ndarray, np.ndarray]:
    H, W = canvas.shape[:2]
    h, w = _MODE_BBOX[mode]

    # Zufallsstart
    x0 = random.randint(0, W - w)
    y0 = random.randint(0, H - h)

    # AUF DIMERRASTER SNAP!!  <-- wichtigste Zeile
    x0, y0 = _snap_to_grid(x0, y0, mode)
    x0, y0 = _clamp_to_fit(H, W, x0, y0, w, h)

    # Defekt anwenden
    return _APPLIERS[mode](canvas, mask, x0, y0)



def apply_random_signatures(
    canvas: np.ndarray,
    mask: np.ndarray,
    n_defects_range=(3, 8),   # min 3, max 8 Defekte
) -> tuple[np.ndarray, np.ndarray]:
    """
    Legt mehrere zufällige Defekte (gemischte Typen) auf die Oberfläche.
    canvas: Bild
    mask:   Labelbild (0 Background, 1/2/3 Signaturen)
    """
    H, W = canvas.shape[:2]

    n_def = np.random.randint(n_defects_range[0], n_defects_range[1] + 1)

    modes = ("gaussian", "v1", "v2")

    for _ in range(n_def):
        mode = random.choices(modes, weights=(0.5, 0.3, 0.2), k=1)[0]

        if mode == "gaussian":
            # 3*TILE_HEIGHT x TILE_WIDTH
            h = 3 * TILE_HEIGHT
            w = TILE_WIDTH
            cls_id = 1
            applier = gaussian_one_pair
        elif mode == "v1":
            # 4*TILE_HEIGHT x TILE_WIDTH
            h = 4 * TILE_HEIGHT
            w = TILE_WIDTH
            cls_id = 2
            applier = apply_defect_overlay_v1
        else:
            h = 4 * TILE_HEIGHT
            w = TILE_WIDTH
            cls_id = 3
            applier = apply_defect_overlay_v2

        # zufällige Startposition (vor Snap)
        x0 = random.randint(0, max(0, W - w))
        y0 = random.randint(0, max(0, H - h))

        # auf Dimerraster snappen (wie in deinem alten Code)
        # hier minimal: nur in X auf TILE_WIDTH, in Y auf TILE_HEIGHT oder 3/4*TILE_HEIGHT
        x0 = (x0 // TILE_WIDTH) * TILE_WIDTH
        y0 = (y0 // TILE_HEIGHT) * TILE_HEIGHT

        # Hard-Bounds
        x0 = min(max(0, x0), max(0, W - w))
        y0 = min(max(0, y0), max(0, H - h))

        # anwenden – wichtig: class_id an Overlay übergeben
        canvas, mask = applier(
            canvas,
            mask,
            x0,
            y0,
            class_id=cls_id,   # sicherstellen, dass die Funktion das unterstützt
        )

    return canvas, mask