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


def _refine_snap_to_dimer_line(
    image: np.ndarray,
    x0: int,
    y0: int,
    bbox_w: int,
    bbox_h: int,
    orientation: str,
    search_radius: int = 4,
) -> tuple[int, int]:
    """
    Verfeinert das Grid-Snapping, indem in einem kleinen Fenster entlang
    der Dimer-Normalen nach dem lokalen Helligkeitsmaximum gesucht wird.

    - orientation "H": Dimerreihen horizontal -> entlang y suchen
    - orientation "V": Dimerreihen vertikal   -> entlang x suchen
    """
    H, W = image.shape[:2]
    best_x, best_y = x0, y0
    best_score = -1e9

    if orientation == "H":
        # Suche in y-Richtung (Normalenrichtung)
        y_min = max(0, y0 - search_radius)
        y_max = min(H - bbox_h, y0 + search_radius)
        for yy in range(y_min, y_max + 1):
            patch = image[yy:yy + bbox_h, x0:x0 + bbox_w]
            score = float(patch.mean())
            if score > best_score:
                best_score = score
                best_y = yy
        return best_x, best_y

    else:  # "V"
        # Suche in x-Richtung (Normalenrichtung)
        x_min = max(0, x0 - search_radius)
        x_max = min(W - bbox_w, x0 + search_radius)
        for xx in range(x_min, x_max + 1):
            patch = image[y0:y0 + bbox_h, xx:xx + bbox_w]
            score = float(patch.mean())
            if score > best_score:
                best_score = score
                best_x = xx
        return best_x, best_y


def _snap_to_grid_oriented(
    x0: int,
    y0: int,
    mode: str,
    orientation: str,
) -> tuple[int, int]:
    """
    Orientierungssensitives Snapping auf das Dimer-Raster.

    - Für H-Terrassen: SNAP aus _MODE_SNAP (dx = TILE_WIDTH, dy = 3/4*TILE_HEIGHT)
    - Für V-Terrassen: dx/dy vertauscht, weil das Dimer-Template um 90° gedreht wurde
    """
    if orientation == "H":
        dx, dy = _MODE_SNAP[mode]
    else:
        # vertikale Terrassen -> Dimer-Template gedreht -> Rasterachsen vertauschen
        dx, dy = _MODE_SNAP[mode][1], _MODE_SNAP[mode][0]

    sx = ((x0 - _LATTICE_PHASE_X) // dx) * dx + _LATTICE_PHASE_X
    sy = ((y0 - _LATTICE_PHASE_Y) // dy) * dy + _LATTICE_PHASE_Y
    return int(sx), int(sy)


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
    n_defects_range=(3, 8),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Legt mehrere zufällige Defekte (gemischte Typen) auf die Oberfläche.
    - nutzt place_single_defect -> korrekte Ausrichtung auf dem Dimer-Raster
    """
    n_def = np.random.randint(n_defects_range[0], n_defects_range[1] + 1)
    modes = ("gaussian", "v1", "v2")

    for _ in range(n_def):
        mode = random.choices(modes, weights=(0.5, 0.3, 0.2), k=1)[0]
        canvas, mask = place_single_defect(canvas, mask, mode)

    return canvas, mask


def apply_random_signatures_oriented(
    canvas: np.ndarray,
    mask: np.ndarray,
    terrace_ids: np.ndarray,
    terrace_orient: np.ndarray,
    n_defects_range=(3, 10),
    min_distance_dimers: int = 2,
):
    """
    Platzierung zufälliger Signaturen auf Terrassen:

    - für v1/v2:
        * Grid-Snap (MODE_SNAP)
        * Refinement auf Dimerreihe über _refine_snap_to_dimer_line (Helligkeitsmaximum)

    - für gaussian (1-Dimer):
        * Grid-Snap
        * KEIN Helligkeits-Refinement
        * Stattdessen: Zentrum strikt auf Dimer-Reihe snappen
          (Reihenabstand = TILE_HEIGHT; Boxhöhe = 3*TILE_HEIGHT)
    """
    H, W = canvas.shape
    margin_px = min_distance_dimers * TILE_HEIGHT
    n_def = random.randint(*n_defects_range)

    modes = ("gaussian", "v1", "v2")

    for _ in range(n_def):
        mode = random.choices(modes, weights=(0.5, 0.3, 0.2))[0]

        # Zufällige Pixelposition
        x = random.randint(0, W - 1)
        y = random.randint(0, H - 1)

        # Welche Terrasse liegt hier?
        t_id = terrace_ids[y, x]
        if t_id < 0:
            continue

        ori = terrace_orient[y, x]   # "H" oder "V"

        # BBOX & SNAP je nach Orientierung
        if ori == "H":
            bbox_h = _MODE_BBOX[mode][0]
            bbox_w = _MODE_BBOX[mode][1]
            snap_dx, snap_dy = _MODE_SNAP[mode]

            defect_func = _APPLIERS[mode]

        else:  # "V" – vertikale Dimerreihen
            bbox_h = _MODE_BBOX[mode][1]  # vertauscht
            bbox_w = _MODE_BBOX[mode][0]
            snap_dx = _MODE_SNAP[mode][1]
            snap_dy = _MODE_SNAP[mode][0]

            def defect_func(img, msk, xx, yy):
                H2, W2 = img.shape[:2]

                img2 = np.rot90(img)
                msk2 = np.rot90(msk)

                # Welt (xx, yy) -> Koordinate im rotierten Bild
                yy_r = W2 - 1 - xx
                xx_r = yy

                img2, msk2 = _APPLIERS[mode](img2, msk2, xx_r, yy_r)

                img2 = np.rot90(img2, 3)
                msk2 = np.rot90(msk2, 3)
                return img2, msk2

        # -----------------------------
        # 1) Grid-Snap (Tile-Raster)
        # -----------------------------
        x0 = (x // snap_dx) * snap_dx
        y0 = (y // snap_dy) * snap_dy

        # Clamp an Bildgrenzen
        x0 = max(0, min(W - bbox_w, x0))
        y0 = max(0, min(H - bbox_h, y0))

        # -----------------------------
        # 2) Reihen-Snap je nach Modus
        # -----------------------------
        if mode == "gaussian":
            # 1-Dimer: Zentrum auf Dimer-Reihe snappen
            # + kleiner Drift (ca. 2% der Defekthöhe) hoch/runter.

            DRIFT_FRACTION = 0.02  # ~2% Drift

            if ori == "H":
                # Reihen horizontal -> periodisch in Y mit TILE_HEIGHT
                H_DEFECT = _MODE_BBOX["gaussian"][0]  # 3 * TILE_HEIGHT
                cy = y0 + H_DEFECT // 2

                # nächstes Dimer-Reihenzentrum: k*TILE_HEIGHT + TILE_HEIGHT//2
                row_idx = int(round((cy - TILE_HEIGHT // 2) / TILE_HEIGHT))
                row_center = row_idx * TILE_HEIGHT + TILE_HEIGHT // 2

                y0 = row_center - H_DEFECT // 2

                # kleiner Drift in Normalenrichtung (hoch/runter)
                drift_max = max(1, int(round(DRIFT_FRACTION * H_DEFECT)))
                drift = random.randint(-drift_max, drift_max)
                y0 += drift

            else:
                # ori == "V": Reihen vertikal -> periodisch in X mit TILE_HEIGHT
                W_DEFECT = bbox_w  # Breite entlang der Reihen-Normalen
                cx = x0 + W_DEFECT // 2

                col_idx = int(round((cx - TILE_HEIGHT // 2) / TILE_HEIGHT))
                col_center = col_idx * TILE_HEIGHT + TILE_HEIGHT // 2

                x0 = col_center - W_DEFECT // 2

                # kleiner Drift in Normalenrichtung (links/rechts)
                drift_max = max(1, int(round(DRIFT_FRACTION * W_DEFECT)))
                drift = random.randint(-drift_max, drift_max)
                x0 += drift

            # clampen nach Drift
            x0 = max(0, min(W - bbox_w, x0))
            y0 = max(0, min(H - bbox_h, y0))

        # -----------------------------
        # 3) Terrassengrenzen-Check
        # -----------------------------
        if terrace_ids is not None:
            cx = x0 + bbox_w // 2
            cy = y0 + bbox_h // 2

            if cx < 0 or cy < 0 or cx >= W or cy >= H:
                continue

            tid_center = terrace_ids[cy, cx]
            if tid_center < 0:
                continue

            ys = max(0, y0 - margin_px)
            ye = min(H, y0 + bbox_h + margin_px)
            xs = max(0, x0 - margin_px)
            xe = min(W, x0 + bbox_w + margin_px)

            region_ids = terrace_ids[ys:ye, xs:xe]
            if not np.all(region_ids == tid_center):
                # zu nah an Terrassenkante
                continue

        # -----------------------------
        # 4) Defekt anwenden
        # -----------------------------
        canvas, mask = defect_func(canvas, mask, x0, y0)

    return canvas, mask
