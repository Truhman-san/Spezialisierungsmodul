from __future__ import annotations
from typing import Tuple
import numpy as np
import random

from .config import tile_repetitions_for_canvas


class DimerOrientation(str):
    H = "H"  # Dimerreihen von links nach rechts
    V = "V"  # Dimerreihen von unten nach oben


def _generate_frayed_edge(
    n_cols: int,
    base_row: int,
    min_row: int,
    max_row: int,
    small_step_units: int = 1,
    big_jump_prob: float = 0.2,
    big_jump_min_units: int = 2,
    big_jump_max_units: int = 5,
    max_flat_tiles: int = 30,
) -> np.ndarray:
    edge = np.empty(n_cols, dtype=np.int32)

    # Start auf gerade Zeile snappen
    y = (base_row // 2) * 2

    # Grenzen ebenfalls auf gerade Zeilen snappen
    min_row = (min_row // 2) * 2
    max_row = (max_row // 2) * 2

    # globale Richtung: +1 = nach oben, -1 = nach unten
    direction_units = random.choice([-1, 1])

    flat_run = 0
    moved = False 

    for x in range(n_cols):
        if x == 0:
            edge[x] = y
            flat_run = 1
            continue

        # Standard-Schrittwahl
        if random.random() < big_jump_prob:
            step_units = random.randint(big_jump_min_units, big_jump_max_units)
        else:
            step_units = random.randint(0, small_step_units)

        dy_units = step_units * direction_units
        dy = 2 * dy_units  # 2 Tiles pro Einheit

        new_y = y + dy
        new_y = max(min_row, min(max_row, new_y))
        new_y = (new_y // 2) * 2 

        if new_y == y:
            flat_run += 1
        else:
            flat_run = 0
            moved = True

        if flat_run > max_flat_tiles:
            forced_y = y + 2 * direction_units

            if forced_y < min_row or forced_y > max_row:
                direction_units *= -1
                forced_y = y + 2 * direction_units

            if min_row <= forced_y <= max_row:
                new_y = (forced_y // 2) * 2
                flat_run = 0
                moved = True
            else:
                flat_run = max_flat_tiles

        y = new_y
        edge[x] = y

    if not moved and (max_row - min_row) >= 2:
        x0 = random.randint(n_cols // 4, 3 * n_cols // 4)
        for dir_candidate in (+1, -1):
            y_new = y + 2 * dir_candidate
            if min_row <= y_new <= max_row:
                y_new = (y_new // 2) * 2
                edge[x0:] = y_new
                break

    return edge


def _generate_stair_edge(
    n_cols: int,
    base_row: int,
    min_row: int,
    max_row: int,
    min_plateau: int = 1,
    max_plateau: int = 3,
) -> np.ndarray:
    edge = np.empty(n_cols, dtype=np.int32)

    # Start- und Grenzzeilen auf gerade Tile-Reihen snappen
    y = (base_row // 2) * 2
    min_row = (min_row // 2) * 2
    max_row = (max_row // 2) * 2

    # globale Richtung (±1 Einheit = ±2 Tiles)
    direction_units = random.choice([-1, 1])
    step_tiles = 2  
    moved = False

    x = 0
    while x < n_cols:
        # Plateau-Breite: 2, 4, 6 Tiles
        k = random.randint(min_plateau, max_plateau)
        plateau_width = 2 * k

        end_x = min(n_cols, x + plateau_width)
        edge[x:end_x] = y
        x = end_x

        if x >= n_cols:
            break

        proposed_y = y + step_tiles * direction_units
        if proposed_y < min_row or proposed_y > max_row:
            edge[x:] = y
            break

        # Schritt ausführen
        y = proposed_y
        moved = True

    if not moved and (max_row - min_row) >= step_tiles:
        for dir_candidate in (+1, -1):
            proposed_y = y + step_tiles * dir_candidate
            if min_row <= proposed_y <= max_row:
                y_new = proposed_y
                break
        else:
            return edge

        x0 = random.randint(n_cols // 4, 3 * n_cols // 4)
        edge[:x0] = y      # unteres Plateau
        edge[x0:] = y_new  # oberes Plateau

    return edge



def generate_terrace_layout(
    canvas_size: Tuple[int, int],
    min_steps: int = 1,
    max_steps: int = 5,
    min_vertical_margin_tiles: int = 2,
    min_terrace_gap_tiles: int = 4,
):
    H, W = int(canvas_size[0]), int(canvas_size[1])
    n_rows_tiles, n_cols_tiles = tile_repetitions_for_canvas((H, W))

    num_steps = random.randint(min_steps, max_steps)
    n_terraces = num_steps + 1

    # Orientierung pro Terrasse
    orientations = [random.choice([DimerOrientation.H, DimerOrientation.V])]
    for _ in range(1, n_terraces):
        orientations.append(
            DimerOrientation.H if orientations[-1] == DimerOrientation.V else DimerOrientation.V
        )

    # Basispositionen der Stufen grob verteilen
    usable_rows = n_rows_tiles - 2 * min_vertical_margin_tiles
    if usable_rows <= num_steps + 1:
        base_rows = np.linspace(
            min_vertical_margin_tiles,
            n_rows_tiles - min_vertical_margin_tiles - 1,
            num_steps,
            dtype=int,
        )
    else:
        base_rows = np.linspace(
            min_vertical_margin_tiles,
            n_rows_tiles - min_vertical_margin_tiles - 1,
            num_steps,
            dtype=int,
        )

    # auf gerade Tile-Reihen snappen
    base_rows = (base_rows // 2) * 2

    edges = np.zeros((num_steps, n_cols_tiles), dtype=np.int32)

    # min_gap: mindestens 2 Tiles, gerade Zahl
    min_gap = max(2, (int(min_terrace_gap_tiles) // 2) * 2)

    # ----------------------------------------------------
    # 1) Kanten generieren + untere Abstände erzwingen
    # ----------------------------------------------------
    for k in range(num_steps):
        center = int(base_rows[k])

        if k == 0:
            min_row = min_vertical_margin_tiles
        else:
            min_row = int(base_rows[k - 1]) + min_gap

        if k == num_steps - 1:
            max_row = n_rows_tiles - min_vertical_margin_tiles - 1
        else:
            max_row = int(base_rows[k + 1]) - min_gap

        # Grenzen clampen
        min_row = max(min_row, 0)
        max_row = min(max_row, n_rows_tiles - 1)

        # auf gerade Zeilen snappen
        min_row = (min_row // 2) * 2
        max_row = (max_row // 2) * 2
        center = (center // 2) * 2
        center = max(min_row, min(max_row, center))

        orientation_lower = orientations[k]

        if orientation_lower == DimerOrientation.V:
            big_jump_prob = random.uniform(0.1, 0.4)
            big_jump_max_units = random.randint(2, 5)  
            edge = _generate_frayed_edge(
                n_cols=n_cols_tiles,
                base_row=center,
                min_row=min_row,
                max_row=max_row,
                small_step_units=1,
                big_jump_prob=big_jump_prob,
                big_jump_min_units=2,
                big_jump_max_units=big_jump_max_units,
                max_flat_tiles=10,
            )
        else:
            # Horizontale Dimerreihen unten → stufige Kante
            edge = _generate_stair_edge(
                n_cols=n_cols_tiles,
                base_row=center,
                min_row=min_row,
                max_row=max_row,
                min_plateau=1,
                max_plateau=3,
            )

        edges[k] = edge

        if k > 0:
            if orientation_lower == DimerOrientation.H:
                required = (edges[k - 1] + min_gap) - edges[k]
                delta = int(required.max())
                if delta > 0:
                    edges[k] = edges[k] + delta
            else:
                edges[k] = np.maximum(edges[k], edges[k - 1] + min_gap)

    # ----------------------------------------------------
    # 2) Oberen Abstand erzwingen
    # ----------------------------------------------------
    for k in reversed(range(num_steps - 1)):
        if orientations[k] == DimerOrientation.H:
            diff = (edges[k + 1] - min_gap) - edges[k]
            delta_max = int(diff.min())  
            if delta_max < 0:
                edges[k] = edges[k] + delta_max
        else:
            edges[k] = np.minimum(edges[k], edges[k + 1] - min_gap)

    return edges, orientations, n_rows_tiles, n_cols_tiles
