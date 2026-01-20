from __future__ import annotations
from typing import Tuple
import random
import numpy as np
import cv2

from .config import (
    CANVAS_SIZE,
    BASE_TILE,
    TILE_HEIGHT,
    TILE_WIDTH,
)


def remove_thin_steps(tile_labels: np.ndarray, min_thickness: int = 2) -> np.ndarray:
    """
    Entfernt Terrassenstufen, die dünner sind als 'min_thickness' Tiles
    entlang der Normalrichtung.
    """
    labels = tile_labels.copy()
    H, W = labels.shape
    unique = np.unique(labels)

    for t in unique:
        mask = (labels == t).astype(np.uint8)

        # Connected components
        num, cc = cv2.connectedComponents(mask, connectivity=8)
        if num <= 2:
            continue

        # prüfe jede Komponente
        for cid in range(1, num):
            comp = (cc == cid)
            ys, xs = np.where(comp)

            # bounding box der Komponente
            h = ys.max() - ys.min() + 1
            w = xs.max() - xs.min() + 1

            thin = (h < min_thickness) or (w < min_thickness)
            if thin:
                # Nachbarlabels sammeln
                neighbors = []
                for y, x in zip(ys, xs):
                    for dy, dx in ((-1,0),(1,0),(0,-1),(0,1)):
                        yy, xx = y+dy, x+dx
                        if 0 <= yy < H and 0 <= xx < W and not comp[yy,xx]:
                            neighbors.append(labels[yy,xx])

                if neighbors:
                    new_label = int(np.bincount(neighbors).argmax())
                    labels[comp] = new_label

    return labels


def perturb_tile_labels(
    tile_labels: np.ndarray,
    n_passes: int = 1,
    move_prob: float = 0.35,
) -> np.ndarray:
    labels = tile_labels.copy()
    n_rows, n_cols = labels.shape
    max_label = int(labels.max())
    if max_label <= 0:
        return labels

    for _ in range(n_passes):
        noise = np.random.randn(n_rows, n_cols).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (7, 7), 0)

        for t in range(1, max_label): 
            for r in range(n_rows):
                for c in range(n_cols):
                    if labels[r, c] != t:
                        continue

                    val = noise[r, c]

                    if val > 0.35 and random.random() < move_prob:
                        # t wächst nach oben zu t+1
                        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                            rr, cc = r + dr, c + dc
                            if 0 <= rr < n_rows and 0 <= cc < n_cols and labels[rr, cc] == t + 1:
                                labels[r, c] = t + 1
                                break

                    elif val < -0.35 and random.random() < move_prob:
                        # t schrumpft nach unten zu t-1
                        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                            rr, cc = r + dr, c + dc
                            if 0 <= rr < n_rows and 0 <= cc < n_cols and labels[rr, cc] == t - 1:
                                labels[r, c] = t - 1
                                break

    return labels


def smooth_terrace_boundaries(labels: np.ndarray,
                              iters: int = 1) -> np.ndarray:
    """
    Glättet zackige Terrassenkanten mit einem 3x3-Mehrheitsfilter.
    Entfernt 1-Pixel-Ausreißer direkt an den Kanten.
    """
    import scipy.ndimage as ndi

    out = labels.copy()
    for _ in range(iters):
        # 3x3 Mehrheitsfilter
        def majority_filter(window):
            w = window.astype(np.int64)
            return np.bincount(w).argmax()

        out = ndi.generic_filter(
            out,
            majority_filter,
            size=3,
            mode="nearest"
        )

    return out


def remove_island_fragments(tile_labels):
    H, W = tile_labels.shape
    cleaned = tile_labels.copy()
    unique = np.unique(tile_labels)

    for t in unique:
        mask = (tile_labels == t).astype(np.uint8)
        if mask.sum() == 0:
            continue

        # Connected components
        num_labels, cc = cv2.connectedComponents(mask, connectivity=8)
        if num_labels <= 2:
            # entweder nur Hintergrund + eine Komponente → ok
            continue

        # größte Komponente finden
        sizes = [(cc == i).sum() for i in range(1, num_labels)]
        main_id = 1 + int(np.argmax(sizes))

        # kleine Komponenten remappen
        for comp_id in range(1, num_labels):
            if comp_id == main_id:
                continue

            comp_mask = (cc == comp_id)

            # angrenzende Labels bestimmen
            ys, xs = np.where(comp_mask)
            neighbors = []
            for y, x in zip(ys, xs):
                for dy, dx in ((-1,0),(1,0),(0,-1),(0,1)):
                    yy, xx = y+dy, x+dx
                    if 0 <= yy < H and 0 <= xx < W:
                        if not comp_mask[yy, xx]:
                            neighbors.append(cleaned[yy, xx])

            if neighbors:
                new_label = int(np.bincount(neighbors).argmax())
            else:
                # fallback: zum nächstliegenden Label
                new_label = t-1 if t > 0 else t+1

            cleaned[comp_mask] = new_label

    return cleaned


def generate_parallel_step_tile_labels(
    n_rows_tiles: int,
    n_cols_tiles: int,
    min_plateau: int = 1,
    max_plateau: int = 3,
) -> tuple[np.ndarray, int]:
    """
    Erzeugt ein Tile-Label-Feld mit GENAU EINER Stufe, deren Kante
    parallel zu den Dimerreihen verläuft (horizontal im Tile-Raster).

    Eigenschaften:
    - Zwei Terrassen (Labels 0 und 1)
    - Kantenverlauf ist stufig:
        * jedes Plateau ist 1–3 Tiles breit
        * die Stufe wandert nur in EINE Richtung (monoton)
    - Kante liegt strikt auf Tile-Grenzen -> keine halben Dimere.
    """
    tile_labels = np.zeros((n_rows_tiles, n_cols_tiles), dtype=np.int32)

    if n_rows_tiles < 2:
        # Degenerierter Fall: alles ein Label
        return tile_labels, 1

    # Stufe irgendwo "mittig" starten, damit oben & unten Platz ist
    start_row = random.randint(1, n_rows_tiles - 1)
    cur_row = start_row

    # Richtung der Stufe festlegen: nach oben (-1) oder nach unten (+1)
    direction = random.choice([-1, 1])

    col = 0
    while col < n_cols_tiles:
        # Plateaubreite in Tiles (1–3)
        plateau = random.randint(min_plateau, max_plateau)
        end_col = min(n_cols_tiles, col + plateau)

        # Für diese Spalten ist die Kante zwischen cur_row-1 und cur_row
        # -> oben: Label 0, unten: Label 1 (oder umgekehrt, wenn du willst)
        # hier: 0 unten, 1 oben oder umgekehrt? Du kannst das nach Bedarf drehen.
        # Ich nehme: 0 unten, 1 oben.
        tile_labels[:cur_row, col:end_col] = 0
        tile_labels[cur_row:, col:end_col] = 1

        col = end_col
        if col >= n_cols_tiles:
            break

        # Nächste Stufe: eine Zeile rauf oder runter (monoton)
        next_row = cur_row + direction

        # Wenn wir an den Rand kommen, keine weiteren Sprünge mehr
        if next_row <= 0 or next_row >= n_rows_tiles:
            # Rest als letztes Plateau mit konstanter Höhe
            tile_labels[:cur_row, col:] = 0
            tile_labels[cur_row:, col:] = 1
            break

        cur_row = next_row

    return tile_labels, 2


def generate_terrace_heightmap(
    H: int,
    W: int,
    max_steps: int = 4,
    step_height_range: Tuple[float, float] = (0.4, 1.0),
    use_parallel_step: bool = False,  # NEU
) -> tuple[np.ndarray, np.ndarray]:
    """
    Erzeugt eine Terrassen-Heightmap + Label-Map im Pixelraum.

    Neues Konzept:
    - Arbeitet im Tile-Raster (TILE_HEIGHT x TILE_WIDTH).
    - Baut eine 2D-Höhenfunktion (globaler Trend + glattes Noise).
    - Quantisiert diese Höhenfunktion in n_terraces Stufen.
    - Upsampling auf Pixel-Grid => Kanten immer auf ganzen Tiles,
      also nie durch halbe Dimere.
    """
    H = int(H)
    W = int(W)

    # --- 1) Tile-Raster-Größe ---
    n_rows_tiles = int(np.ceil(H / TILE_HEIGHT))
    n_cols_tiles = int(np.ceil(W / TILE_WIDTH))

    if use_parallel_step:
        # SPEZIALFALL: horizontale Dimerreihen -> stufige Kante
        tile_labels, n_terraces = generate_parallel_step_tile_labels(
            n_rows_tiles,
            n_cols_tiles,
            min_plateau=1,
            max_plateau=3,
        )
    else:
        # DEIN bisheriger Code: 2D-Höhenfunktion + Noise + Quantisierung
        # --- 2) Anzahl Terrassen ---
        if max_steps < 1:
            max_steps = 1
        n_terraces = random.randint(2, max_steps + 1)

        # --- 3) 2D-Höhenfunktion im Tile-Raster aufbauen ---
        yy, xx = np.mgrid[0:n_rows_tiles, 0:n_cols_tiles]
        yy = yy.astype(np.float32)
        xx = xx.astype(np.float32)

        angle = random.uniform(0, 2 * np.pi)
        grad_strength = random.uniform(0.1, 0.4)
        gx = np.cos(angle) * grad_strength
        gy = np.sin(angle) * grad_strength
        base_grad = gx * xx + gy * yy

        noise1 = np.random.randn(n_rows_tiles, n_cols_tiles).astype(np.float32)
        noise1 = cv2.GaussianBlur(noise1, (5, 5), 0)

        noise2 = np.random.randn(n_rows_tiles, n_cols_tiles).astype(np.float32)
        noise2 = cv2.GaussianBlur(noise2, (9, 9), 0)

        base_grad = base_grad + 1.0 * noise1 + 0.7 * noise2

        g_min = float(base_grad.min())
        g_max = float(base_grad.max())
        if g_max > g_min:
            base_grad = (base_grad - g_min) / (g_max - g_min + 1e-6)
        else:
            base_grad[:] = 0.5

        thresholds = sorted(random.uniform(0.0, 1.0) for _ in range(n_terraces - 1))
        tile_labels = np.digitize(base_grad, thresholds).astype(np.int32)

        n_passes = random.randint(2, 4)
        move_prob = random.uniform(0.4, 0.7)
        tile_labels = perturb_tile_labels(tile_labels, n_passes=n_passes, move_prob=move_prob)
        tile_labels = remove_thin_steps(tile_labels, min_thickness=2)
        tile_labels = remove_island_fragments(tile_labels)

        # --- 4) Quantisierung in diskrete Terrassen ---
        # n_terraces-1 zufällige Schwellwerte in (0,1)
        thresholds = sorted(random.uniform(0.0, 1.0) for _ in range(n_terraces - 1))
        tile_labels = np.digitize(base_grad, thresholds).astype(np.int32)

        # mehr Durchgänge + höhere Move-Probability für extreme Klippen
        n_passes = random.randint(2, 4)      # statt 1
        move_prob = random.uniform(0.4, 0.7) # statt fix 0.35

        tile_labels = perturb_tile_labels(tile_labels, n_passes=n_passes, move_prob=move_prob)
        tile_labels = remove_thin_steps(tile_labels, min_thickness=2)
        tile_labels = remove_island_fragments(tile_labels)

        # --- 4b) Sanity-Check: Mindestens zwei Terrassen erzwingen ---
        unique = np.unique(tile_labels)
        if unique.size < 2:
            # Notfall-Fall: eine horizontale Stufe ins Tile-Raster schneiden
            n_rows_tiles, n_cols_tiles = tile_labels.shape

            if n_rows_tiles >= 2:
                split_row = random.randint(1, n_rows_tiles - 1)

                # zwei Labels erzwingen: 0 unten, 1 oben
                tile_labels[:split_row, :] = 0
                tile_labels[split_row:, :] = 1
            else:
                # fallback: vertikale Stufe
                split_col = random.randint(1, n_cols_tiles - 1)
                tile_labels[:, :split_col] = 0
                tile_labels[:, split_col:] = 1

            # 2 Terrassen
            n_terraces = 2

        # --- 5) Terrassen-Höhen definieren ---
        base_height = random.uniform(0.1, 0.3)
        step_height = random.uniform(*step_height_range)
        terrace_heights = base_height + step_height * np.arange(n_terraces, dtype=np.float32)

        # --- 6) Upsampling auf Pixel-Grid ---
        labels = np.zeros((H, W), dtype=np.int32)
        z = np.zeros((H, W), dtype=np.float32)
        labels = smooth_terrace_boundaries(labels, iters=1)

        for r in range(n_rows_tiles):
            y0 = r * TILE_HEIGHT
            y1 = min(H, (r + 1) * TILE_HEIGHT)
            if y0 >= H:
                continue
            for c in range(n_cols_tiles):
                x0 = c * TILE_WIDTH
                x1 = min(W, (c + 1) * TILE_WIDTH)
                if x0 >= W:
                    continue

                t = int(tile_labels[r, c])
                # Safety, falls mal ein Level > n_terraces-1 entsteht (sollte nicht passieren)
                t = max(0, min(t, n_terraces - 1))

                labels[y0:y1, x0:x1] = t
                z[y0:y1, x0:x1] = terrace_heights[t]

        # --- 7) z auf [0, 1] normalisieren ---
        z_min = float(z.min())
        z_max = float(z.max())
        if z_max > z_min:
            z = (z - z_min) / (z_max - z_min)
        else:
            z[:] = 0.5

    return z.astype(np.float32), labels.astype(np.int32)


def jitter_tile(tile):
    t = tile.astype(np.float32)

    drift = np.random.uniform(-0.05, 0.05)
    t = t * (1.0 + drift)

    row_noise = np.random.normal(0, 3, size=(t.shape[0],1))
    t = t + row_noise

    # Blur ENTLANG der Dimerachse (hier: horizontal)
    t = cv2.GaussianBlur(t, (3,1), 0)   # statt (1,3)

    return np.clip(t, 0, 255).astype(np.uint8)


def apply_dimer_amplitude_noise(
    image: np.ndarray,
    sigma: float = 0.06,
    blur_ksize: int = 3,
) -> np.ndarray:
    """
    Legt ein weiches 2D-Noisefeld auf die Dimeramplitude.

    - sigma: Stärke der Amplitudenschwankung (≈ 0.03–0.10 sinnvoll)
    - blur_ksize: bestimmt die Korrelation (3 -> 1–3 px, also gut für DL)

    Ergebnis: lokale Variation der Dimerhöhe, ohne Kanten zu verschmieren.
    """
    import cv2

    img = image.astype(np.float32) / 255.0

    # weißes Rauschen
    noise = np.random.randn(*img.shape).astype(np.float32)

    # glätten -> lokale Korrelation über ein paar Pixel
    if blur_ksize is not None and blur_ksize > 1:
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        noise = cv2.GaussianBlur(noise, (blur_ksize, blur_ksize), 0)

    # normalisieren auf Mittelwert 0, Varianz 1
    m = float(noise.mean())
    s = float(noise.std() + 1e-6)
    noise = (noise - m) / s

    # Amplitudenfeld aufbauen: 1 ± sigma * noise
    amp = 1.0 + sigma * noise

    # zu wilde Extremwerte begrenzen (3σ)
    amp_min = 1.0 - 3.0 * sigma
    amp_max = 1.0 + 3.0 * sigma
    amp = np.clip(amp, amp_min, amp_max)

    out = img * amp
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return out


def render_dimers_for_terraces(
    labels: np.ndarray,
    base_tile: np.ndarray,
) -> np.ndarray:
    """
    Rendert Dimerpattern pro Terrasse *maskenbasiert*.

    - Jede Terrasse bekommt ein eigenes Dimerfeld.
    - Orientierung:
        - Terrassen mit geradem Label (0, 2, 4, ...)  -> 0° (BASE_TILE)
        - Terrassen mit ungeradem Label (1, 3, 5, ...) -> 90° (np.rot90(BASE_TILE))
    - Es wird NICHT mehr mit cv2 rotiert, dadurch keine Misch-/Alias-Artefakte.
    """

    if labels.ndim != 2:
        raise ValueError("labels muss 2D (H, W) sein.")

    H, W = labels.shape
    canvas = np.zeros((H, W), dtype=np.float32)

    unique_labels = np.unique(labels)

    for t in unique_labels:
        mask = (labels == t)
        if not np.any(mask):
            continue

        # Bounding Box der Terrasse
        ys, xs = np.where(mask)
        y_min, y_max = ys.min(), ys.max() + 1
        x_min, x_max = xs.min(), xs.max() + 1

        box_h = y_max - y_min
        box_w = x_max - x_min

        # Orientierung nach Terrassen-Label:
        # gerade -> 0°, ungerade -> 90°
        if int(t) % 2 == 0:
            tile_used = jitter_tile(base_tile)
        else:
            tile_used = jitter_tile(np.rot90(base_tile))

        # Wiederholen, bis Bounding Box abgedeckt ist
        tile_h, tile_w = tile_used.shape

        reps_h = (box_h // tile_h) + 2
        reps_w = (box_w // tile_w) + 2

        tile_big = np.tile(tile_used, (reps_h, reps_w))


        # exakt auf Bounding Box zuschneiden
        crop = tile_big[:box_h, :box_w]

        # In Canvas nur an Masken-Pixeln schreiben
        region_mask = mask[y_min:y_max, x_min:x_max]          # (box_h, box_w)
        region = canvas[y_min:y_max, x_min:x_max]             # (box_h, box_w)
        region[region_mask] = crop[region_mask]               # korrekt (gleiche shape!)


    return np.clip(canvas, 0, 255).astype(np.uint8)


def boost_contrast(x, strength=1.4):
    # strength > 1 → mehr Punch
    mid = 0.5
    return mid + (x - mid) * strength


def compute_terrace_edge_field(
    labels: np.ndarray,
    min_val: float = 0.20,
    max_val: float = 0.85,
    gamma: float = 1.0,
) -> np.ndarray:
    """
    Berechnet ein Helligkeitsfeld für jede Terrasse:
    oben hell (max_val), unten dunkel (min_val).
    Der Gradient geht vollständig über die gesamte Terrasse.
    """

    if labels.ndim != 2:
        raise ValueError("labels muss 2D sein.")

    H, W = labels.shape
    grad = np.ones((H, W), dtype=np.float32) * ((min_val + max_val) * 0.5)

    for t in np.unique(labels):
        mask = (labels == t)
        if not np.any(mask):
            continue

        for x in range(W):
            ys = np.where(mask[:, x])[0]
            if ys.size == 0:
                continue

            y_top = ys.min()
            y_bottom = ys.max()

            if y_top == y_bottom:
                grad[y_top, x] = max_val
                continue

            span = float(y_bottom - y_top)
            alphas = (ys - y_top) / span  # nur die Indizes der Terrasse
            col_vals = max_val - (alphas ** gamma) * (max_val - min_val)

            grad[ys, x] = col_vals

    return grad


def apply_terrace_edge_gradient(
    image: np.ndarray,
    labels: np.ndarray,
    blend: float = 0.5,
    min_val: float = 0.0,
    max_val: float = 1.00,
) -> np.ndarray:
    """
    Wendet den Terrassen-Gradienten auf ein beliebiges Bild an, indem
    Bild und Gradient gemischt werden.

    - image: uint8, [0,255]
    - labels: Terrassen-Labels
    - blend: 0 -> nur Bild, 1 -> nur Gradient
    - min_val/max_val: grobe Zielhelligkeiten unten/oben

    Neu:
    - Gradient wird immer auf [0,1] normalisiert
    - in einem Teil der Fälle wird ein extremer Kontrastmodus aktiviert:
      Kante fast weiß, gegenüberliegende Seite fast schwarz
    """

    if image.shape != labels.shape:
        raise ValueError("image und labels müssen gleiche Shape haben.")

    # 1) Bild in [0,1]
    img = image.astype(np.float32) / 255.0

    # 2) Terrassen-Gradient berechnen (liefert grob min_val..max_val)
    grad = compute_terrace_edge_field(
        labels,
        min_val=min_val,
        max_val=max_val,
        gamma=1.0,
    )

    grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-6)

    # 3) Gradient sauber auf [0,1] normalisieren
    g_min = float(grad.min())
    g_max = float(grad.max())
    if g_max > g_min:
        grad = (grad - g_min) / (g_max - g_min + 1e-6)
    else:
        grad[:] = 0.5

    # 4) Baseline: etwas nichtlinear machen (leichte Verstärkung der Kanten)
    base_gamma = random.uniform(1.0, 2.0)
    grad = np.clip(grad, 0.0, 1.0) ** base_gamma

    # 5) Mit gewisser Wahrscheinlichkeit EXTREM-Fall erzwingen:
    #    eine Seite sehr hell, andere sehr dunkel.
    extreme_prob = 0.30  # 30 % der Bilder werden "krass"
    local_blend = float(blend)

    if random.random() < extreme_prob:
        # a) starken Kontrast über exponentielle Verzerrung
        extreme_gamma = random.uniform(3.0, 6.0)
        grad = np.clip(grad, 0.0, 1.0) ** extreme_gamma

        # b) erneut auf [0,1] normalisieren, damit volle Spreizung genutzt wird
        g_min = float(grad.min())
        g_max = float(grad.max())
        if g_max > g_min:
            grad = (grad - g_min) / (g_max - g_min + 1e-6)
        else:
            grad[:] = 0.5

        # c) Hälfte der Fälle invertieren -> mal oben hell, mal unten hell
        if random.random() < 0.5:
            grad = 1.0 - grad

        # d) Blend lokal erhöhen, damit der Gradient wirklich dominiert
        local_blend = max(local_blend, random.uniform(0.7, 1.0))

    # 6) Blend auf gültigen Bereich clampen
    local_blend = float(max(0.0, min(1.0, local_blend)))

    # 7) Mischung Bild + Gradient
    out = (1.0 - local_blend) * img + local_blend * grad
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)

    return out


def soften_and_flatten_contrast(
    image: np.ndarray,
    blur_ksize: int = 3,
    lift_black: float = 0.30,
    reduce_range: float = 0.20,
) -> np.ndarray:
    """
    Postprocessing für die fertige Surface:

    - hebt die sehr dunklen Zwischenräume etwas an (lift_black)
    - reduziert den Dynamikbereich (reduce_range)
    - verwischt Dimere leicht (Gaussian Blur), damit sie weniger hart getrennt sind
    """
    import cv2

    img = image.astype(np.float32) / 255.0

    # 1) Dynamikbereich reduzieren: Werte Richtung Mittelwert ziehen
    mean_val = float(img.mean())
    img = mean_val + (img - mean_val) * (1.0 - reduce_range)

    # 2) Schwarze Bereiche anheben (Zwischenräume werden heller)
    img = img + lift_black * (1.0 - img)  # 0 -> lift_black, 1 -> 1

    # 3) Leicht weichzeichnen, damit Dimere "verschmelzen"
    if blur_ksize is not None and blur_ksize > 1:
        # Kernelgröße muss ungerade sein
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        img = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)

    img = np.clip(img * 255.0, 0, 255)
    return img.astype(np.uint8)


def apply_terrace_edge_gradient(
    image: np.ndarray,
    labels: np.ndarray,
    blend: float = 0.5,
    min_val: float = 0.20,
    max_val: float = 0.85,
    edge_blur_ksize: int = 5,
    edge_blur_strength: float = 1.0,
) -> np.ndarray:
    """
    Wendet den Terrassen-Gradienten auf ein beliebiges Bild an, indem
    Bild und Gradient *additiv geblendet* werden.

    NEU:
    - Terrassenkanten werden lokal geglättet:
      Der Gradient wird nur in Kanten-Nähe weichgezeichnet,
      so dass die Kante weniger hart, aber weiterhin sichtbar ist.
    """

    import cv2

    if image.shape != labels.shape:
        raise ValueError("image und labels müssen gleiche Shape haben.")

    # 1) Normiertes Bild
    img = image.astype(np.float32) / 255.0

    # 2) Gradientfeld erzeugen (Terrassen oben hell, unten dunkel)
    grad = compute_terrace_edge_field(labels, min_val=min_val, max_val=max_val)

    # 3) Terrassenkanten aus labels bestimmen
    #    -> überall dort, wo sich Nachbarn im Label unterscheiden
    lab = labels.astype(np.int32)
    edge_mask = np.zeros_like(lab, dtype=np.uint8)

    # horizontale Kanten
    edge_mask[:, 1:] |= (lab[:, 1:] != lab[:, :-1]).astype(np.uint8)
    edge_mask[:, :-1] |= (lab[:, 1:] != lab[:, :-1]).astype(np.uint8)
    # vertikale Kanten
    edge_mask[1:, :] |= (lab[1:, :] != lab[:-1, :]).astype(np.uint8)
    edge_mask[:-1, :] |= (lab[1:, :] != lab[:-1, :]).astype(np.uint8)

    # 4) Kantenmaske in einen weichen Übergangsbereich umwandeln
    #    -> Gaussian Blur auf die Maske, dann normalisieren nach [0,1]
    if edge_blur_ksize is not None and edge_blur_ksize > 1:
        if edge_blur_ksize % 2 == 0:
            edge_blur_ksize += 1
        edge_soft = cv2.GaussianBlur(
            edge_mask.astype(np.float32), 
            (edge_blur_ksize, edge_blur_ksize), 
            0
        )
        if edge_soft.max() > 0:
            edge_soft /= edge_soft.max()
    else:
        edge_soft = edge_mask.astype(np.float32)

    # 5) Glatte Version des Gradienten berechnen
    grad_blur = cv2.GaussianBlur(grad.astype(np.float32), (edge_blur_ksize, edge_blur_ksize), 0)

    # 6) Nur in Kanten-Nähe vom glatten Gradient überblenden
    #    edge_soft ~ 0 -> nur ursprünglicher Gradient
    #    edge_soft ~ 1 -> komplett glatter Gradient
    alpha = np.clip(edge_soft * edge_blur_strength, 0.0, 1.0)
    grad_smooth = (1.0 - alpha) * grad + alpha * grad_blur

    # 7) Blend-Faktor clampen
    blend = float(blend)
    if blend < 0.0:
        blend = 0.0
    if blend > 1.0:
        blend = 1.0

    # 8) Additiv mischen: Bild + (ggf. geglätteter) Gradient
    out = (1.0 - blend) * img + blend * grad_smooth

    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def generate_base_surface(
    canvas_size: Tuple[int, int] = CANVAS_SIZE,
    add_terraces: bool = True,
    terrace_strength: float = 0.7,
    horizontal_dimers: bool = False,   # NEU: globale Dimer-Ausrichtung
) -> np.ndarray:
    H, W = int(canvas_size[0]), int(canvas_size[1])
    """
    High-Level:
    - erzeugt Terrassen-Heightmap + Labels
    - rendert Dimerreihen pro Terrasse (0°/90° alternierend)
    - legt Helligkeitsgradienten entlang der Terrassenkanten über die *gesamte* Terrasse
    - komprimiert anschließend den Kontrast und verschmilzt Dimere leicht
    """
    H, W = int(canvas_size[0]), int(canvas_size[1])

    if add_terraces:
        z, labels = generate_terrace_heightmap(
            H,
            W,
            use_parallel_step=horizontal_dimers,  # hier schaltest du um
        )
    else:
        z = np.zeros((H, W), dtype=np.float32)
        labels = np.zeros((H, W), dtype=np.int32)
    # Dimeroberfläche pro Terrasse
    dimer_canvas = render_dimers_for_terraces(labels, BASE_TILE)

    # Lokale Amplitudenvariation der Dimere (1–3 px korreliert)
    dimer_canvas = apply_dimer_amplitude_noise(
        dimer_canvas,
        sigma=0.08,     # Stärke der Variationen
        blur_ksize=10,   # ~1–3 Pixel Korrelation
    )

    # Terrassen-Gradient auf gesamte Terrasse (Dimere + Zwischenräume + später Defekte)
    canvas = apply_terrace_edge_gradient(
        dimer_canvas,
        labels,
        blend=0.7,
        min_val=0.0,
        max_val=1.0,
        edge_blur_ksize=6,
        edge_blur_strength=2.5,
    )

    # >>> Kontrast glätten & Dimere verschmelzen lassen <<<
    # blur_choices = (0, 3)  # 3 = eher scharf, 7 = ziemlich weich
    # blur_ksize = random.choice(blur_choices)
    # blur_ksize = 3

    # canvas = soften_and_flatten_contrast(
    #     canvas,
    #     blur_ksize=blur_ksize,      # ggf. auf 5 erhöhen, wenn es noch zu scharf ist
    #     lift_black=0.00,   # höher -> Zwischenräume heller
    #     reduce_range=0.00, # höher -> Gesamt-Kontrast flacher
    # )

    return canvas
