from __future__ import annotations
import numpy as np
import cv2
import random
import math
from PIL import Image

from .dimer_physics import build_physical_dimer_tile
from .config import BASE_TILE, TILE_HEIGHT, TILE_WIDTH, BASE_TILE_ROW_MASK
from .tiles import make_repeated_tile
from .terraces import generate_terrace_layout, DimerOrientation
from .postprocessing import (
    blur_terrace_edges, apply_surface_tilt_and_oscillation, white_artifacts,
    distort_xy, distort_zones, apply_dimer_amplitude_noise,
    build_xy_flow, apply_flow_bilinear_u8, apply_flow_nearest_u8
)

from .overlay_engine import apply_random_signatures_oriented


def _make_oriented_canvas(
    canvas_size,
    orientation: str,
    tile_H: np.ndarray,
    tile_V: np.ndarray,
) -> np.ndarray:
    if orientation == DimerOrientation.H:
        base_tile = tile_H
    elif orientation == DimerOrientation.V:
        base_tile = cv2.rotate(tile_V, cv2.ROTATE_90_CLOCKWISE)
    else:
        raise ValueError(f"Unbekannte Orientierung: {orientation}")

    return make_repeated_tile(canvas_size, base_tile=base_tile)


def build_terraced_dimer_canvas(
    canvas_size=(600, 600),
    min_steps: int = 1,
    max_steps: int = 5,
    same_dimer_for_all: bool = False,
    add_signatures: bool = False,
    n_defects_range=(4, 16),
    min_distance_dimers: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Erzeugt ein Terrassenbild mit Dimeren und optional bereits
    eingebetteten Signaturen.

    Rückgabe:
        image:            fertiges Dimerbild (inkl. Gradient & Terrassen-Blur)
        terrace_pixels:   Terrassen-IDs pro Pixel (int32)
        terrace_orient:   Dimer-Orientierung pro Pixel ("H"/"V")
        mask:             Signaturmaske (0=none, 1=1Dimer, 2=2Dimer, 3=Double)
    """
    H, W = canvas_size

    # 1) Terrassenarchitektur
    edges, orientations, n_rows_tiles, n_cols_tiles = generate_terrace_layout(
        canvas_size=canvas_size,
        min_steps=min_steps,
        max_steps=max_steps,
        min_terrace_gap_tiles=6,
    )
    num_steps = edges.shape[0]
    n_terraces = num_steps + 1

    # 2) Terrassen-ID pro Tile
    terrace_tiles = np.zeros((n_rows_tiles, n_cols_tiles), dtype=np.int32)
    for x in range(n_cols_tiles):
        col_edges = np.sort(edges[:, x]) if num_steps > 0 else np.array([], dtype=np.int32)
        for row in range(n_rows_tiles):
            terrace_tiles[row, x] = int(np.sum(row >= col_edges))

    # 3) Tile-Terrassenkarte → Pixel
    terrace_pixels = np.repeat(
        np.repeat(terrace_tiles, TILE_HEIGHT, axis=0),
        TILE_WIDTH,
        axis=1,
    )
    terrace_pixels = terrace_pixels[:H, :W]

    # 4) Orientierung pro Terrasse auf Pixel-Ebene abbilden
    orientation_map = np.array(orientations, dtype=object)
    terrace_orient_pixels = orientation_map[terrace_pixels]  

    # 5) Dimer-Canvases aufbauen
    final_dimer = np.zeros((H, W), dtype=np.uint8)
    row_mask    = np.zeros((H, W), dtype=np.uint8) 

    if same_dimer_for_all:
        tile_base = build_physical_dimer_tile()
        canvas_H = _make_oriented_canvas(canvas_size, DimerOrientation.H, tile_base, tile_base)
        canvas_V = _make_oriented_canvas(canvas_size, DimerOrientation.V, tile_base, tile_base)

        # Dimerreihen-Masken auf Basis der vorgefertigten 9x18-Maske
        row_H = _make_oriented_canvas(canvas_size, DimerOrientation.H, BASE_TILE_ROW_MASK, BASE_TILE_ROW_MASK)
        row_V = _make_oriented_canvas(canvas_size, DimerOrientation.V, BASE_TILE_ROW_MASK, BASE_TILE_ROW_MASK)

        mask_H = terrace_orient_pixels == DimerOrientation.H
        mask_V = terrace_orient_pixels == DimerOrientation.V

        final_dimer[mask_H] = canvas_H[mask_H]
        final_dimer[mask_V] = canvas_V[mask_V]

        # class_id = 1 für alle Dimerreihen-Pixel
        row_mask[mask_H] = np.where(row_H[mask_H] == 1, 1, row_mask[mask_H])
        row_mask[mask_V] = np.where(row_V[mask_V] == 1, 1, row_mask[mask_V])

    else:
        for t_id, ori in enumerate(orientations):
            tile_base = build_physical_dimer_tile()

            canvas_t = _make_oriented_canvas(
                canvas_size,
                ori,
                tile_base,
                tile_base,
            )

            row_t = _make_oriented_canvas(
                canvas_size,
                ori,
                BASE_TILE_ROW_MASK,
                BASE_TILE_ROW_MASK,
            )

            mask_t = terrace_pixels == t_id
            if mask_t.any():
                final_dimer[mask_t] = canvas_t[mask_t]
                row_mask[mask_t] = np.where(row_t[mask_t] == 1, 1, row_mask[mask_t])

    # -------------------------------------------------------
    # 5b) Signaturen direkt auf diese Dimerstruktur legen
    # -------------------------------------------------------
    if add_signatures:
        sig_mask = np.zeros_like(final_dimer, dtype=np.uint8)
        final_dimer, sig_mask = apply_random_signatures_oriented(
            final_dimer,
            sig_mask,
            terrace_ids=terrace_pixels,
            terrace_orient=terrace_orient_pixels,
            n_defects_range=n_defects_range,
            min_distance_dimers=min_distance_dimers,
        )
    else:
        sig_mask = np.zeros_like(final_dimer, dtype=np.uint8)

    # -------------------------------------------------------
    # 6) Gradient pro Terrasse: oben hell, unten dunkel
    # -------------------------------------------------------
    grad_tiles = np.zeros((n_rows_tiles, n_cols_tiles), dtype=np.float32)

    if num_steps == 0:
        col_grad = np.linspace(1.0, 0.0, n_rows_tiles, dtype=np.float32)
        grad_tiles = np.repeat(col_grad[:, None], n_cols_tiles, axis=1)
    else:
        # Gradient pro Terrassen-Fläche
        for t in range(n_terraces):
            mask_t = terrace_tiles == t  
            if not mask_t.any():
                continue

            rows_t = np.where(mask_t.any(axis=1))[0]
            if rows_t.size == 0:
                continue

            top = int(rows_t.min())
            bottom = int(rows_t.max())
            height = bottom - top + 1
            if height <= 0:
                continue

            if height == 1:
                grad_tiles[top, mask_t[top]] = 1.0
            else:
                vals = np.linspace(1.0, 0.0, height, dtype=np.float32)
                for i, row in enumerate(range(top, bottom + 1)):
                    col_mask = mask_t[row]
                    if col_mask.any():
                        grad_tiles[row, col_mask] = vals[i]

    # Tiles → Pixel
    grad_pixels = np.repeat(
        np.repeat(grad_tiles, TILE_HEIGHT, axis=0),
        TILE_WIDTH,
        axis=1,
    )
    grad_pixels = grad_pixels[:H, :W]  

    # 7) Range für Helligkeit festlegen
    grad_low = random.uniform(0.0, 60.0)
    grad_high = random.uniform(195.0, 255.0)
    if grad_high <= grad_low:
        grad_high = grad_low + 1.0
    gradient_img = grad_low + (grad_high - grad_low) * grad_pixels

    grad_low = random.uniform(0.0, 60.0)
    grad_high = random.uniform(195.0, 255.0)
    if grad_high <= grad_low:
        grad_high = grad_low + 1.0
    gradient_img = grad_low + (grad_high - grad_low) * grad_pixels

    # 8) Gradient und Dimerstruktur mischen
    mix_alpha = random.uniform(0.4, 0.9)
    fd = final_dimer.astype(np.float32)
    out = (1.0 - mix_alpha) * fd + mix_alpha * gradient_img
    out = np.clip(out, 0, 255).astype(np.uint8)

    # Terrassenkanten weichzeichnen
    out = blur_terrace_edges(
        out,
        terrace_pixels=terrace_pixels,
        edge_band_px=4,
        blur_ksize=20,
    )

    return out, terrace_pixels, terrace_orient_pixels, sig_mask, row_mask


def build_rotated_terraced_dimer_canvas(
    target_size: tuple[int, int] = (600, 600),
    scale_factor: float = 2.5,
    angle_deg: float = 45.0,
    min_steps: int = 1,
    max_steps: int = 5,
    perturbations: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Erzeugt ein fertiges STM-Bild + Signatur-Maske:

    1) größeres Terrassenbild + Terrassenlabels + Signaturmaske erzeugen
    2) gesamtes Bild + Masken + Terrassenlabels drehen
    3) Lokaler Drift distort_xy PRO TERRASSE (Bild+Maske) – vor Crop
    4) globale Zonenverzerrung (Bild+Maske)
    5) zentrierter Crop
    6) Tilt / Wobble
    7) Amplitudenrauschen
    8) White-artifacts (nur Bild)
    """

    H_t, W_t = target_size
    if perturbations is None:
        perturbations = {"tilt", "xy", "zones", "amp_noise", "white_artifacts"}

    # ----------------------------------------------------------------------
    # 1) größeres Canvas erzeugen (Bild + Terrassen-Labels + Signaturen)
    # ----------------------------------------------------------------------
    H_big = int(math.ceil(H_t * scale_factor))
    W_big = int(math.ceil(W_t * scale_factor))

    base, terrace_pixels, terrace_orient, mask, row_mask = build_terraced_dimer_canvas(
        canvas_size=(H_big, W_big),
        min_steps=min_steps,
        max_steps=max_steps,
        add_signatures=True,
        n_defects_range=(1, 25),
        min_distance_dimers=2,
    )
    base = base.astype(np.uint8)
    terrace_pixels = terrace_pixels.astype(np.int32)
    mask = mask.astype(np.uint8)
    row_mask = row_mask.astype(np.uint8)

    # ----------------------------------------------------------------------
    # 2) gesamte Oberfläche + Maske + Terrassenlabels drehen
    # ----------------------------------------------------------------------
    center = (W_big / 2.0, H_big / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    rotated = cv2.warpAffine(
        base.astype(np.float32),
        M,
        (W_big, H_big),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    ).astype(np.uint8)

    rotated_mask = cv2.warpAffine(
        mask,
        M,
        (W_big, H_big),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_REFLECT101,
        borderValue=0,
    )

    rotated_row = cv2.warpAffine(
        row_mask,
        M,
        (W_big, H_big),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_REFLECT101,
        borderValue=0,
    )

    rotated_terraces = cv2.warpAffine(
        terrace_pixels.astype(np.float32),
        M,
        (W_big, H_big),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_REFLECT101,
        borderValue=-1.0,
    ).astype(np.int32)

    # ----------------------------------------------------------------------
    # 3) Lokaler Drift distort_xy PRO TERRASSE (Bild + Maske) – vor Crop
    # ----------------------------------------------------------------------
    if "xy" in perturbations:
        img_base  = rotated
        mask_base = rotated_mask
        row_base  = rotated_row

        img_out  = img_base.copy()
        mask_out = mask_base.copy()
        row_out  = row_base.copy()

        unique_terraces = np.unique(rotated_terraces)
        for t_id in unique_terraces:
            if t_id < 0:
                continue  # -1 = außerhalb

            terr_mask_full = (rotated_terraces == t_id)
            if not terr_mask_full.any():
                continue

            # Bounding Box nur zur Beschleunigung
            ys, xs = np.where(terr_mask_full)
            y_min, y_max = ys.min(), ys.max()
            x_min, x_max = xs.min(), xs.max()

            sub_img_base  = img_base[y_min:y_max + 1, x_min:x_max + 1]
            sub_mask_base = mask_base[y_min:y_max + 1, x_min:x_max + 1]
            sub_row_base  = row_base[y_min:y_max + 1, x_min:x_max + 1]

            # Terrassenmaske in diesem Ausschnitt (0/255)
            sub_terr_mask = terr_mask_full[y_min:y_max + 1, x_min:x_max + 1]
            sub_terr_mask_u8 = (sub_terr_mask.astype(np.uint8) * 255)

            pil_img   = Image.fromarray(sub_img_base,      mode="L")
            pil_sig   = Image.fromarray(sub_mask_base,     mode="L")
            pil_row   = Image.fromarray(sub_row_base,      mode="L")
            pil_tmask = Image.fromarray(sub_terr_mask_u8,  mode="L")

            # --- Zufällige Drift-Parameter pro Terrasse ---
            # max_shift_px = 6.0
            # strength     = 1.5
            max_shift_px = random.uniform(3.0, 8.0)
            strength     = random.uniform(1.0, 1.5)

            Hsub, Wsub = sub_img_base.shape[:2]

            # Seed pro Terrasse 
            seed = random.randint(0, 2**31 - 1)
            rng = np.random.default_rng(seed)

            map_x, map_y = build_xy_flow(
                Hsub, Wsub,
                max_shift_px=max_shift_px,
                strength=strength,
                rng=rng,
            )

            sub_img2   = apply_flow_bilinear_u8(sub_img_base.astype(np.uint8), map_x, map_y)
            sub_sig2   = apply_flow_nearest_u8(sub_mask_base.astype(np.uint8), map_x, map_y, border_value=0)
            sub_row2   = apply_flow_nearest_u8(sub_row_base.astype(np.uint8),  map_x, map_y, border_value=0)
            sub_tmask2 = apply_flow_nearest_u8(sub_terr_mask_u8.astype(np.uint8), map_x, map_y, border_value=0)

            sel = sub_tmask2 > 0

            img_patch  = img_out[y_min:y_max + 1, x_min:x_max + 1]
            mask_patch = mask_out[y_min:y_max + 1, x_min:x_max + 1]
            row_patch  = row_out[y_min:y_max + 1, x_min:x_max + 1]

            img_patch[sel]  = sub_img2[sel]
            mask_patch[sel] = sub_sig2[sel]
            row_patch[sel]  = sub_row2[sel]

            img_out[y_min:y_max + 1, x_min:x_max + 1]  = img_patch
            mask_out[y_min:y_max + 1, x_min:x_max + 1] = mask_patch
            row_out[y_min:y_max + 1, x_min:x_max + 1]  = row_patch

        rotated      = img_out
        rotated_mask = mask_out
        rotated_row  = row_out


    # ----------------------------------------------------------------------
    # 4) globale Zonenverzerrung (Bild + Maske)
    # ----------------------------------------------------------------------
    if "zones" in perturbations:
        state = random.getstate()

        random.setstate(state)
        rotated, rotated_mask = distort_zones(rotated, rotated_mask)

        random.setstate(state)
        rotated_row = distort_zones(rotated_row)

    # ----------------------------------------------------------------------
    # 5) zentrierter Crop
    # ----------------------------------------------------------------------
    y0 = (H_big - H_t) // 2
    x0 = (W_big - W_t) // 2
    cropped = rotated[y0:y0 + H_t, x0:x0 + W_t]
    cropped_mask = rotated_mask[y0:y0 + H_t, x0:x0 + W_t]
    cropped_row  = rotated_row[y0:y0 + H_t, x0:x0 + W_t]

    # ----------------------------------------------------------------------
    # 6) Tilt / Wobble (nur Bild)
    # ----------------------------------------------------------------------
    if "tilt" in perturbations:
        cropped = apply_surface_tilt_and_oscillation(
            cropped,
            slanted=True,
            oscillation=True,
        )

    # ----------------------------------------------------------------------
    # 7) Amplitudenrauschen auf Dimeren (Bild)
    # ----------------------------------------------------------------------
    if "amp_noise" in perturbations:
        cropped = apply_dimer_amplitude_noise(cropped, sigma=0.06, blur_ksize=3)

    # ----------------------------------------------------------------------
    # 8) White artifacts (nur Bild)
    # ----------------------------------------------------------------------
    if "white_artifacts" in perturbations:
        pil = Image.fromarray(cropped, mode="L")
        pil = white_artifacts(pil)
        cropped = np.array(pil, dtype=np.uint8)

    return cropped, cropped_mask, cropped_row


