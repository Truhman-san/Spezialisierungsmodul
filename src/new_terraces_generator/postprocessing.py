import numpy as np
import random
from PIL import Image, ImageFilter
import math
import cv2


def blur_terrace_edges(
    image: np.ndarray,
    terrace_pixels: np.ndarray,
    edge_band_px: int = 9,
    blur_ksize: int = 9,
) -> np.ndarray:
    """
    Weicht die Terrassenkanten etwas auf, ohne das ganze Bild zu verwischen.

    - edge_band_px: Breite der Übergangszone um die Kante (in Pixeln)
    - blur_ksize: Kernelgröße für Gaussian-Blur (muss ungerade sein)

    Strategie:
    1) Kanten auf Basis von terrace_pixels finden
    2) Distanz zur Kante berechnen
    3) Weights 0..1 im Band um die Kante
    4) Originalbild mit geblurrter Version entlang der Kanten mischen
    """
    import cv2

    img = image.astype(np.float32)
    H, W = img.shape[:2]

    # --- 1) Kantenmaske aus den Terrassenlabels ---
    labels = terrace_pixels.astype(np.int32)

    # Unterschiede zu Nachbarn
    edge = np.zeros_like(labels, dtype=bool)

    edge[1:, :] |= labels[1:, :] != labels[:-1, :]
    edge[:-1, :] |= labels[:-1, :] != labels[1:, :]
    edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    edge[:, :-1] |= labels[:, :-1] != labels[:, 1:]

    edge_uint8 = np.zeros((H, W), dtype=np.uint8)
    edge_uint8[edge] = 255  # Kanten = 255

    # --- 2) Distanz zur Kante ---
    # Für distanceTransform sollen Nicht-Kanten 255, Kanten 0 sein
    inv_edge = np.where(edge_uint8 > 0, 0, 255).astype(np.uint8)
    dist = cv2.distanceTransform(inv_edge, distanceType=cv2.DIST_L2, maskSize=3)

    # --- 3) Weights 0..1 im Band um die Kante ---
    # dist = 0  -> direkt an der Kante
    # dist >= edge_band_px -> weit weg
    weights = (edge_band_px - dist) / float(edge_band_px)
    weights = np.clip(weights, 0.0, 1.0)

    # --- 4) Geblurrtes Bild & Mischung ---
    if blur_ksize % 2 == 0:
        blur_ksize += 1

    blurred = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)

    # falls Bild 2D ist: Weights 2D, sonst anpassen
    if img.ndim == 2:
        w = weights
    else:
        w = weights[..., None]

    out = img * (1.0 - w) + blurred * w
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def apply_surface_tilt_and_oscillation(
    image: np.ndarray,
    slanted: bool = True,
    oscillation: bool = False,
    C: float | None = None,
    B: float | None = None,
    H: float | None = None,
    A: float | None = None,
    L: float | None = None,
) -> np.ndarray:
    """
    Globale Schräge (Tilt) und/oder langsame Oszillation auf das STM-Bild legen.

    - image: uint8 [0..255], 2D
    - Rückgabe: uint8 [0..255]

    Parameter (typische Ranges, wenn None):
      * Tilt:
        - C: Richtung der Schräge (0..2π)
        - B: Steigung ~ [-1e-4, 1e-4]
        - H: globaler Offset ~ [-0.03, 0.03]
      * Oszillation:
        - A: Amplitude ~ [0.01, 0.03] (entspricht ~3–8 Graustufen)
        - L: Wellenlänge ~ [0.5*min(H,W), 2*min(H,W)]
    """

    # In [0,1] normalisieren
    image_norm = image.astype(np.float64) / 255.0
    height, width = image_norm.shape
    x = np.arange(width, dtype=np.float64)
    y = np.arange(height, dtype=np.float64)
    X, Y = np.meshgrid(x, y)

    # -----------------------
    # Oszillation (grobe Welle)
    # -----------------------
    if oscillation:
        if A is None:
            # 1–3 % Intensität -> 2.5–7.5 Graustufen
            A = random.uniform(0.01, 0.1)
        if L is None:
            # Sehr lange Wellen: 0.5–2 Bildbreiten
            min_dim = float(min(height, width))
            L = random.uniform(0.5 * min_dim, 2.0 * min_dim)
    else:
        A = 0.0
        L = 1.0  # dummy

    # -----------------------
    # Schräge Ebene (Tilt)
    # -----------------------
    if slanted:
        if C is None:
            # beliebige Richtung der Schräge
            C = random.uniform(0.0, 2.0 * np.pi)
        if B is None:
            # kleine Steigung: über 600 px ~ 10–30 Graustufen
            B = random.uniform(-7.0e-4, 7.0e-4)
        if H is None:
            # globaler Offset, sehr klein
            H = random.uniform(-0.03, 0.03)
    else:
        C = 0.0
        B = 0.0
        H = 0.0

    # Schräge und Oszillation
    if slanted:
        slanted_term = B * (np.cos(C) * X - np.sin(C) * Y) + H
    else:
        slanted_term = 0.0

    if oscillation:
        oscillation_term = A * np.cos(2.0 * np.pi * (X + Y) / L)
    else:
        oscillation_term = 0.0

    total_augmentation = slanted_term + oscillation_term

    augmented = image_norm + total_augmentation
    augmented = np.clip(augmented, 0.0, 1.0)
    return (augmented * 255.0).astype(np.uint8)


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


def white_artifacts(
    img_pil: Image.Image,
    max_lines: int = 12,       # maximale Anzahl Artefakt-Streifen pro Bild (deutlich weniger)
    min_width: int = 15,       # minimale Stripe-Länge
    max_width: int = 120,      # maximale Stripe-Länge
    typical_min: int = 30,     # typischer Bereich für die meisten Stripes
    typical_max: int = 80,
) -> Image.Image:
    """
    Fügt weiche, weiße STM-ähnliche Artefakte hinzu.

    - Weniger, aber realistischere Streifen.
    - Jeder Stripe hat einen hellen "Kern" am Rand und läuft entlang seiner Länge weich aus.
    - Keine harten 255er-Balken, sondern additive Aufhellung um typ. +20..+70 Graustufen,
      weich geblurrt.
    """

    img_gray = img_pil.convert("L")
    w, h = img_gray.size

    # Basis-Bild als float
    base = np.array(img_gray, dtype=np.float32)

    # Artefakt-Layer (Delta-Helligkeit, in Graustufen)
    artifact = np.zeros((h, w), dtype=np.float32)

    # Wie viele Stripes wirklich zeichnen?
    if max_lines <= 0:
        return img_pil

    n_stripes = random.randint(0, max_lines)  # 0..max_lines (also oft auch wenige)

    for _ in range(n_stripes):
        # Höhe des Stripes (1..stripe_height)
        stripe_h = 1

        # Breite: meist im "typischen" Bereich, gelegentlich breiter/schmaler
        if random.random() < 0.8:
            stripe_w = random.randint(typical_min, typical_max)
        else:
            stripe_w = random.randint(min_width, max_width)

        if stripe_w <= 0 or stripe_h <= 0:
            continue

        if stripe_w > w:
            stripe_w = w

        # zufällige Position
        y = random.randint(0, max(0, h - stripe_h))
        x0 = random.randint(0, max(0, w - stripe_w))
        x1 = x0 + stripe_w

        # Maximale zusätzliche Helligkeit am hellsten Punkt des Stripes (Delta)
        peak_delta = random.uniform(230.0, 255.0)

        # Helles Ende -> dunkles Ende als 1D-Gradient
        # Richtung: entweder links->rechts oder rechts->links
        grad = np.linspace(peak_delta, 0.0, stripe_w, dtype=np.float32)
        if random.random() < 0.5:
            grad = grad[::-1]

        # Anfang des Stripes richtig hell machen
        if stripe_w >= 4:
            grad[:3] = 255.0

        # Stripe in das Artifact-Array eintragen (gleicher Verlauf für jede Zeile des Stripes)
        patch = artifact[y : y + stripe_h, x0:x1]
        # Broadcasting: (stripe_h, stripe_w) + (stripe_w,) -> (stripe_h, stripe_w)
        patch = np.maximum(patch, grad[None, :])
        artifact[y : y + stripe_h, x0:x1] = patch

    # Wenn gar keine Artefakte erzeugt wurden: direkt zurück
    if artifact.max() <= 0:
        return img_pil

    # In PIL-Image umwandeln und weichzeichnen, damit die Ränder auslaufen
    artifact_img = Image.fromarray(
        np.clip(artifact, 0, 255).astype(np.uint8), mode="L"
    )

    # Weichzeichnen: größerer Radius -> weicheres Auslaufen
    artifact_img = artifact_img.filter(ImageFilter.GaussianBlur(radius=1.3))

    artifact_blurred = np.array(artifact_img, dtype=np.float32)

    # Artefakt-Layer additiv zum Bild hinzufügen
    out = base + artifact_blurred

    # Clipping auf gültigen Bereich
    out = np.clip(out, 0, 255).astype(np.uint8)

    return Image.fromarray(out, mode="L")


def build_xy_flow(
    H: int,
    W: int,
    max_shift_px: float = 8.0,
    freq_min: float = 1/600.0,
    freq_max: float = 1/150.0,
    strength: float = 1.5,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Baut ein deterministisches Flowfeld map_x/map_y (float32) für cv2.remap.
    """
    if rng is None:
        rng = np.random.default_rng()

    fx = rng.uniform(freq_min, freq_max)
    fy = rng.uniform(freq_min, freq_max)
    phase_x = rng.uniform(0.0, 2.0 * math.pi)
    phase_y = rng.uniform(0.0, 2.0 * math.pi)

    xs = np.linspace(0, 1, W, dtype=np.float32)
    ys = np.linspace(0, 1, H, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)

    shift_x = (
        max_shift_px * strength
        * np.sin(2 * math.pi * fx * X + phase_x)
        * np.cos(2 * math.pi * fy * Y + phase_y)
    )

    shift_y = (
        max_shift_px * strength
        * np.cos(2 * math.pi * fx * X + phase_x)
        * np.sin(2 * math.pi * fy * Y + phase_y)
    )

    map_x = (X * (W - 1) + shift_x).astype(np.float32)
    map_y = (Y * (H - 1) + shift_y).astype(np.float32)

    map_x = np.clip(map_x, 0, W - 1)
    map_y = np.clip(map_y, 0, H - 1)

    return map_x, map_y


def apply_flow_bilinear_u8(img_u8: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    """
    Bilinear remap für Graustufenbilder.
    """
    out = cv2.remap(
        img_u8,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    return out.astype(np.uint8)


def apply_flow_nearest_u8(mask_u8: np.ndarray, map_x: np.ndarray, map_y: np.ndarray, border_value: int = 0) -> np.ndarray:
    """
    Nearest remap für Labelmasken (keine Mischklassen).
    """
    out = cv2.remap(
        mask_u8,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    return out.astype(np.uint8)


def distort_xy(
    img_pil: Image.Image,
    msk_pil: Image.Image | None = None,
    max_shift_px: float = 8.0,
    freq_min: float = 1/600.0,
    freq_max: float = 1/150.0,
    strength: float = 1.5,
    rng: np.random.Generator | None = None,
):
    """
    Backward-kompatibel: baut intern Flow und remappt Bild (bilinear) + optional Maske (nearest).
    """
    img = np.array(img_pil.convert("L"), dtype=np.uint8)
    H, W = img.shape

    map_x, map_y = build_xy_flow(
        H, W,
        max_shift_px=max_shift_px,
        freq_min=freq_min,
        freq_max=freq_max,
        strength=strength,
        rng=rng,
    )

    img_out = apply_flow_bilinear_u8(img, map_x, map_y)
    img_out_pil = Image.fromarray(img_out, mode="L")

    if msk_pil is None:
        return img_out_pil

    mask = np.array(msk_pil.convert("L"), dtype=np.uint8)
    if mask.shape != img.shape:
        raise ValueError("Maske und Bild müssen gleiche Größe haben.")

    mask_out = apply_flow_nearest_u8(mask, map_x, map_y, border_value=0)
    mask_out_pil = Image.fromarray(mask_out, mode="L")

    return img_out_pil, mask_out_pil


def distort_zones(
    img: np.ndarray,
    mask: np.ndarray | None = None,
    scale_min: float = 0.85,
    scale_max: float = 1.2,
):
    """
    Globale anisotrope Skalierung um das Bildzentrum.
    - img: Graustufenbild (H,W)
    - mask: optionale Labelmaske (H,W), uint8

    Rückgabe:
        wenn mask is None: img_out
        sonst: (img_out, mask_out)
    """
    h, w = img.shape[:2]

    direction = random.choice(["x", "y"])
    scale = random.uniform(scale_min, scale_max)

    if direction == "x":
        sx, sy = scale, 1.0
    else:
        sx, sy = 1.0, scale

    cx, cy = w / 2.0, h / 2.0

    M = np.array(
        [
            [sx, 0.0, (1.0 - sx) * cx],
            [0.0, sy, (1.0 - sy) * cy],
        ],
        dtype=np.float32,
    )

    img_out = cv2.warpAffine(
        img.astype(np.float32),
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    img_out = np.clip(img_out, 0, 255).astype(np.uint8)

    if mask is None:
        return img_out

    mask_out = cv2.warpAffine(
        mask.astype(np.uint8),
        M,
        (w, h),
        flags=cv2.INTER_NEAREST,         
        borderMode=cv2.BORDER_REFLECT_101,
    )

    return img_out, mask_out

