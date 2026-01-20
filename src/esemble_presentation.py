import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import matplotlib.cm as cm
import matplotlib.patches as mpatches

from src.predictions.utils import load_png_gray  # (H,W,1) float32, [0,1]

# Ordner, in dem dieses Skript liegt: <projekt>/src
ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent

# === Pfade zu deinen beiden Modellen (ANPASSEN!) ============================
MODEL1_PATH = PROJECT_ROOT / "runs" / "ng_f60" / "model_final.keras"
MODEL2_PATH = PROJECT_ROOT / "runs" / "ngns_f80" / "model_final.keras"

RAW_TIF_DIR    = PROJECT_ROOT / "real_stm_raw_tifs"
REAL_IMAGE_DIR = PROJECT_ROOT / "real_stm_converted_png"

# Eigener Output-Ordner für Ensemble
OUTPUT_DIR     = PROJECT_ROOT / "runs" / "ensemble_ngns_f80_ng_f80" / "predictions_real_tripanel_presentation"
# ============================================================================


def tif_to_colormap_png(tif_path, cmap_name="inferno"):
    img = Image.open(tif_path)
    if img.mode != "L":
        img = img.convert("L")

    raw = np.array(img).astype(np.float32)  # 0..255
    raw_norm = raw / 255.0                  # 0..1

    cmap = cm.get_cmap(cmap_name)
    colored = cmap(raw_norm)[..., :3]
    colored = (colored * 255).astype(np.uint8)
    return Image.fromarray(colored)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Lade Modelle ...")
    model1 = tf.keras.models.load_model(str(MODEL1_PATH), compile=False)
    model2 = tf.keras.models.load_model(str(MODEL2_PATH), compile=False)
    print("Modelle geladen.")

    img_paths = sorted(p for p in REAL_IMAGE_DIR.glob("*.png"))
    n_total = len(img_paths)
    print(f"Gefundene Realbilder (PNG): {n_total}")

    if n_total == 0:
        print("Keine Realbilder gefunden. Abbruch.")
        return

    for idx, img_path in enumerate(img_paths, start=1):
        base_name = img_path.stem
        print(f"[{idx}/{n_total}] {base_name}")

        # Raw-TIF als Referenzbild (Panel 1)
        tif_path = RAW_TIF_DIR / f"{base_name}.tif"
        if tif_path.exists():
            orig_img = tif_to_colormap_png(tif_path, cmap_name="inferno")
            orig_arr = np.array(orig_img)
        else:
            orig_arr = None

        # Modell-Input
        image_tf = load_png_gray(img_path)        # (H,W,1), float32, [0,1]
        img_model = image_tf.numpy()[..., 0]      # für Anzeige Panel 3

        # Graustufenbild für Panel 2
        gray_arr = np.array(Image.open(img_path))  # (H,W) uint8

        # === ENSEMBLE PREDICTION (Modell 1 = Basis, Modell 2 = Korrektor) ===
        logits1 = model1.predict(
            tf.expand_dims(image_tf, axis=0), verbose=0
        )[0]  # (H,W,C)
        logits2 = model2.predict(
            tf.expand_dims(image_tf, axis=0), verbose=0
        )[0]  # (H,W,C)

        # Falls deine Modelle schon Softmax im letzten Layer haben,
        # sind logits1/logits2 bereits Wahrscheinlichkeiten.
        p1 = logits1
        p2 = logits2

        # Vorhersagen + Konfidenzen je Modell
        pred1 = np.argmax(p1, axis=-1).astype(np.uint8)   # (H,W)
        conf1 = np.max(p1, axis=-1).astype(np.float32)    # (H,W)

        pred2 = np.argmax(p2, axis=-1).astype(np.uint8)   # (H,W)
        conf2 = np.max(p2, axis=-1).astype(np.float32)    # (H,W)

        # Start: Modell 1 ist Basis
        final_pred = pred1.copy()

        # Klassenspezifische Infos für 2-Dimer
        class_idx_2dimer = 2  # 2-Dimer-Klasse (anpassen, falls andere ID)
        p1_2d = p1[..., class_idx_2dimer]  # (H,W), Prob für 2-Dimer laut Modell 1

        # --- Regel 1: Modell 2 darf zusätzliche Defekte hinzufügen ----------
        # Schwelle, ab wann Modell 2 "sicher genug" ist:
        T_ADD = 0.75        # Modell 2 Konfidenz
        T_SUPPORT_2D = 0.80 # Mindest-Prob für 2-Dimer in Modell 1

        base_add_mask = (pred1 == 0) & (pred2 != 0) & (conf2 > T_ADD)

        # a) Nicht-2-Dimer-Klassen wie bisher
        add_mask_non2 = base_add_mask & (pred2 != class_idx_2dimer)

        # b) 2-Dimer nur, wenn Modell 1 auch nennenswerte 2-Dimer-Prob hat
        add_mask_2 = (
            base_add_mask
            & (pred2 == class_idx_2dimer)
            & (p1_2d > T_SUPPORT_2D)
        )

        add_mask = add_mask_non2 | add_mask_2
        final_pred[add_mask] = pred2[add_mask]

        # --- (Optional) Regel 2: Modell 2 darf unsichere Defekte von Modell 1 löschen ---
        # Nur, wenn Modell 2 sehr sicher "Hintergrund" sagt und Modell 1 nicht sehr sicher ist.
        T_CLEAR = 0.65  # Modell 2 sehr sicher auf Hintergrund
        T_LOW   = 0.65  # Modell 1 eher unsicher

        clear_mask = (
            (pred1 != 0) &              # Modell 1 sieht Defekt
            (pred2 == 0) &              # Modell 2 sieht Hintergrund
            (conf2 > T_CLEAR) &         # Modell 2 sehr sicher
            (conf1 < T_LOW)             # Modell 1 eher unsicher
        )
        final_pred[clear_mask] = 0

        pred_labels = final_pred
        # =====================================================================

        # Tripanel
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Panel 1: Original (TIF-Colormap oder PNG)
        if orig_arr is not None:
            axes[0].imshow(orig_arr, origin="lower")
            axes[0].set_title("Original TIF")
        else:
            axes[0].imshow(gray_arr, cmap="gray", origin="lower")
            axes[0].set_title("Original (PNG-Ersatz)")
        axes[0].axis("off")

        # Panel 2: Graustufen 600×600
        axes[1].imshow(gray_arr, cmap="gray", origin="lower")
        axes[1].set_title("Graustufen 600×600")
        axes[1].axis("off")

        # Panel 3: Ensemble-Overlay
        axes[2].imshow(img_model, cmap="gray", origin="lower")

        H, W = pred_labels.shape
        overlay = np.zeros((H, W, 4), dtype=np.float32)

        color_map = {
            0: (0.0, 0.5, 1.0, 0.4),  # Dimerreihe / Hintergrund
            1: (1.0, 0.0, 0.0, 0.7),  # 1-Dimer Signatur
            2: (0.0, 1.0, 0.0, 0.7),  # 2-Dimer Signatur
            3: (1.0, 1.0, 0.0, 0.7),  # Double-Dimer Signatur
        }

        for cls_id, (r, g, b, a) in color_map.items():
            mask = (pred_labels == cls_id)
            overlay[mask] = (r, g, b, a)

        axes[2].imshow(overlay, origin="lower")
        axes[2].set_title("Vorhersage-Overlay (Ensemble, gewichtet)")
        axes[2].axis("off")

        handles = [
            mpatches.Patch(color=(0.0, 0.5, 1.0, 0.4), label="Hintergrund"),
            mpatches.Patch(color=(1.0, 0.0, 0.0, 0.7),   label="1-Dimer"),
            mpatches.Patch(color=(0.0, 1.0, 0.0, 0.7),   label="2-Dimer"),
            mpatches.Patch(color=(1.0, 1.0, 0.0, 0.7),   label="Double-Dimer"),
        ]

        axes[2].legend(handles=handles, loc="upper right", fontsize=8)

        fig.suptitle(f"{base_name} (Ensemble, gewichtet)")
        fig.tight_layout()

        out_path = OUTPUT_DIR / f"{base_name}_tripanel_ensemble.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    print("Fertig. Ensemble-Tripanel-Visualisierungen liegen in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
