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
MODEL1_PATH = PROJECT_ROOT / "runs" / "sixth_training_f80" / "model_final.keras"
MODEL2_PATH = PROJECT_ROOT / "runs" / "ng_f60" / "model_final.keras"

RAW_TIF_DIR    = PROJECT_ROOT / "real_stm_raw_tifs"
REAL_IMAGE_DIR = PROJECT_ROOT / "real_stm_converted_png"

# Eigener Output-Ordner für Ensemble
OUTPUT_DIR     = PROJECT_ROOT / "runs" / "ensemble_sixth_training_f80_ng_f60" / "predictions_real_tripanel_presentation"
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

                # === ENSEMBLE PREDICTION (M1 Basis, M2 kontrolliert 2-Dimer) ===
        logits1 = model1.predict(
            tf.expand_dims(image_tf, axis=0), verbose=0
        )[0]  # (H,W,C)
        logits2 = model2.predict(
            tf.expand_dims(image_tf, axis=0), verbose=0
        )[0]  # (H,W,C)

        # Falls Modelle Softmax im letzten Layer haben, sind das schon Wahrscheinlichkeiten
        p1 = logits1
        p2 = logits2

        pred1 = np.argmax(p1, axis=-1).astype(np.uint8)   # (H,W)
        conf1 = np.max(p1, axis=-1).astype(np.float32)    # (H,W)

        pred2 = np.argmax(p2, axis=-1).astype(np.uint8)   # (H,W)
        conf2 = np.max(p2, axis=-1).astype(np.float32)    # (H,W)

        # Basis: Modell 1
        final_pred = pred1.copy()

        # --- Spezielle Behandlung für 2-Dimer (Klasse 2) -----------------
        class_idx_2dimer = 2

        # Hintergund- und 2-Dimer-Probabilitäten beider Modelle
        p1_bg  = p1[..., 0]
        p2_bg  = p2[..., 0]
        p1_2d  = p1[..., class_idx_2dimer]
        p2_2d  = p2[..., class_idx_2dimer]

        # Schwellen
        T_BG_VETO = 0.90   # Ab hier sagt das andere Modell: "ziemlich sicher Background"
        T_2D_MIN  = 0.50   # Mindest-Prob für 2-Dimer im jeweiligen Modell

        # (1) 2-Dimer-Hallus von Modell 1 entfernen, wenn Modell 2 klar BG sieht
        clear_2_from_m1 = (
            (final_pred == class_idx_2dimer) &
            (p2_bg >= T_BG_VETO)
        )
        final_pred[clear_2_from_m1] = 0  # auf Hintergrund zurücksetzen

        # (2) 2-Dimer aus Modell 1 übernehmen, wenn:
        #     - Modell 1 2-Dimer will,
        #     - Modell 1 halbwegs sicher,
        #     - Modell 2 nicht mit hoher Sicherheit BG sagt.
        cand_2_from_m1 = (
            (pred1 == class_idx_2dimer) &
            (p1_2d >= T_2D_MIN) &
            (p2_bg < T_BG_VETO)
        )

        # (3) 2-Dimer aus Modell 2 übernehmen, wenn:
        #     - Modell 2 2-Dimer will,
        #     - Modell 2 halbwegs sicher,
        #     - Modell 1 nicht mit hoher Sicherheit BG sagt.
        cand_2_from_m2 = (
            (pred2 == class_idx_2dimer) &
            (p2_2d >= T_2D_MIN) &
            (p1_bg < T_BG_VETO)
        )

        # Zuerst alles, was als 2-Dimer weg soll, auf 0 (oben passiert),
        # dann alle gültigen Kandidaten auf 2 setzen (egal von welchem Modell)
        add_2_mask = cand_2_from_m1 | cand_2_from_m2
        final_pred[add_2_mask] = class_idx_2dimer

        pred_labels = final_pred
        # =====================================================================

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

        axes[2].legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),  # rechts außerhalb der Achse
            borderaxespad=0.0,
            fontsize=8,
        )

        fig.suptitle(f"{base_name} (Ensemble, gewichtet)")
        fig.tight_layout()

        out_path = OUTPUT_DIR / f"{base_name}_tripanel_ensemble.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


    print("Fertig. Ensemble-Tripanel-Visualisierungen liegen in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
