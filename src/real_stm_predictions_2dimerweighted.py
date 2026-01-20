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

MODEL_PATH     = PROJECT_ROOT / "runs" / "seventh_training_f80" / "model_final.keras"
RAW_TIF_DIR    = PROJECT_ROOT / "real_stm_raw_tifs"
REAL_IMAGE_DIR = PROJECT_ROOT / "real_stm_converted_png"
OUTPUT_DIR     = PROJECT_ROOT / "runs" / "seventh_training_f80" / "predictions_real_tripanel_2dimerweighted"

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

    print("Lade Modell ...")
    model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    print("Modell geladen.")

    img_paths = sorted(p for p in REAL_IMAGE_DIR.glob("*.png"))
    n_total = len(img_paths)
    print(f"Gefundene Realbilder (PNG): {n_total}")

    if n_total == 0:
        print("Keine Realbilder gefunden. Abbruch.")
        return

    for idx, img_path in enumerate(img_paths, start=1):
        base_name = img_path.stem
        print(f"[{idx}/{n_total}] {base_name}")

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

                # Prediction
        pred_logits = model.predict(
            tf.expand_dims(image_tf, axis=0), verbose=0
        )[0].astype(np.float32)  # (H,W,C)

        # Falls das Modell mit Logits endet, Softmax anwenden.
        # Wenn es schon Softmax ist, ist das trotzdem ok (monoton).
        probs = tf.nn.softmax(pred_logits, axis=-1).numpy()  # (H,W,4)

        p0 = probs[..., 0]  # Hintergrund / Dimerreihe
        p1 = probs[..., 1]  # 1-Dimer
        p2 = probs[..., 2]  # 2-Dimer
        p3 = probs[..., 3]  # Double-Dimer

        # --- Basis-Prediction OHNE Klasse 2 ---
        # Stapel nur aus Klassen 0,1,3
        others = np.stack([p0, p1, p3], axis=-1)           # (H,W,3)
        others_argmax01_3 = np.argmax(others, axis=-1)     # 0..2 (entspricht Klassen 0,1,3)
        others_max = np.max(others, axis=-1)

        # Mapping zurück auf echte Klassenindizes (0,1,3):
        # 0 -> 0, 1 -> 1, 2 -> 3
        pred = others_argmax01_3.copy()
        pred = np.where(pred == 2, 3, pred).astype(np.uint8)

        # --- Gating für Klasse 2 (2-Dimer) ---
        T_HIGH_2 = 0.45   # Mindestwahrscheinlichkeit für 2-Dimer
        MARGIN_2 = 0.3   # 2-Dimer muss so viel besser sein als andere Klassen

        is_2dimer = (p2 > T_HIGH_2) & ((p2 - others_max) > MARGIN_2)

        # Dort, wo Bedingungen erfüllt sind, setzen wir Klasse 2
        pred[is_2dimer] = 2

        pred_labels = pred  # (H,W), Werte in {0,1,2,3}


        # Tripanel
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Panel 1
        if orig_arr is not None:
            axes[0].imshow(orig_arr, origin="lower")
            axes[0].set_title("Original TIF")
        else:
            axes[0].imshow(gray_arr, cmap="gray", origin="lower")
            axes[0].set_title("Original (PNG-Ersatz)")
        axes[0].axis("off")

        # Panel 2
        axes[1].imshow(gray_arr, cmap="gray", origin="lower")
        axes[1].set_title("Graustufen 600×600")
        axes[1].axis("off")

        # Panel 3
        axes[2].imshow(img_model, cmap="gray", origin="lower")

        H, W = pred_labels.shape
        overlay = np.zeros((H, W, 4), dtype=np.float32)

        color_map = {
            0: (0.0, 0.5, 1.0, 0.4),  # Dimerreihe (halbtransparent, sonst siehst du Signaturen nicht mehr)
            1: (1.0, 0.0, 0.0, 0.7),  # 1-Dimer Signatur
            2: (0.0, 1.0, 0.0, 0.7),  # 2-Dimer Signatur
            3: (1.0, 1.0, 0.0, 0.7),  # Double-Dimer Signatur
        }


        for cls_id, (r, g, b, a) in color_map.items():
            mask = (pred_labels == cls_id)
            overlay[mask] = (r, g, b, a)

        axes[2].imshow(overlay, origin="lower")
        axes[2].set_title("Vorhersage-Overlay")
        axes[2].axis("off")

        handles = [
            mpatches.Patch(color=(0.0, 0.5, 1.0, 0.4), label="Hintergrund"),
            mpatches.Patch(color=(1.0, 0.0, 0.0, 0.7),   label="1-Dimer"),
            mpatches.Patch(color=(0.0, 1.0, 0.0, 0.7),   label="2-Dimer"),
            mpatches.Patch(color=(1.0, 1.0, 0.0, 0.7),   label="Double-Dimer"),
        ]

        axes[2].legend(handles=handles, loc="upper right", fontsize=8)

        fig.suptitle(base_name)
        fig.tight_layout()

        out_path = OUTPUT_DIR / f"{base_name}_tripanel.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    print("Fertig. Tripanel-Visualisierungen liegen in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
