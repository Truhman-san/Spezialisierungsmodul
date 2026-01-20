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

MODEL_PATH     = PROJECT_ROOT / "runs" / "second_training_finetuning" / "model_final.keras"
RAW_TIF_DIR    = PROJECT_ROOT / "real_stm_raw_tifs"
REAL_IMAGE_DIR = PROJECT_ROOT / "real_stm_converted_png"
OUTPUT_DIR     = PROJECT_ROOT / "runs" / "second_training_finetuning" / "predictions_real_tripanel"

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
        )[0]  # (H,W,C)

        pred_labels = np.argmax(pred_logits, axis=-1).astype(np.uint8)

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
