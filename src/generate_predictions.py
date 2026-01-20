import os
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from src.predictions.utils import load_png_gray, load_mask_labels, plot_image_and_labels

# Ordner, in dem dieses Skript liegt: <projekt>/src
ROOT_DIR = Path(__file__).resolve().parent
# Projekt-Root: eine Ebene drüber: <projekt>
PROJECT_ROOT = ROOT_DIR.parent

STM_SUBFOLDER = "stm_dx_1_zones_1_amp_1_wa_1"

# Modell
MODEL_PATH = PROJECT_ROOT / "runs" / "ninth_training_f80" / "model_final.keras"

# Bilder & Masken
VAL_IMAGE_DIR = PROJECT_ROOT / "data" / "ninth_training_fixseventh" / "images" / STM_SUBFOLDER
VAL_MASK_DIR  = PROJECT_ROOT / "data" / "ninth_training_fixseventh" / "masks" / STM_SUBFOLDER

# Output
OUTPUT_DIR = PROJECT_ROOT / "runs" / "ninth_training_f80" / f"predictions_{STM_SUBFOLDER}"

NUM_SAMPLES = 10
RANDOM_SEED = 42

print("PROJECT_ROOT (modul-level):", PROJECT_ROOT)


def main():
    # print("PROJECT_ROOT:", PROJECT_ROOT)
    # print("STM_SUBFOLDER:", STM_SUBFOLDER)
    # print("MODEL_PATH:", MODEL_PATH)
    # print("VAL_IMAGE_DIR:", VAL_IMAGE_DIR)
    # print("VAL_MASK_DIR:", VAL_MASK_DIR)
    # print("OUTPUT_DIR:", OUTPUT_DIR)

    # Path.make dir → funktioniert jetzt, weil OUTPUT_DIR ein Path ist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Modell laden
    print("Lade Modell ...")
    model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    print("Modell geladen.")

    # 2) Bild- und Maskenpfade einsammeln (Path.glob)
    img_paths = sorted(p for p in VAL_IMAGE_DIR.glob("*.png"))
    msk_paths = sorted(p for p in VAL_MASK_DIR.glob("*.png"))

    assert len(img_paths) == len(msk_paths), (
        f"Anzahl Bilder ({len(img_paths)}) != Anzahl Masken ({len(msk_paths)})"
    )
    n_total = len(img_paths)
    print(f"Gefundene Paare: {n_total}")

    if n_total == 0:
        print("Keine Bild/Masken-Paare gefunden. Abbruch.")
        return

    # 3) nur einige Beispiele auswählen
    num_samples = min(NUM_SAMPLES, n_total)
    random.seed(RANDOM_SEED)
    indices = random.sample(range(n_total), num_samples)
    print(f"Sample-Indices: {indices}")

    for idx, i in enumerate(indices):
        img_path = img_paths[i]
        msk_path = msk_paths[i]
        base_name = img_path.stem

        print(f"[{idx+1}/{num_samples}] {base_name}")

        # 4) Daten laden
        image_tf = load_png_gray(img_path)     # (H,W,1) float32
        gt_labels = load_mask_labels(msk_path) # (H,W) int

        # 5) Prediction
        pred_logits = model.predict(
            tf.expand_dims(image_tf, axis=0), verbose=0
        )[0]  # (H,W,C)

        pred_labels = np.argmax(pred_logits, axis=-1)  # (H,W)

        # 6) Plot speichern (Paths statt Strings)
        overlay_path = OUTPUT_DIR / f"{base_name}_vis.png"
        plot_image_and_labels(
            image_tf.numpy(),
            gt_labels,
            pred_labels,
            save_path=overlay_path,
            title=base_name,
        )

        # 7) Pred-Labelmap zusätzlich als PNG (für spätere Vergleiche)
        pred_mask_path = OUTPUT_DIR / f"{base_name}_pred_labels.png"
        plt.imsave(pred_mask_path, pred_labels, cmap="tab10", origin="lower")

    print("Fertig. Visualisierungen liegen in:", OUTPUT_DIR)


if __name__ == "__main__":
    print(">>> __name__ == '__main__' -> main() wird jetzt aufgerufen")
    main()
else:
    print(">>> __name__ != '__main__' -> main() wird NICHT automatisch aufgerufen")
