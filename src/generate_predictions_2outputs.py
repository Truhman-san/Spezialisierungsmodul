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
MODEL_PATH = PROJECT_ROOT / "runs" / "multitask_test" / "model_final.keras"

# Bilder & Masken (Hauptmasken)
VAL_IMAGE_DIR = PROJECT_ROOT / "data" / "training_two_outputs" / "images" / STM_SUBFOLDER
VAL_MASK_DIR  = PROJECT_ROOT / "data" / "training_two_outputs" / "masks"  / STM_SUBFOLDER

# OPTIONAL: Row-Masken (GT) – falls du sie später doch brauchst
ROW_MASK_DIR  = PROJECT_ROOT / "data" / "training_two_outputs" / "row_masks" / STM_SUBFOLDER

# Output
OUTPUT_DIR = PROJECT_ROOT / "runs" / "multitask_test" / f"predictions_{STM_SUBFOLDER}"

NUM_SAMPLES = 10
RANDOM_SEED = 42

print("PROJECT_ROOT (modul-level):", PROJECT_ROOT)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Modell laden
    print("Lade Modell ...")
    model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    print("Modell geladen.")

    # 2) Bild- und Maskenpfade einsammeln
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

    # Check, ob es einen 'rows'-Output gibt (nur informativ)
    has_rows_head = False
    if isinstance(model.output, dict):
        has_rows_head = "rows" in model.output_names or "rows" in model.output

    print(f"Modell hat Rows-Head: {has_rows_head}")

    for idx, i in enumerate(indices):
        img_path = img_paths[i]
        msk_path = msk_paths[i]
        base_name = img_path.stem

        print(f"[{idx+1}/{num_samples}] {base_name}")

        # 4) Daten laden
        image_tf = load_png_gray(img_path)     # (H,W,1) float32 [0,1]
        gt_labels = load_mask_labels(msk_path) # (H,W) int

        # 5) Prediction
        pred_raw = model.predict(
            tf.expand_dims(image_tf, axis=0), verbose=0
        )

        # --- Multi-Output vs. Single-Output ---
        if isinstance(pred_raw, dict):
            # Neues Multi-Task-Modell: {"main": ..., "rows": ...}
            pred_main_logits = pred_raw["main"][0]  # (H,W,C)
            pred_rows_probs  = pred_raw.get("rows", None)
            if pred_rows_probs is not None:
                pred_rows_probs = pred_rows_probs[0, ..., 0]  # (H,W)
        else:
            # Altes Modell: Tensor (H,W,C)
            pred_main_logits = pred_raw[0]  # (H,W,C)
            pred_rows_probs = None

        # Haupt-Labels
        pred_labels = np.argmax(pred_main_logits, axis=-1)  # (H,W)

        # 6) Plot speichern (Haupt-Head)
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

        # 8) Rows-Head (falls vorhanden): binäre 0/1-Maske speichern
        if pred_rows_probs is not None:
            rows_bin = (pred_rows_probs > 0.5).astype(np.uint8)  # (H,W)

            rows_pred_path = OUTPUT_DIR / f"{base_name}_rows_pred.png"
            plt.imsave(rows_pred_path, rows_bin, cmap="gray", origin="lower")

            # Optional: GT-Row-Maske laden, falls vorhanden – nur Logging/Debug
            if ROW_MASK_DIR.exists():
                # heuristische Suche: gleicher Stem, evtl. Suffixe
                candidate_names = [
                    f"{base_name}.png",
                    f"{base_name}_rows.png",
                    f"{base_name}_row.png",
                    f"{base_name}_rowmask.png",
                ]
                row_gt_path = None
                for name in candidate_names:
                    p = ROW_MASK_DIR / name
                    if p.exists():
                        row_gt_path = p
                        break

                if row_gt_path is not None:
                    # wenn du irgendwann auch GT-Row überlagern willst, kannst du hier
                    # load_mask_labels(row_gt_path) benutzen und ein separates Plot bauen.
                    pass

    print("Fertig. Visualisierungen liegen in:", OUTPUT_DIR)


if __name__ == "__main__":
    print(">>> __name__ == '__main__' -> main() wird jetzt aufgerufen")
    main()
else:
    print(">>> __name__ != '__main__' -> main() wird NICHT automatisch aufgerufen")
