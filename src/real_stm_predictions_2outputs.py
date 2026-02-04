import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import matplotlib.patches as mpatches

from src.predictions.utils import load_png_gray 

ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent

MODEL_PATH     = PROJECT_ROOT / "runs" / "multitask_test" / "model_final.keras"
REAL_IMAGE_DIR = PROJECT_ROOT / "real_stm_converted_png"
OUTPUT_DIR     = PROJECT_ROOT / "runs" / "multitask_test" / "predictions_real_tripanel"


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

    has_rows_head = False
    if isinstance(model.output, dict):
        has_rows_head = "rows" in model.output
    elif isinstance(model.output, (list, tuple)):
        has_rows_head = "rows" in getattr(model, "output_names", [])

    print(f"Modell hat Rows-Head: {has_rows_head}")

    for idx, img_path in enumerate(img_paths, start=1):
        base_name = img_path.stem
        print(f"[{idx}/{n_total}] {base_name}")

        # Modell-Input
        image_tf = load_png_gray(img_path)       
        img_model = image_tf.numpy()[..., 0]    
        gray_arr = np.array(Image.open(img_path)) 

        # Prediction
        pred_raw = model.predict(
            tf.expand_dims(image_tf, axis=0), verbose=0
        )

        # --- Multi- vs Single-Output ---
        if isinstance(pred_raw, dict):
            # Multi-Task-Modell
            pred_main_logits = pred_raw["main"][0]  
            pred_rows_probs = None
            if has_rows_head and "rows" in pred_raw:
                pred_rows_probs = pred_raw["rows"][0, ..., 0] 
        else:
            # Single-Output-Modell
            pred_main_logits = pred_raw[0] 
            pred_rows_probs = None

        # Klassenlabels (main-Head)
        pred_labels = np.argmax(pred_main_logits, axis=-1).astype(np.uint8)  

        # Anzahl Spalten dynamisch: ohne rows 2 Panels, mit rows 3 Panels
        n_cols = 3 if pred_rows_probs is not None else 2
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

        if n_cols == 2:
            ax_gray, ax_main = axes
            ax_rows = None
        else:
            ax_gray, ax_main, ax_rows = axes

        # Panel 1: Graustufenbild
        ax_gray.imshow(gray_arr, cmap="gray", origin="lower")
        ax_gray.set_title("Graustufen 600×600")
        ax_gray.axis("off")

        H, W = pred_labels.shape

        # -----------------------------
        # Panel 2: Defekt-Vorhersage (main-Head)
        # -----------------------------
        ax_main.imshow(img_model, cmap="gray", origin="lower")

        overlay_main = np.zeros((H, W, 4), dtype=np.float32)

        # 0 = Hintergrund, 1..3 = Defektklassen
        color_map_main = {
            1: (1.0, 0.0, 0.0, 0.7),  # 1-Dimer Signatur
            2: (0.0, 1.0, 0.0, 0.7),  # 2-Dimer Signatur
            3: (1.0, 1.0, 0.0, 0.7),  # Double-Dimer Signatur
        }

        for cls_id, (r, g, b, a) in color_map_main.items():
            mask = (pred_labels == cls_id)
            overlay_main[mask] = (r, g, b, a)

        ax_main.imshow(overlay_main, origin="lower")
        ax_main.set_title("Defekt-Vorhersage (main)")
        ax_main.axis("off")

        handles_main = [
            mpatches.Patch(color=(1.0, 0.0, 0.0, 0.7),   label="1-Dimer"),
            mpatches.Patch(color=(0.0, 1.0, 0.0, 0.7),   label="2-Dimer"),
            mpatches.Patch(color=(1.0, 1.0, 0.0, 0.7),   label="Double-Dimer"),
        ]
        ax_main.legend(handles=handles_main, loc="upper right", fontsize=8)

        # -----------------------------
        # Panel 3: Dimerreihen-Vorhersage (rows-Head)
        # -----------------------------
        if ax_rows is not None and pred_rows_probs is not None:
            ax_rows.imshow(img_model, cmap="gray", origin="lower")

            rows_mask = pred_rows_probs > 0.5 

            overlay_rows = np.zeros((H, W, 4), dtype=np.float32)
            overlay_rows[rows_mask] = (0.0, 0.5, 1.0, 0.4)

            ax_rows.imshow(overlay_rows, origin="lower")
            ax_rows.set_title("Dimerreihen-Vorhersage (rows)")
            ax_rows.axis("off")

            handles_rows = [
                mpatches.Patch(color=(0.0, 0.5, 1.0, 0.4), label="Dimerreihe"),
            ]
            ax_rows.legend(handles=handles_rows, loc="upper right", fontsize=8)

        fig.suptitle(base_name)
        fig.tight_layout()

        out_path = OUTPUT_DIR / f"{base_name}_tripanel.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    print("Fertig. Tripanel-Visualisierungen liegen in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
