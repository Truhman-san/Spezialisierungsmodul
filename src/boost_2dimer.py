from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import matplotlib.patches as mpatches

from src.predictions.utils import load_png_gray 

ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent

ROWS_PATH   = PROJECT_ROOT / "runs" / "multitask_test" / "model_final.keras"

MODEL1_PATH = PROJECT_ROOT / "runs" / "ng_f60" / "model_final.keras"
MODEL2_PATH = PROJECT_ROOT / "runs" / "ngns_f80" / "model_final.keras"

print("Lade Row-Modell ...")
row_model = tf.keras.models.load_model(str(ROWS_PATH), compile=False)
print("Row-Modell geladen.")

print("Lade Defekt-Modelle ...")
model1 = tf.keras.models.load_model(str(MODEL1_PATH), compile=False)
model2 = tf.keras.models.load_model(str(MODEL2_PATH), compile=False)
print("Defekt-Modelle geladen.")

CLASS_IDX_2DIMER = 2

def boost_2dimer_along_rows(
    pred_labels: np.ndarray,
    p_ens: np.ndarray,
    row_prob: np.ndarray,
    row_mask: np.ndarray,
    th_row_prob: float = 0.7,
    th_p2: float = 0.4,
) -> np.ndarray:
    """
    Boostet 2-Dimer (Klasse 2) an Stellen, wo:
      - der Row-Head eine hohe Reihenwahrscheinlichkeit hat
      - das Ensemble eine signifikante 2-Dimer-Wahrscheinlichkeit hat
      - bisher Hintergrund (0) vorhergesagt wurde.

    Shapes:
        pred_labels: (H, W) uint8
        p_ens:       (H, W, C) float32
        row_prob:    (H, W) float32
        row_mask:    (H, W) uint8/bool (optional, hier nur zur Sicherheit)
    """
    boosted = pred_labels.copy()

    p2 = p_ens[..., CLASS_IDX_2DIMER]  # 2-Dimer-Wahrscheinlichkeiten

    candidates = (
        (row_prob > th_row_prob) &          # Row-Head ist sich sicher
        (row_mask.astype(bool)) &           # Pixel liegt in der Row-Maske
        (boosted == 0) &                    # bisher Hintergrund
        (p2 > th_p2)                        # Ensemble sieht 2-Dimer-Peak
    )

    boosted[candidates] = CLASS_IDX_2DIMER

    return boosted


def ensemble_defect_predict(image_tf):
    """
    image_tf: (H,W,1) float32 [0,1]
    Returns:
        final_pred: (H,W) uint8 (Klassenlabels nach Heuristiken)
        p_ens:      (H,W,C) float32 (gemittelte Probabilities)
    """
    x = tf.expand_dims(image_tf, axis=0)

    # Vorhersagen beider Modelle
    logits1 = model1.predict(x, verbose=0)[0]  
    logits2 = model2.predict(x, verbose=0)[0] 

    p1 = logits1
    p2 = logits2

    pred1 = np.argmax(p1, axis=-1).astype(np.uint8)   
    conf1 = np.max(p1, axis=-1).astype(np.float32)   

    pred2 = np.argmax(p2, axis=-1).astype(np.uint8)
    conf2 = np.max(p2, axis=-1).astype(np.float32)

    final_pred = pred1.copy()

    class_idx_2dimer = 2
    p1_2d = p1[..., class_idx_2dimer]

    # --- Regel 1: Modell 2 darf Defekte hinzufügen ---
    T_ADD = 0.75
    T_SUPPORT_2D = 0.80

    base_add_mask = (pred1 == 0) & (pred2 != 0) & (conf2 > T_ADD)

    add_mask_non2 = base_add_mask & (pred2 != class_idx_2dimer)
    add_mask_2 = (
        base_add_mask &
        (pred2 == class_idx_2dimer) &
        (p1_2d > T_SUPPORT_2D)
    )

    add_mask = add_mask_non2 | add_mask_2
    final_pred[add_mask] = pred2[add_mask]

    # --- Regel 2: Modell 2 darf unsichere Defekte löschen ---
    T_CLEAR = 0.65
    T_LOW   = 0.65

    clear_mask = (
        (pred1 != 0) &
        (pred2 == 0) &
        (conf2 > T_CLEAR) &
        (conf1 < T_LOW)
    )
    final_pred[clear_mask] = 0

    # Gemitteltes Ensemble als Probabilities
    p_ens = 0.5 * (p1 + p2)

    return final_pred, p_ens


def predict_pipeline(image_tf):
    """
    image_tf: (H,W,1) float32 [0,1] (wie aus load_png_gray)
    Returns:
        row_mask:    (H,W) uint8 {0,1}
        pred_labels: (H,W) uint8 (mit 2-Dimer-Boost)
        p_ens:       (H,W,C) Probabilities (vom Defektensemble)
        p_base:      (H,W,C) ungeänderte Ensemble-Probabilities
    """
    img_np = np.array(image_tf.numpy(), dtype=np.float32)  
    x = np.expand_dims(img_np, axis=0)  

    # --- Stage 1: Rows aus Multitask-Modell ---
    preds_row = row_model.predict(x, verbose=0)

    if isinstance(preds_row, dict):
        row_prob = preds_row["rows"][0, ..., 0]
    else:
        out_names = list(getattr(row_model, "output_names", []))
        rows_idx = out_names.index("rows")
        row_prob = preds_row[rows_idx][0, ..., 0]

    row_mask = (row_prob > 0.5).astype(np.uint8)

    # --- Stage 2: Ensemble-Defekte (ohne hartes Gating) ---
    _, p_ens = ensemble_defect_predict(image_tf) 

    # Basisprediction des Ensembles
    pred_base = np.argmax(p_ens, axis=-1).astype(np.uint8)

    # --- Stage 3: 2-Dimer-Boost entlang der Reihen ---
    pred_boosted = boost_2dimer_along_rows(
        pred_labels=pred_base,
        p_ens=p_ens,
        row_prob=row_prob,
        row_mask=row_mask,
        th_row_prob=0.7,
        th_p2=0.4,
    )

    return row_mask, pred_boosted, p_ens, p_ens



# ---- Skript-Teil für Batch-Predictions und Tripanels ----

REAL_IMAGE_DIR = PROJECT_ROOT / "real_stm_converted_png"
OUTPUT_DIR     = PROJECT_ROOT / "runs" / "pipeline_rows_plus_ensemble" / "predictions_real_tripanel_2dimer_boost"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(REAL_IMAGE_DIR.glob("*.png"))
    n_total = len(img_paths)
    print(f"Gefundene Realbilder (PNG): {n_total}")

    if n_total == 0:
        print("Keine Realbilder gefunden. Abbruch.")
        return

    for idx, img_path in enumerate(img_paths, start=1):
        base_name = img_path.stem
        print(f"[{idx}/{n_total}] {base_name}")

        image_tf = load_png_gray(img_path)    
        img_model = image_tf.numpy()[..., 0]     
        gray_arr = np.array(Image.open(img_path))

        # --- Pipeline aufrufen ---
        row_mask, pred_labels, p_ens, gated_def = predict_pipeline(image_tf)
        H, W = pred_labels.shape

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        ax_orig, ax_def, ax_rows = axes

        # Panel 1: Original-Graustufe
        ax_orig.imshow(gray_arr, cmap="gray", origin="lower")
        ax_orig.set_title("Graustufen 600×600")
        ax_orig.axis("off")

        # Panel 2: Defekt-Vorhersage (gated)
        ax_def.imshow(img_model, cmap="gray", origin="lower")
        overlay = np.zeros((H, W, 4), dtype=np.float32)

        color_map = {
            1: (1.0, 0.0, 0.0, 0.7),  # 1-Dimer
            2: (0.0, 1.0, 0.0, 0.7),  # 2-Dimer
            3: (1.0, 1.0, 0.0, 0.7),  # Double-Dimer
        }

        for cls_id, (r, g, b, a) in color_map.items():
            mask = (pred_labels == cls_id)
            overlay[mask] = (r, g, b, a)

        ax_def.imshow(overlay, origin="lower")
        ax_def.set_title("Defekte (Ensemble nur auf Hintergrund)")
        ax_def.axis("off")

        # Panel 3: Rows
        ax_rows.imshow(img_model, cmap="gray", origin="lower")
        overlay_rows = np.zeros((H, W, 4), dtype=np.float32)
        overlay_rows[row_mask.astype(bool)] = (0.0, 0.5, 1.0, 0.4)
        ax_rows.imshow(overlay_rows, origin="lower")
        ax_rows.set_title("Dimerreihen (Rows-Head)")
        ax_rows.axis("off")

        fig.suptitle(base_name)
        fig.tight_layout()
        out_path = OUTPUT_DIR / f"{base_name}_tripanel_pipeline.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    print("Fertig. Tripanel-Visualisierungen liegen in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
