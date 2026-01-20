# evaluate_unet.py
from __future__ import annotations
import argparse
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import tensorflow as tf

from src.unet.config import Config
from src.unet.data import DatasetBuilder


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate U-Net on test split (multiclass + optional row head)")
    p.add_argument("--model_path", type=str, required=True, help="Pfad zu *.keras (trainiertes Modell)")
    p.add_argument("--images_dir", type=str, default=None, help="Optional: überschreibt Config.images_dir")
    p.add_argument("--masks_dir", type=str, default=None, help="Optional: überschreibt Config.masks_dir")
    p.add_argument("--row_masks_dir", type=str, default=None,
                  help="Optional: Pfad zu 0/1-Row-Masken für Dimerreihen. Wenn gesetzt -> Eval für 'rows'-Head.")
    p.add_argument("--batch_size", type=int, default=None, help="Optional: überschreibt Config.batch_size")
    p.add_argument("--out_dir", type=str, default=None, help="Optional: Ausgabeordner")
    return p.parse_args()


def make_test_dataset(cfg: Config, row_masks_dir: str | None) -> tf.data.Dataset:
    """
    Baut das Test-Dataset. Kein Squeeze mehr, wir lassen die Masken wie im Training:
    - Single-Task: (img, mask) mit mask [H,W,1] int
    - Multi-Task:  (img, {"main": main_mask, "rows": row_mask})
    """
    aug_cfg = {}  # keine Augmentierung für Eval
    dsb = DatasetBuilder(
        cfg.images_dir,
        cfg.masks_dir,
        (cfg.image_height, cfg.image_width),
        cfg.batch_size,
        aug_cfg,
        seed=cfg.random_seed,
        row_masks_dir=row_masks_dir,
    )
    _, _, test_ds, *_ = dsb.train_val_test(cfg.val_fraction, cfg.test_fraction)
    return test_ds


def per_class_iou(cm: np.ndarray) -> np.ndarray:
    # cm: [C,C], Zeilen = true, Spalten = pred
    # IoU_k = TP / (TP + FP + FN)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = tp + fp + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(denom > 0, tp / denom, 0.0)
    return iou


def main():
    args = parse_args()

    # --- Config laden & ggf. überschreiben ---
    cfg = Config()
    if args.images_dir:
        cfg.images_dir = args.images_dir
    if args.masks_dir:
        cfg.masks_dir = args.masks_dir
    if args.batch_size:
        cfg.batch_size = args.batch_size

    # --- Ausgabeordner ---
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        mp = Path(args.model_path)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = (mp.parent / f"eval_{stamp}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Modell laden (nur Inferenz) ---
    model = tf.keras.models.load_model(args.model_path, compile=False)

    # --- Test-Dataset ---
    multi_task = args.row_masks_dir is not None
    test_ds = make_test_dataset(cfg, row_masks_dir=args.row_masks_dir if multi_task else None)

    # --- Vorhersagen sammeln: MAIN-HEAD ---
    y_true_main_all, y_pred_main_all = [], []
    # Optional für Rows
    y_true_rows_all, y_pred_rows_all = [], []

    for xb, yb in test_ds:
        # Vorhersage
        pred = model.predict(xb, verbose=0)

        # --- Fallunterscheidung: Single- vs. Multi-Output ---
        if isinstance(pred, dict):
            # Neues Modell: {"main": ..., "rows": ...}
            probs_main = pred["main"]  # [B,H,W,C]
            # Labels:
            if isinstance(yb, dict):
                yb_main = yb["main"].numpy()    # [B,H,W,1] int (oder one-hot, aber bei dir int)
            else:
                # falls DatasetBuilder doch nur single-task zurückgibt
                yb_main = yb.numpy()
        else:
            # Altes Modell: ein einziger Tensor
            probs_main = pred  # [B,H,W,C]
            if isinstance(yb, dict):
                # etwas schräg, aber zur Sicherheit: nimm "main"
                yb_main = yb["main"].numpy()
            else:
                yb_main = yb.numpy()

        # Hauptklassen-Label [B,H,W]
        if yb_main.ndim == 4 and yb_main.shape[-1] == 1:
            yb_main_cls = np.squeeze(yb_main, axis=-1)
        else:
            yb_main_cls = yb_main

        # argmax über Klassen
        y_pred_main_cls = np.argmax(probs_main, axis=-1)  # [B,H,W]

        y_true_main_all.append(yb_main_cls)
        y_pred_main_all.append(y_pred_main_cls)

        # --- Rows-Head nur auswerten, wenn:
        # 1) Multi-Task-Dataset (row_masks_dir gesetzt)
        # 2) Modell tatsächlich einen 'rows'-Output hat
        if multi_task and isinstance(pred, dict) and "rows" in pred and isinstance(yb, dict) and "rows" in yb:
            probs_rows = pred["rows"]          # [B,H,W,1], Sigmoid
            yb_rows = yb["rows"].numpy()       # [B,H,W,1] float 0/1

            # Ground Truth [B,H,W] int 0/1
            y_true_rows = np.squeeze(yb_rows, axis=-1)
            y_true_rows = (y_true_rows > 0.5).astype(np.uint8)  # falls nicht exakt 0/1

            # Prediction [B,H,W] int 0/1
            y_pred_rows = (probs_rows > 0.5).astype(np.uint8)
            y_pred_rows = np.squeeze(y_pred_rows, axis=-1)

            y_true_rows_all.append(y_true_rows)
            y_pred_rows_all.append(y_pred_rows)

    # --- Flatten für MAIN ---
    y_true_main = np.concatenate([a.reshape(-1) for a in y_true_main_all], axis=0)  # [N_px]
    y_pred_main = np.concatenate([a.reshape(-1) for a in y_pred_main_all], axis=0)  # [N_px]

    # evtl. num_classes bestimmen
    num_classes_cfg = getattr(cfg, "num_classes", None)
    if num_classes_cfg is None:
        num_classes_main = int(y_true_main.max()) + 1
    else:
        num_classes_main = int(num_classes_cfg)
    labels_main = list(range(num_classes_main))

    from sklearn.metrics import classification_report, confusion_matrix

    # --- MAIN-Head Kennzahlen ---
    report_main = classification_report(
        y_true_main,
        y_pred_main,
        labels=labels_main,
        output_dict=True,
        zero_division=0,
    )
    cm_main = confusion_matrix(y_true_main, y_pred_main, labels=labels_main)
    iou_per_class_main = per_class_iou(cm_main)
    miou_main = float(np.mean(iou_per_class_main))

    # --- ROWS-Head Kennzahlen (falls vorhanden) ---
    rows_metrics = None
    if multi_task and len(y_true_rows_all) > 0:
        y_true_rows = np.concatenate([a.reshape(-1) for a in y_true_rows_all], axis=0)
        y_pred_rows = np.concatenate([a.reshape(-1) for a in y_pred_rows_all], axis=0)

        labels_rows = [0, 1]
        report_rows = classification_report(
            y_true_rows,
            y_pred_rows,
            labels=labels_rows,
            output_dict=True,
            zero_division=0,
        )
        cm_rows = confusion_matrix(y_true_rows, y_pred_rows, labels=labels_rows)
        iou_per_class_rows = per_class_iou(cm_rows)
        miou_rows = float(np.mean(iou_per_class_rows))

        rows_metrics = {
            "labels": labels_rows,
            "classification_report": report_rows,
            "confusion_matrix": cm_rows.tolist(),
            "iou_per_class": {str(k): float(v) for k, v in zip(labels_rows, iou_per_class_rows)},
            "mean_iou": miou_rows,
        }

    # --- Speichern ---
    # 1) JSON
    out_json = {
        "main": {
            "labels": labels_main,
            "classification_report": report_main,
            "confusion_matrix": cm_main.tolist(),
            "iou_per_class": {str(k): float(v) for k, v in zip(labels_main, iou_per_class_main)},
            "mean_iou": miou_main,
        }
    }
    if rows_metrics is not None:
        out_json["rows"] = rows_metrics

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2, ensure_ascii=False)

    # 2) Text-Zusammenfassung
    def fmt_pct(x): return f"{100.0 * x:5.2f}%"
    lines = []
    lines.append("=== MAIN-HEAD: Per-Klasse Precision / Recall / F1 ===")
    for k in labels_main:
        r = report_main[str(k)]
        lines.append(
            f"Klasse {k}: Prec {fmt_pct(r['precision'])} | Rec {fmt_pct(r['recall'])} "
            f"| F1 {fmt_pct(r['f1-score'])} | Support {int(r['support'])}"
        )
    lines.append("")
    lines.append("=== MAIN-HEAD: IoU pro Klasse ===")
    for k, iou in zip(labels_main, iou_per_class_main):
        lines.append(f"Klasse {k}: IoU {fmt_pct(iou)}")
    lines.append(f"\nMAIN-HEAD Mean IoU: {fmt_pct(miou_main)}")
    lines.append("\nMAIN-HEAD Confusion-Matrix (rows=true, cols=pred):")
    for row in cm_main:
        lines.append(" ".join(f"{v:8d}" for v in row))

    if rows_metrics is not None:
        lines.append("\n")
        lines.append("=== ROWS-HEAD (Dimerreihen 0/1) ===")
        for k in [0, 1]:
            r = rows_metrics["classification_report"][str(k)]
            lines.append(
                f"Klasse {k}: Prec {fmt_pct(r['precision'])} | Rec {fmt_pct(r['recall'])} "
                f"| F1 {fmt_pct(r['f1-score'])} | Support {int(r['support'])}"
            )
        lines.append("")
        lines.append("=== ROWS-HEAD: IoU ===")
        for k in [0, 1]:
            iou = rows_metrics["iou_per_class"][str(k)]
            lines.append(f"Klasse {k}: IoU {fmt_pct(iou)}")
        lines.append(f"\nROWS-HEAD Mean IoU: {fmt_pct(rows_metrics['mean_iou'])}")
        lines.append("\nROWS-HEAD Confusion-Matrix (rows=true, cols=pred):")
        for row in rows_metrics["confusion_matrix"]:
            lines.append(" ".join(f"{v:8d}" for v in row))

    with open(out_dir / "metrics.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # --- Konsolen-Output (kurz) ---
    print(f"[OK] Auswertung gespeichert in: {out_dir}")
    print(f"MAIN mIoU: {miou_main:.4f}")
    if rows_metrics is not None:
        print(f"ROWS mIoU: {rows_metrics['mean_iou']:.4f}")


if __name__ == "__main__":
    main()
