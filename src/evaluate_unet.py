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
    p = argparse.ArgumentParser(description="Evaluate U-Net on test split (multiclass, per-pixel)")
    p.add_argument("--model_path", type=str, required=True, help="Pfad zu *.keras (trainiertes Modell)")
    p.add_argument("--images_dir", type=str, default=None, help="Optional: überschreibt Config.images_dir")
    p.add_argument("--masks_dir", type=str, default=None, help="Optional: überschreibt Config.masks_dir")
    p.add_argument("--batch_size", type=int, default=None, help="Optional: überschreibt Config.batch_size")
    p.add_argument("--out_dir", type=str, default=None, help="Optional: Ausgabeordner")
    return p.parse_args()

def make_test_dataset(cfg: Config) -> tf.data.Dataset:
    aug_cfg = {} 
    dsb = DatasetBuilder(
        cfg.images_dir,
        cfg.masks_dir,
        (cfg.image_height, cfg.image_width),
        cfg.batch_size,
        aug_cfg,
        seed=cfg.random_seed,
    )
    _, _, test_ds, *_ = dsb.train_val_test(cfg.val_fraction, cfg.test_fraction)
    test_ds = test_ds.map(lambda i, m: (i, tf.squeeze(m, axis=-1)))
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
    if args.images_dir: cfg.images_dir = args.images_dir
    if args.masks_dir:  cfg.masks_dir  = args.masks_dir
    if args.batch_size: cfg.batch_size = args.batch_size

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
    test_ds = make_test_dataset(cfg)

    # --- Vorhersagen sammeln ---
    y_true_all, y_pred_all = [], []
    for xb, yb in test_ds:
        probs = model.predict(xb, verbose=0)          
        y_pred_cls = np.argmax(probs, axis=-1)      
        y_true_all.append(yb.numpy())             
        y_pred_all.append(y_pred_cls)

    y_true = np.concatenate([a.reshape(-1) for a in y_true_all], axis=0)  
    y_pred = np.concatenate([a.reshape(-1) for a in y_pred_all], axis=0)  

    # --- Kennzahlen ---
    # sklearn für Precision/Recall/F1 pro Klasse
    from sklearn.metrics import classification_report, confusion_matrix

    labels = [0, 1, 2, 3]
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    iou_per_class = per_class_iou(cm)
    miou = float(np.mean(iou_per_class))

    # --- Speichern ---
    # 1) JSON
    out_json = {
        "labels": labels,
        "classification_report": report, 
        "confusion_matrix": cm.tolist(),
        "iou_per_class": {str(k): float(v) for k, v in zip(labels, iou_per_class)},
        "mean_iou": miou,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2, ensure_ascii=False)

    # 2) Text-Zusammenfassung
    def fmt_pct(x): return f"{100.0 * x:5.2f}%"
    lines = []
    lines.append("=== Per-Klasse Precision / Recall / F1 ===")
    for k in labels:
        r = report[str(k)]
        lines.append(
            f"Klasse {k}: Prec {fmt_pct(r['precision'])} | Rec {fmt_pct(r['recall'])} | F1 {fmt_pct(r['f1-score'])} | Support {int(r['support'])}"
        )
    lines.append("")
    lines.append("=== IoU pro Klasse ===")
    for k, iou in zip(labels, iou_per_class):
        lines.append(f"Klasse {k}: IoU {fmt_pct(iou)}")
    lines.append(f"\nMean IoU: {fmt_pct(miou)}")
    lines.append("\nConfusion-Matrix (rows=true, cols=pred):")
    for row in cm:
        lines.append(" ".join(f"{v:8d}" for v in row))

    with open(out_dir / "metrics.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # --- Konsolen-Output (kurz) ---
    print(f"[OK] Auswertung gespeichert in: {out_dir}")
    print(f"mIoU: {miou:.4f}")
    for k, iou in zip(labels, iou_per_class):
        print(f"  IoU[{k}] = {iou:.4f}")

if __name__ == "__main__":
    main()
