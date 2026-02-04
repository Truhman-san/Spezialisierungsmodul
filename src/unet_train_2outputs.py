from __future__ import annotations
import argparse
from pathlib import Path
import os
import json
import tensorflow as tf

from src.unet.config import Config
from src.unet.data import DatasetBuilder
from src.unet.model import build_unet
from src.unet.losses import focal_plus_dice
from src.unet.utils import prepare_output_dir, save_run_config
from src.unet.metric_train import mean_iou_metric, mean_f1_metric


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train U-Net (multiclass + optional row head)")
    p.add_argument("--images_dir", type=str, default="data/training_two_outputs/images")
    p.add_argument("--masks_dir", type=str, default="data/training_two_outputs/masks")
    p.add_argument(
        "--row_masks_dir",
        type=str,
        default="data/training_two_outputs/masks_row",
        help="Pfad zu den 0/1-Dimerreihenmasken (PNG). Wenn gesetzt -> Multi-Task mit 2 Outputs.",
    )

    p.add_argument("--output_root", type=str, default="runs")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--train_fraction",
        type=float,
        default=1.0,
        help="Anteil der Trainingsdaten, der verwendet wird (0 < f ≤ 1)."
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Config
    cfg = Config(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        output_root=args.output_root,
        run_name=args.run_name,
    )
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.seed is not None:
        cfg.random_seed = args.seed

    tf.keras.utils.set_random_seed(cfg.random_seed)

    out_dir = cfg.output_dir()
    prepare_output_dir(out_dir)
    save_run_config(out_dir, cfg)

    # ---------------------------------
    # Data
    # ---------------------------------
    aug_cfg = dict(
        rotate=cfg.aug_rotate,
        flip_h=cfg.aug_flip_h,
        flip_v=cfg.aug_flip_v,
        random_zoom=cfg.aug_random_zoom,
        zoom_range=cfg.aug_zoom_range,
    )

    # Multi-Task aktiv, wenn Row-Masken angegeben
    multi_task = args.row_masks_dir is not None

    dsb = DatasetBuilder(
        cfg.images_dir,
        cfg.masks_dir,
        (cfg.image_height, cfg.image_width),
        cfg.batch_size,
        aug_cfg,
        seed=cfg.random_seed,
        row_masks_dir=args.row_masks_dir if multi_task else None,
    )
    train_ds, val_ds, test_ds, *_ = dsb.train_val_test(cfg.val_fraction, cfg.test_fraction)

    # ---- Trainings-Subset für Learning Curve ----
    train_fraction = args.train_fraction
    if not (0 < train_fraction <= 1.0):
        raise ValueError(f"train_fraction muss in (0,1] liegen, bekommen: {train_fraction}")

    # Cardinality in Batches bestimmen
    train_steps_full = tf.data.experimental.cardinality(train_ds).numpy()
    val_steps        = tf.data.experimental.cardinality(val_ds).numpy()
    test_steps       = tf.data.experimental.cardinality(test_ds).numpy()

    if train_steps_full <= 0:
        raise RuntimeError("Train-Set hat unbekannte oder leere Cardinality (kein Subsampling möglich).")

    subset_steps = max(1, int(round(train_steps_full * train_fraction)))

    train_ds = train_ds.take(subset_steps)

    train_size = subset_steps * cfg.batch_size
    val_size   = val_steps * cfg.batch_size
    test_size  = test_steps * cfg.batch_size

    stats_txt = out_dir / "dataset_stats.txt"
    with stats_txt.open("w", encoding="utf-8") as f:
        f.write(f"train_fraction: {train_fraction}\n")
        f.write(f"train_steps: {subset_steps}\n")
        f.write(f"val_steps:   {val_steps}\n")
        f.write(f"test_steps:  {test_steps}\n")
        f.write(f"batch_size:  {cfg.batch_size}\n")
        f.write(f"approx_train_samples: {train_size}\n")
        f.write(f"approx_val_samples:   {val_size}\n")
        f.write(f"approx_test_samples:  {test_size}\n")

    print(f"[INFO] train_fraction={train_fraction}, approx_train_samples={train_size}")


    num_classes = getattr(cfg, "num_classes", 4)

    if cfg.use_mixed_precision:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")

    # ---------------------------------
    # Modell
    # ---------------------------------
    model = build_unet(
        input_shape=(cfg.image_height, cfg.image_width, cfg.channels),
        num_classes=num_classes,
        add_row_head=multi_task,   
    )

    opt = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)

    # Loss für Head "main"
    main_loss = focal_plus_dice(
        num_classes=num_classes,
        focal_w=0.5,
        dice_w=0.5,
        gamma=2.0,
        alpha=0.25,
        from_logits=False,
    )

    if multi_task:
        losses = {
            "main": main_loss,
            "rows": tf.keras.losses.BinaryCrossentropy(from_logits=False),
        }
        loss_weights = {
            "main": 1.0,
            "rows": 0.5,  
        }
        metrics = {
            "main": [
                mean_iou_metric(num_classes=num_classes, ignore_background=True),
                mean_f1_metric(num_classes=num_classes, ignore_background=True),
            ],
            "rows": [
                tf.keras.metrics.BinaryAccuracy(name="rows_bin_acc"),
                tf.keras.metrics.MeanIoU(num_classes=2, name="rows_iou"),
            ],
        }
        monitor_metric = "val_main_mean_iou"
    else:
        losses = main_loss
        loss_weights = None
        metrics = [
            mean_iou_metric(num_classes=num_classes, ignore_background=True),
            mean_f1_metric(num_classes=num_classes, ignore_background=True),
        ]
        monitor_metric = "val_mean_iou"

    model.compile(
        optimizer=opt,
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics,
    )

    # ---------------------------------
    # Callbacks
    # ---------------------------------
    cbs = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(out_dir / "checkpoints" / "ckpt_best.keras"),
            monitor=monitor_metric,
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor_metric,
            mode="max",
            factor=0.5,
            patience=3,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor_metric,
            mode="max",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(out_dir / "training_log.csv")),
    ]

    # ---------------------------------
    # Train
    # ---------------------------------
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=cbs,
    )

    # Save
    model.save(out_dir / "model_final.keras")

    # Test
    eval_metrics = model.evaluate(test_ds, return_dict=True)
    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2)

    print("Saved to:", out_dir)


if __name__ == "__main__":
    if os.environ.get("TF_USE_CPU", "0") == "1":
        tf.config.set_visible_devices([], "GPU")
    main()
