from __future__ import annotations
import os
from pathlib import Path
import tensorflow as tf
from sklearn.model_selection import train_test_split


AUTOTUNE = tf.data.AUTOTUNE


class DatasetBuilder:
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        image_size: tuple[int, int],
        batch_size: int,
        aug_cfg: dict[str, object] | None = None,
        seed: int = 42,
        num_classes: int = 4,  
        one_hot_masks: bool = False,
        row_masks_dir: str | None = None,
        one_hot_row_masks: bool = False, 
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = image_size
        self.batch_size = batch_size
        self.aug_cfg = aug_cfg or {}
        self.seed = seed
        self.num_classes = num_classes
        self.one_hot_masks = one_hot_masks

        self.row_masks_dir = Path(row_masks_dir) if row_masks_dir is not None else None
        self.one_hot_row_masks = one_hot_row_masks

        self._pairs = self._match_image_mask_pairs()

    # -------------------------------------------------------------------------
    # Matching von Bildern, Hauptmasken und optional Row-Masken
    # -------------------------------------------------------------------------
    def _match_image_mask_pairs(self) -> list[tuple[str, ...]]:
        """
        - Bilder:        stm_<nummer>.png
        - Hauptmasken:   stm_<nummer>_mask-sig.png
        - Row-Masken:    stm_<nummer>_mask-row.png
        """
        # Alle PNGs rekursiv einsammeln
        img_files = sorted(self.images_dir.rglob("*.png"))
        msk_files = sorted(self.masks_dir.rglob("*.png"))

        # Hilfsfunktionen für Schlüssel-Normalisierung
        def norm_img_key(p: Path) -> str:
            # z.B. stm_000123.png -> "stm_000123"
            return p.stem

        def norm_main_key(p: Path) -> str:
            """
            stm_000123_mask-sig  -> stm_000123
            stm_000123_mask      -> stm_000123
            stm_000123_maskXYZ   -> stm_000123
            """
            key = p.stem
            if "_mask" in key:
                key = key.split("_mask", 1)[0]
            return key

        def norm_row_key(p: Path) -> str:
            """
            Row-Masken:
            stm_000123_mask-row  -> stm_000123
            plus ein paar generische Fälle wie *_rows, *_rowmask, *_row
            """
            key = p.stem
            if key.endswith("_mask_row"):
                return key[: -len("_mask_row")]
            if key.endswith("_rows"):
                return key[: -len("_rows")]
            if key.endswith("_rowmask"):
                return key[: -len("_rowmask")]
            if key.endswith("_row"):
                return key[: -len("_row")]
            if "_row" in key:
                key = key.split("_row", 1)[0]
            return key

        # Bilder nach key indexieren
        img_by_key: dict[str, Path] = {}
        for p in img_files:
            key = norm_img_key(p)
            img_by_key[key] = p

        # Hauptmasken nach normalisiertem key indexieren
        msk_by_key: dict[str, Path] = {}
        for p in msk_files:
            key = norm_main_key(p)
            msk_by_key[key] = p

        if self.row_masks_dir is None:
            common_keys = sorted(set(img_by_key.keys()) & set(msk_by_key.keys()))
            pairs = [(str(img_by_key[k]), str(msk_by_key[k])) for k in common_keys]
        else:
            # Multi-Task: zusätzlich Row-Masken
            row_files = sorted(self.row_masks_dir.rglob("*.png"))
            row_by_key: dict[str, Path] = {}
            for p in row_files:
                key = norm_row_key(p)
                row_by_key[key] = p

            common_keys = sorted(
                set(img_by_key.keys()) &
                set(msk_by_key.keys()) &
                set(row_by_key.keys())
            )

            pairs = [
                (str(img_by_key[k]), str(msk_by_key[k]), str(row_by_key[k]))
                for k in common_keys
            ]

        if not pairs:
            msg = (
                "No matching PNG filenames found between images_dir, masks_dir "
                "und ggf. row_masks_dir.\n"
                f"  #images: {len(img_files)}, #masks: {len(msk_files)}, "
                f"#row_masks: {len(list(self.row_masks_dir.rglob('*.png'))) if self.row_masks_dir else 0}\n"
                f"  Beispiel image-keys: {list(sorted({norm_img_key(p) for p in img_files}) )[:5]}\n"
                f"  Beispiel mask-keys:  {list(sorted({norm_main_key(p) for p in msk_files}))[:5]}"
            )
            if self.row_masks_dir is not None:
                row_files = list(self.row_masks_dir.rglob("*.png"))
                msg += f"\n  Beispiel row-keys:   {list(sorted({norm_row_key(p) for p in row_files}))[:5]}"
            raise FileNotFoundError(msg)

        return pairs

    # -------------------------------------------------------------------------
    # PNG laden / parsen
    # -------------------------------------------------------------------------
    def _parse_png_pair(
        self,
        img_path: tf.Tensor,
        msk_path: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        img_bytes = tf.io.read_file(img_path)
        msk_bytes = tf.io.read_file(msk_path)

        # 1-Kanal lesen
        img = tf.image.decode_png(img_bytes, channels=1)  
        msk = tf.image.decode_png(msk_bytes, channels=1)  

        # Bild normalisieren
        img = tf.image.convert_image_dtype(img, tf.float32)  

        msk = tf.cast(msk, tf.int32)  

        # Resize
        h, w = self.image_size
        img = tf.image.resize(img, (h, w), method="bilinear")
        msk = tf.image.resize(msk, (h, w), method="nearest")

        # Optional: one-hot für Hauptmaske
        if self.one_hot_masks:
            msk_2d = tf.squeeze(msk, axis=-1) 
            msk = tf.one_hot(
                msk_2d, depth=self.num_classes, dtype=tf.float32
            )  
        else:
            pass

        return img, msk

    def _parse_png_triplet(
        self,
        img_path: tf.Tensor,
        msk_path: tf.Tensor,
        row_path: tf.Tensor,
    ) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
        """
        Multi-Task:
        - "main": mehrklassige Defektmaske 
        - "rows": Binärmaske 0/1 für Dimerreihen
        """
        img, main_mask = self._parse_png_pair(img_path, msk_path)

        # Row-Maske laden
        row_bytes = tf.io.read_file(row_path)
        row = tf.image.decode_png(row_bytes, channels=1) 
        row = tf.cast(row, tf.int32)

        h, w = self.image_size
        row = tf.image.resize(row, (h, w), method="nearest")

        row = tf.where(row > 0, 1, 0)

        if self.one_hot_row_masks:
            row_2d = tf.squeeze(row, axis=-1)  
            row = tf.one_hot(row_2d, depth=2, dtype=tf.float32)  
        else:
            row = tf.cast(row, tf.float32)  

        labels = {
            "main": main_mask,
            "rows": row,
        }
        return img, labels

    # -------------------------------------------------------------------------
    # Augmentierung 
    # -------------------------------------------------------------------------
    def _apply_to_labels(self, labels, fn):
        if isinstance(labels, dict):
            return {k: fn(v) for k, v in labels.items()}
        else:
            return fn(labels)

    def _augment(self, img: tf.Tensor, labels):
        cfg = self.aug_cfg

        # Horizontal Flip
        if cfg.get("flip_h", True):
            do = tf.random.uniform(())
            def _flip_h(x): return tf.image.flip_left_right(x)

            img, labels = tf.cond(
                do < 0.5,
                lambda: (
                    tf.image.flip_left_right(img),
                    self._apply_to_labels(labels, _flip_h),
                ),
                lambda: (img, labels),
            )

        # Vertikal Flip
        if cfg.get("flip_v", False):
            do = tf.random.uniform(())
            def _flip_v(x): return tf.image.flip_up_down(x)

            img, labels = tf.cond(
                do < 0.1,
                lambda: (
                    tf.image.flip_up_down(img),
                    self._apply_to_labels(labels, _flip_v),
                ),
                lambda: (img, labels),
            )

        # Rotation 0,90,180,270
        if cfg.get("rotate", True):
            k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)

            def _rot(x): return tf.image.rot90(x, k)

            img = tf.image.rot90(img, k)
            labels = self._apply_to_labels(labels, _rot)

        # Random Zoom
        if cfg.get("random_zoom", True):
            zr = cfg.get("zoom_range", 0.1)
            if zr and zr > 0:
                scale = 1.0 + tf.random.uniform((), -zr, zr)
                h = tf.cast(tf.shape(img)[0], tf.int32)
                w = tf.cast(tf.shape(img)[1], tf.int32)
                nh = tf.cast(tf.round(scale * tf.cast(h, tf.float32)), tf.int32)
                nw = tf.cast(tf.round(scale * tf.cast(w, tf.float32)), tf.int32)

                img = tf.image.resize(img, (nh, nw), method="bilinear")

                def _resize_mask(x):
                    return tf.image.resize(x, (nh, nw), method="nearest")

                labels = self._apply_to_labels(labels, _resize_mask)

                img = tf.image.resize_with_crop_or_pad(img, h, w)

                def _crop_or_pad(x):
                    return tf.image.resize_with_crop_or_pad(x, h, w)

                labels = self._apply_to_labels(labels, _crop_or_pad)

        return img, labels

    # -------------------------------------------------------------------------
    # Dataset bauen
    # -------------------------------------------------------------------------
    def _build_ds(self, file_pairs: list[tuple[str, ...]], training: bool) -> tf.data.Dataset:
        if self.row_masks_dir is None:
            # Single-Task
            img_paths = [p[0] for p in file_pairs]
            msk_paths = [p[1] for p in file_pairs]
            ds = tf.data.Dataset.from_tensor_slices((img_paths, msk_paths))
            ds = ds.map(
                lambda ip, mp: self._parse_png_pair(ip, mp),
                num_parallel_calls=AUTOTUNE,
            )
        else:
            # Multi-Task: zusätzlich Row-Maske
            img_paths = [p[0] for p in file_pairs]
            msk_paths = [p[1] for p in file_pairs]
            row_paths = [p[2] for p in file_pairs]
            ds = tf.data.Dataset.from_tensor_slices((img_paths, msk_paths, row_paths))
            ds = ds.map(
                lambda ip, mp, rp: self._parse_png_triplet(ip, mp, rp),
                num_parallel_calls=AUTOTUNE,
            )

        if training:
            ds = ds.shuffle(len(file_pairs), seed=self.seed, reshuffle_each_iteration=True)
            ds = ds.map(
                lambda i, y: self._augment(i, y),
                num_parallel_calls=AUTOTUNE,
            )

        options = tf.data.Options()
        options.experimental_deterministic = False
        ds = ds.with_options(options)

        ds = ds.batch(self.batch_size, drop_remainder=False).prefetch(AUTOTUNE)
        return ds

    # -------------------------------------------------------------------------
    # Split in Train / Val / Test
    # -------------------------------------------------------------------------
    def train_val_test(
        self,
        val_frac: float,
        test_frac: float,
    ) -> tuple[
        tf.data.Dataset,
        tf.data.Dataset,
        tf.data.Dataset,
        list[tuple[str, ...]],
        list[tuple[str, ...]],
        list[tuple[str, ...]],
    ]:
        pairs = self._pairs
        # first split off test, then val from remaining
        train_pairs, test_pairs = train_test_split(
            pairs,
            test_size=test_frac,
            random_state=self.seed,
            shuffle=True,
        )
        train_pairs, val_pairs = train_test_split(
            train_pairs,
            test_size=val_frac / (1.0 - test_frac),
            random_state=self.seed,
            shuffle=True,
        )
        return (
            self._build_ds(train_pairs, training=True),
            self._build_ds(val_pairs, training=False),
            self._build_ds(test_pairs, training=False),
            train_pairs,
            val_pairs,
            test_pairs,
        )
