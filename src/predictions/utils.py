from pathlib import Path
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def load_png_gray(path: Path) -> tf.Tensor:
    """PNG als Graustufenbild laden, float32 [0,1]."""
    image = tf.io.read_file(str(path))
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image


def load_mask_labels(path: Path) -> np.ndarray:
    """
    Maske laden, als Integer-Labelmap (0..K-1).
    Wir gehen davon aus, dass die PNG direkt Klassenlabels 0..3 enthält.
    """
    raw = tf.io.read_file(str(path))
    mask = tf.image.decode_png(raw, channels=1)  # (H,W,1), dtype uint8/int
    mask = tf.squeeze(mask, axis=-1)            # (H,W)
    return mask.numpy()


def plot_image_and_labels(image: np.ndarray,
                          gt_labels: np.ndarray,
                          pred_labels: np.ndarray,
                          save_path: Path,
                          title: str = ""):
    """
    Bild, GT-Labelmap und Pred-Labelmap nebeneinander plotten.
    image: (H,W) oder (H,W,1)
    gt_labels/pred_labels: (H,W) mit Werten 0..4
    """
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image, cmap="gray", origin="lower")
    axes[0].set_title("Image")
    axes[0].axis("off")

    # 5 Klassen: 0..4
    vmin, vmax = 0, 4

    im1 = axes[1].imshow(gt_labels, cmap="tab10", origin="lower", vmin=vmin, vmax=vmax)
    axes[1].set_title("GT labels")
    axes[1].axis("off")

    im2 = axes[2].imshow(pred_labels, cmap="tab10", origin="lower", vmin=vmin, vmax=vmax)
    axes[2].set_title("Pred labels")
    axes[2].axis("off")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

