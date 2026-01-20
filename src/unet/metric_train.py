import tensorflow as tf


def mean_iou_metric(num_classes: int, ignore_background: bool = True):
    """
    Mean IoU für Multiclass-Segmentierung mit Integer-Masken.
    - y_true: (B, H, W) oder (B, H, W, 1), Werte in {0,...,num_classes-1}
    - y_pred: (B, H, W, C), Softmax-Output
    """
    def metric(y_true, y_pred):
        # y_true ggf. (B, H, W, 1) -> (B, H, W)
        if y_true.shape.rank == 4 and y_true.shape[-1] == 1:
            y_true_ = tf.squeeze(y_true, axis=-1)
        else:
            y_true_ = y_true

        # harte Vorhersage: Argmax über Klassen
        y_pred_labels = tf.argmax(y_pred, axis=-1)  # (B, H, W)

        # One-Hot
        y_true_oh = tf.one_hot(tf.cast(y_true_, tf.int32), depth=num_classes, dtype=tf.float32)
        y_pred_oh = tf.one_hot(tf.cast(y_pred_labels, tf.int32), depth=num_classes, dtype=tf.float32)

        # Intersection und Union pro Klasse
        axes = (0, 1, 2)  # B, H, W
        intersection = tf.reduce_sum(y_true_oh * y_pred_oh, axis=axes)           # (C,)
        true_sum    = tf.reduce_sum(y_true_oh, axis=axes)                        # (C,)
        pred_sum    = tf.reduce_sum(y_pred_oh, axis=axes)                        # (C,)
        union       = true_sum + pred_sum - intersection                         # (C,)

        eps = 1e-7
        iou_per_class = (intersection + eps) / (union + eps)                     # (C,)

        # ggf. Hintergrundklasse (0) ignorieren
        if ignore_background and num_classes > 1:
            iou_per_class = iou_per_class[1:]

        return tf.reduce_mean(iou_per_class)

    metric.__name__ = "mean_iou"
    return metric


def mean_f1_metric(num_classes: int, ignore_background: bool = True):
    """
    Mean F1-Score (Dice-ähnlich, aber mit Precision/Recall) über Klassen.
    - y_true: (B, H, W) oder (B, H, W, 1)
    - y_pred: (B, H, W, C), Softmax-Output
    """
    def metric(y_true, y_pred):
        # y_true ggf. (B, H, W, 1) -> (B, H, W)
        if y_true.shape.rank == 4 and y_true.shape[-1] == 1:
            y_true_ = tf.squeeze(y_true, axis=-1)
        else:
            y_true_ = y_true

        # harte Vorhersage
        y_pred_labels = tf.argmax(y_pred, axis=-1)

        # One-Hot
        y_true_oh = tf.one_hot(tf.cast(y_true_, tf.int32), depth=num_classes, dtype=tf.float32)
        y_pred_oh = tf.one_hot(tf.cast(y_pred_labels, tf.int32), depth=num_classes, dtype=tf.float32)

        axes = (0, 1, 2)  # B, H, W
        tp = tf.reduce_sum(y_true_oh * y_pred_oh, axis=axes)                     # (C,)
        true_sum = tf.reduce_sum(y_true_oh, axis=axes)                           # (C,)
        pred_sum = tf.reduce_sum(y_pred_oh, axis=axes)                           # (C,)

        eps = 1e-7
        precision = (tp + eps) / (pred_sum + eps)
        recall    = (tp + eps) / (true_sum + eps)
        f1_per_class = 2.0 * precision * recall / (precision + recall + eps)     # (C,)

        if ignore_background and num_classes > 1:
            f1_per_class = f1_per_class[1:]

        return tf.reduce_mean(f1_per_class)

    metric.__name__ = "mean_f1"
    return metric
