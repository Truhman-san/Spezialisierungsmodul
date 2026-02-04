import tensorflow as tf

def sparse_ce(from_logits: bool = False):
    return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits)

def multiclass_dice_loss(y_true, y_pred, num_classes: int, from_logits: bool = False, smooth: float = 1.0):
    if y_true.shape.rank == 4 and y_true.shape[-1] == 1:
        y_true = tf.squeeze(y_true, axis=-1)
    if from_logits:
        y_pred = tf.nn.softmax(y_pred, axis=-1)
    y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes, dtype=tf.float32)
    inter = tf.reduce_sum(y_true * y_pred, axis=(0, 1, 2))
    denom = tf.reduce_sum(y_true, axis=(0, 1, 2)) + tf.reduce_sum(y_pred, axis=(0, 1, 2))
    dice = (2.0 * inter + smooth) / (denom + smooth)
    return 1.0 - tf.reduce_mean(dice)

def ce_plus_dice(num_classes: int, ce_w: float = 0.5, dice_w: float = 0.5, from_logits: bool = False):
    """Kompakte Kombi: Sparse CE + Dice."""
    ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits)
    def loss(y_true, y_pred):
        return ce_w * ce(y_true, y_pred) + dice_w * multiclass_dice_loss(
            y_true, y_pred, num_classes=num_classes, from_logits=from_logits
        )
    return loss

# ----------------------------------------------------------------------
# Sparse Categorical Focal Loss + Multiclass Dice
# ----------------------------------------------------------------------

def sparse_focal_loss(
    num_classes: int,
    gamma: float = 2.0,
    alpha: float | None = None,
    from_logits: bool = False,
):
    """
    Sparse Categorical Focal Loss für Multiclass-Segmentation.
    - y_true: Integer-Masken (B, H, W) oder (B, H, W, 1)
    - y_pred: Logits oder Softmax-Outputs (B, H, W, C)
    """
    def loss(y_true, y_pred):
        if y_true.shape.rank == 4 and y_true.shape[-1] == 1:
            y_true_ = tf.squeeze(y_true, axis=-1)
        else:
            y_true_ = y_true

        # Softmax, falls Logits
        if from_logits:
            y_pred_ = tf.nn.softmax(y_pred, axis=-1)
        else:
            y_pred_ = y_pred

        # One-Hot für einfaches p_t-Handling
        y_true_oh = tf.one_hot(tf.cast(y_true_, tf.int32), depth=num_classes, dtype=tf.float32)

        # p_t = Wahrscheinlichkeit der korrekten Klasse pro Pixel
        p_t = tf.reduce_sum(y_true_oh * y_pred_, axis=-1)  # (B, H, W)

        # Numerische Stabilität
        eps = 1e-7
        p_t = tf.clip_by_value(p_t, eps, 1.0 - eps)

        # Standard Cross-Entropy-Komponente
        ce = -tf.math.log(p_t)  # (B, H, W)

        # Focal-Multiplikator (1 - p_t)^gamma
        focal_factor = tf.pow(1.0 - p_t, gamma)

        loss_val = focal_factor * ce

        # Optional: Alpha-Gewichtung (einfacher globaler Faktor)
        if alpha is not None:
            loss_val = alpha * loss_val

        # Mittel über alle Pixel und Batch
        return tf.reduce_mean(loss_val)

    return loss


def focal_plus_dice(
    num_classes: int,
    focal_w: float = 0.5,
    dice_w: float = 0.5,
    gamma: float = 2.0,
    alpha: float | None = None,
    from_logits: bool = False,
):
    """
    Kombi-Loss: Sparse Categorical Focal Loss + Multiclass Dice Loss.

    - focal_w, dice_w: Gewichte der beiden Loss-Komponenten.
    - gamma: Focal-Parameter (üblich: 1.0–2.0)
    - alpha: optionaler globaler Faktor für positives Label (z. B. 0.25)
    """
    focal = sparse_focal_loss(
        num_classes=num_classes,
        gamma=gamma,
        alpha=alpha,
        from_logits=from_logits,
    )

    def loss(y_true, y_pred):
        lf = focal(y_true, y_pred)
        ld = multiclass_dice_loss(
            y_true,
            y_pred,
            num_classes=num_classes,
            from_logits=from_logits,
        )
        return focal_w * lf + dice_w * ld

    return loss


def masked_focal_plus_dice(
    num_classes: int,
    ignore_index: int = 255,
    focal_w: float = 0.5,
    dice_w: float = 0.5,
    gamma: float = 2.0,
    alpha: float | None = None,
    from_logits: bool = False,
):
    eps = 1e-7

    def loss(y_true, y_pred):
        if y_true.shape.rank == 4 and y_true.shape[-1] == 1:
            y_true = tf.squeeze(y_true, axis=-1)  

        y_true = tf.cast(y_true, tf.int32)
        valid = tf.not_equal(y_true, ignore_index)            
        valid_f = tf.cast(valid, tf.float32)

        y_true_safe = tf.where(valid, y_true, 0)

        # Softmax falls logits
        if from_logits:
            y_prob = tf.nn.softmax(y_pred, axis=-1)
        else:
            y_prob = y_pred

        # === FOCAL (sparse) ===
        y_true_oh = tf.one_hot(y_true_safe, depth=num_classes, dtype=tf.float32) 
        p_t = tf.reduce_sum(y_true_oh * y_prob, axis=-1)                       
        p_t = tf.clip_by_value(p_t, eps, 1.0 - eps)

        ce = -tf.math.log(p_t)
        focal_factor = tf.pow(1.0 - p_t, gamma)
        focal = focal_factor * ce
        if alpha is not None:
            focal = alpha * focal

        # mask anwenden + normalisieren
        focal = focal * valid_f
        focal = tf.reduce_sum(focal) / (tf.reduce_sum(valid_f) + eps)

        # === DICE (multiclass) ===
        # nur gültige Pixel zählen
        inter = tf.reduce_sum(y_true_oh * y_prob * valid_f[..., None], axis=(0,1,2)) 
        true_sum = tf.reduce_sum(y_true_oh * valid_f[..., None], axis=(0,1,2))
        pred_sum = tf.reduce_sum(y_prob * valid_f[..., None], axis=(0,1,2))
        dice = (2.0 * inter + 1.0) / (true_sum + pred_sum + 1.0)
        dice_loss = 1.0 - tf.reduce_mean(dice)

        return focal_w * focal + dice_w * dice_loss

    return loss