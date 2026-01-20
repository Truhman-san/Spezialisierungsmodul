# src/ssl/pseudo_label.py
from __future__ import annotations
import tensorflow as tf

def pseudo_labels_from_probs(probs: tf.Tensor):
    """
    probs: (B,H,W,C) softmax probabilities
    returns:
      yhat: (B,H,W) int32
      conf: (B,H,W) float32 in [0,1]
    """
    conf = tf.reduce_max(probs, axis=-1)
    yhat = tf.argmax(probs, axis=-1, output_type=tf.int32)
    return yhat, conf

def masked_sparse_ce_probs(student_probs: tf.Tensor, yhat: tf.Tensor, mask: tf.Tensor):
    """
    student_probs: (B,H,W,C) probabilities (softmax output)
    yhat:          (B,H,W) int
    mask:          (B,H,W) float in [0,1]
    """
    ce = tf.keras.losses.sparse_categorical_crossentropy(
        yhat, student_probs, from_logits=False
    )  # (B,H,W)
    ce = ce * mask
    denom = tf.reduce_sum(mask) + 1e-6
    return tf.reduce_sum(ce) / denom
