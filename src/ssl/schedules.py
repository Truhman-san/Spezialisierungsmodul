from __future__ import annotations
import tensorflow as tf

@tf.function
def linear_rampup(step: tf.Tensor, start: int, length: int, max_value: float):
    step_f = tf.cast(step, tf.float32)
    start_f = tf.cast(start, tf.float32)
    length_f = tf.cast(length, tf.float32)
    t = (step_f - start_f) / (length_f + 1e-6)
    t = tf.clip_by_value(t, 0.0, 1.0)
    return tf.cast(max_value, tf.float32) * t
