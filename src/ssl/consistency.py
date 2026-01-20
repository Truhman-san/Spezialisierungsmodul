from __future__ import annotations
import tensorflow as tf

def kl_divergence(p_teacher: tf.Tensor, p_student: tf.Tensor, eps: float = 1e-6):
    """
    p_teacher, p_student: (B,H,W,C) probabilities, sum over C = 1
    returns mean KL per pixel averaged over batch
    """
    p_teacher = tf.clip_by_value(p_teacher, eps, 1.0)
    p_student = tf.clip_by_value(p_student, eps, 1.0)
    kl = tf.reduce_sum(p_teacher * (tf.math.log(p_teacher) - tf.math.log(p_student)), axis=-1)  # (B,H,W)
    return tf.reduce_mean(kl)

def simple_stm_augment(x: tf.Tensor, seed: int | None = None):
    """
    Plausible STM-ish augmentations:
    - small brightness/contrast jitter
    - gaussian noise
    - random flips
    - small translation
    Input x: (B,H,W,1) float [0,1]
    """
    if seed is None:
        seed = 0

    # flips
    x = tf.image.random_flip_left_right(x, seed=seed)
    x = tf.image.random_flip_up_down(x, seed=seed + 1)

    # brightness/contrast
    x = tf.image.random_brightness(x, max_delta=0.08, seed=seed + 2)
    x = tf.image.random_contrast(x, lower=0.9, upper=1.1, seed=seed + 3)

    # gaussian noise
    noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=0.03, seed=seed + 4)
    x = x + noise

    # small translation via padding + random crop
    # (works well when input size is fixed, like yours)
    pad = 8
    x_pad = tf.pad(x, paddings=[[0,0],[pad,pad],[pad,pad],[0,0]], mode="REFLECT")
    x = tf.image.random_crop(x_pad, size=tf.shape(x), seed=seed + 5)

    # keep range sane
    x = tf.clip_by_value(x, 0.0, 1.0)
    return x

def weak_stm_augment(x: tf.Tensor, seed: int = 0):
    x = tf.image.random_flip_left_right(x, seed=seed)
    x = tf.image.random_flip_up_down(x, seed=seed+1)
    x = tf.clip_by_value(x, 0.0, 1.0)
    return x

def strong_stm_augment(x: tf.Tensor, seed: int = 0):
    x = tf.image.random_flip_left_right(x, seed=seed)
    x = tf.image.random_flip_up_down(x, seed=seed+1)
    x = tf.image.random_brightness(x, max_delta=0.10, seed=seed+2)
    x = tf.image.random_contrast(x, lower=0.90, upper=1.10, seed=seed+3)
    noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=0.02, seed=seed+4)
    x = x + noise
    pad = 8
    x_pad = tf.pad(x, [[0,0],[pad,pad],[pad,pad],[0,0]], mode="REFLECT")
    x = tf.image.random_crop(x_pad, size=tf.shape(x), seed=seed+5)
    x = tf.clip_by_value(x, 0.0, 1.0)
    return x
