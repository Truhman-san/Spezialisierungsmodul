from __future__ import annotations

import argparse
from pathlib import Path
import tensorflow as tf

from src.ssl.schedules import linear_rampup
from src.ssl.datasets import build_unlabeled_ds
from src.ssl.consistency import weak_stm_augment, strong_stm_augment


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--unlabeled_dir", type=str, required=True)
    p.add_argument("--image_h", type=int, default=256)
    p.add_argument("--image_w", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--steps", type=int, default=2000)

    p.add_argument("--lr", type=float, default=2e-5)

    p.add_argument("--lambda_max", type=float, default=1.0)
    p.add_argument("--lambda_ramp_start", type=int, default=200)
    p.add_argument("--lambda_ramp_len", type=int, default=1200)

    # FixMatch-ish gate
    p.add_argument("--tau_start", type=float, default=0.85)
    p.add_argument("--tau_end", type=float, default=0.75)
    p.add_argument("--tau_ramp_steps", type=int, default=1200)

    p.add_argument("--ignore_bg", type=int, default=1)  # ignore class 0 in unlabeled loss
    p.add_argument("--debug_every", type=int, default=100)

    p.add_argument("--bg_w", type=float, default=0.2)          # weight für Background-Pixel im SSL-Loss
    p.add_argument("--target_fg", type=float, default=0.08)  
    p.add_argument("--beta_area", type=float, default=0.05)  

    return p.parse_args()


def _pick_main_output(out):
    """Robust: handle single output, tuple/list outputs, or dict outputs."""
    if isinstance(out, (list, tuple)):
        return out[0]
    if isinstance(out, dict):
        if "main" in out:
            return out["main"]
        return list(out.values())[0]
    return out


def _ensure_probs(x: tf.Tensor) -> tf.Tensor:
    """If output is not probs (sum over C not ~1), apply softmax."""
    x = tf.cast(x, tf.float32)
    s = tf.reduce_mean(tf.reduce_sum(x, axis=-1))
    return tf.cond(
        tf.logical_or(s < 0.90, s > 1.10),
        lambda: tf.nn.softmax(x, axis=-1),
        lambda: x
    )


def main():
    args = parse_args()
    image_size = (args.image_h, args.image_w)

    model = tf.keras.models.load_model(args.model_path, compile=False)
    opt = tf.keras.optimizers.Adam(args.lr, clipnorm=1.0)

    ds_u = build_unlabeled_ds(args.unlabeled_dir, image_size, args.batch_size)
    it_u = iter(ds_u.repeat())

    step_var = tf.Variable(0, dtype=tf.int32)

    @tf.function
    def train_step(x):
        step = step_var
        lam = linear_rampup(step, args.lambda_ramp_start, args.lambda_ramp_len, args.lambda_max)

        # tau schedule
        t = tf.cast(tf.minimum(step, args.tau_ramp_steps), tf.float32) / float(args.tau_ramp_steps)
        tau = tf.cast(args.tau_start, tf.float32) * (1.0 - t) + tf.cast(args.tau_end, tf.float32) * t

        # Weak/Strong views
        x_w = weak_stm_augment(x, seed=1)
        x_s = strong_stm_augment(x, seed=2)

        with tf.GradientTape() as tape:
            # teacher prediction on weak (no grad)
            p_w = _ensure_probs(_pick_main_output(model(x_w, training=False)))
            # student prediction on strong (grad)
            p_s = _ensure_probs(_pick_main_output(model(x_s, training=True)))

            # pseudo-label + confidence from weak
            conf = tf.reduce_max(p_w, axis=-1)  
            yhat = tf.argmax(p_w, axis=-1, output_type=tf.int32) 

            # gate: confident pixels
            mask = tf.cast(conf > tau, tf.float32)

            # Background bleibt drin, aber schwächer gewichtet
            mask = mask * tf.where(tf.equal(yhat, 0), tf.cast(args.bg_w, tf.float32), 1.0)


            # consistency via CE to pseudo-labels 
            ce = tf.keras.losses.sparse_categorical_crossentropy(
                yhat, p_s, from_logits=False
            ) 

            denom = tf.reduce_sum(mask) + 1e-6
            loss_u = tf.reduce_sum(ce * mask) / denom
            # Area regularizer: target foreground fraction (1 - p_bg)
            p_bg = p_s[..., 0]  
            mean_fg = tf.reduce_mean(1.0 - p_bg)
            loss_area = tf.square(mean_fg - tf.cast(args.target_fg, tf.float32))

            total = lam * loss_u + tf.cast(args.beta_area, tf.float32) * loss_area


        grads = tape.gradient(total, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))

        # ---- debug ----
        keep = tf.reduce_mean(mask)
        bg_ratio = tf.reduce_mean(tf.cast(tf.equal(yhat, 0), tf.float32))

        # confidence/entropy on student probs 
        mean_conf_s = tf.reduce_mean(tf.reduce_max(p_s, axis=-1))
        entropy_s = -tf.reduce_mean(
            tf.reduce_sum(p_s * tf.math.log(tf.clip_by_value(p_s, 1e-6, 1.0)), axis=-1)
        )

        return total, loss_u, lam, tau, keep, bg_ratio, mean_fg, mean_conf_s, entropy_s

    for _ in range(args.steps):
        x = next(it_u)
        total, loss_u, lam, tau, keep, bg_ratio, mean_fg, mean_conf_s, entropy_s = train_step(x)
        step_var.assign_add(1)

        if int(step_var.numpy()) % args.debug_every == 0:
            print(
                f"step={int(step_var.numpy())} "
                f"total={float(total.numpy()):.6f} Lu={float(loss_u.numpy()):.6f} "
                f"lam={float(lam.numpy()):.3f} tau={float(tau.numpy()):.3f} "
                f"keep={float(keep.numpy()):.4f} bg≈{float(bg_ratio.numpy()):.3f} "
                f"mean_conf_s={float(mean_conf_s.numpy()):.3f} entropy_s={float(entropy_s.numpy()):.4f}"
                f" mean_fg={float(mean_fg.numpy()):.4f}"
            )

    out_path = args.model_path.replace(".keras", "_ssl_consistency.keras")
    model.save(out_path)
    print("saved:", out_path)


if __name__ == "__main__":
    main()