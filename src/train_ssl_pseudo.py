# train_ssl_pseudo.py
from __future__ import annotations
import argparse
from pathlib import Path
import tensorflow as tf

from src.ssl.ema import EMATeacher
from src.ssl.pseudo_label import pseudo_labels_from_probs, masked_sparse_ce_probs
from src.ssl.schedules import linear_rampup
from src.ssl.datasets import build_unlabeled_ds


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--unlabeled_dir", type=str, required=True)
    p.add_argument("--image_h", type=int, default=256)
    p.add_argument("--image_w", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--steps", type=int, default=2000)

    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--ema_decay", type=float, default=0.999)

    p.add_argument("--tau_start", type=float, default=0.90)
    p.add_argument("--tau_end", type=float, default=0.80)
    p.add_argument("--tau_ramp_steps", type=int, default=1500)

    p.add_argument("--lambda_max", type=float, default=1.0)
    p.add_argument("--lambda_ramp_start", type=int, default=200)
    p.add_argument("--lambda_ramp_len", type=int, default=1200)

    # SSL-spezifisch
    p.add_argument("--ignore_bg", type=int, default=1, help="1: Background (class 0) im unlabeled loss ignorieren")
    p.add_argument("--mask_floor", type=float, default=0.0, help="Optional: clamp mask to <= mask_floor if needed (debug)")
    p.add_argument("--debug_every", type=int, default=100)

    return p.parse_args()


def _pick_main_output(out):
    """
    Robust: handle single output, tuple/list outputs, or dict outputs.
    Returns the first / 'main' output tensor that is expected to be (B,H,W,C).
    """
    if isinstance(out, (list, tuple)):
        return out[0]
    if isinstance(out, dict):
        # prefer "main" if exists, else first value
        if "main" in out:
            return out["main"]
        return list(out.values())[0]
    return out


def main():
    args = parse_args()
    image_size = (args.image_h, args.image_w)

    student = tf.keras.models.load_model(args.model_path, compile=False)
    teacher_mgr = EMATeacher(student, decay=args.ema_decay)

    # stabiler bei wenig Daten:
    opt = tf.keras.optimizers.Adam(args.lr, clipnorm=1.0)

    ds_u = build_unlabeled_ds(args.unlabeled_dir, image_size, args.batch_size)
    it_u = iter(ds_u.repeat())

    step_var = tf.Variable(0, dtype=tf.int32)

    @tf.function
    def train_step(x_u):
        step = step_var

        lam = linear_rampup(step, args.lambda_ramp_start, args.lambda_ramp_len, args.lambda_max)

        # tau schedule
        t = tf.cast(tf.minimum(step, args.tau_ramp_steps), tf.float32) / float(args.tau_ramp_steps)
        tau = tf.cast(args.tau_start, tf.float32) * (1.0 - t) + tf.cast(args.tau_end, tf.float32) * t

        with tf.GradientTape() as tape:
            # ---- teacher forward ----
            t_out = teacher_mgr.teacher(x_u, training=False)
            t_probs = _pick_main_output(t_out)
            t_probs = tf.cast(t_probs, tf.float32)

            # sanity: if the model outputs logits by accident, make it probs
            # (shouldn't happen in your setup, but keeps it robust)
            # If sums aren't ~1, apply softmax:
            sum_probs = tf.reduce_mean(tf.reduce_sum(t_probs, axis=-1))
            t_probs = tf.cond(
                tf.logical_or(sum_probs < 0.90, sum_probs > 1.10),
                lambda: tf.nn.softmax(t_probs, axis=-1),
                lambda: t_probs
            )

            yhat, conf = pseudo_labels_from_probs(t_probs)

            # soft mask
            mask = tf.clip_by_value((conf - tau) / (1.0 - tau + 1e-6), 0.0, 1.0)

            # ignore background in unlabeled loss (recommended in your setting)
            if args.ignore_bg == 1:
                mask = mask * tf.cast(tf.not_equal(yhat, 0), tf.float32)

            # optional clamp (normally leave at 0)
            if args.mask_floor > 0:
                mask = tf.minimum(mask, tf.cast(args.mask_floor, tf.float32))

            # ---- student forward ----
            s_out = student(x_u, training=True)
            s_probs = _pick_main_output(s_out)
            s_probs = tf.cast(s_probs, tf.float32)

            sum_probs_s = tf.reduce_mean(tf.reduce_sum(s_probs, axis=-1))
            s_probs = tf.cond(
                tf.logical_or(sum_probs_s < 0.90, sum_probs_s > 1.10),
                lambda: tf.nn.softmax(s_probs, axis=-1),
                lambda: s_probs
            )

            loss_u = masked_sparse_ce_probs(s_probs, yhat, mask)
            total = lam * loss_u

        grads = tape.gradient(total, student.trainable_variables)
        opt.apply_gradients(zip(grads, student.trainable_variables))
        teacher_mgr.update(student)

        # ===== DEBUG =====
        conf_min = tf.reduce_min(conf)
        conf_mean = tf.reduce_mean(conf)
        conf_max = tf.reduce_max(conf)

        keep_hard = tf.reduce_mean(tf.cast(conf > tau, tf.float32))
        keep_soft = tf.reduce_mean(mask)

        # sample label stats (cheap)
        y0 = yhat[0]
        y0_small = y0[::8, ::8]
        uniq = tf.unique(tf.reshape(y0_small, [-1]))[0]
        uniq_count = tf.cast(tf.size(uniq), tf.float32)
        bg_ratio = tf.reduce_mean(tf.cast(tf.equal(y0_small, 0), tf.float32))

        tp_min = tf.reduce_min(t_probs)
        tp_mean = tf.reduce_mean(t_probs)
        tp_max = tf.reduce_max(t_probs)
        sum_probs_dbg = tf.reduce_mean(tf.reduce_sum(t_probs, axis=-1))

        return (total, loss_u, lam, tau,
                keep_soft, keep_hard,
                conf_min, conf_mean, conf_max,
                uniq_count, bg_ratio,
                tp_min, tp_mean, tp_max, sum_probs_dbg)

    for _ in range(args.steps):
        x_u = next(it_u)
        (total, loss_u, lam, tau,
         keep_soft, keep_hard,
         conf_min, conf_mean, conf_max,
         uniq_count, bg_ratio,
         tp_min, tp_mean, tp_max, sum_probs_dbg) = train_step(x_u)

        step_var.assign_add(1)

        if int(step_var.numpy()) % args.debug_every == 0:
            print(
                f"step={int(step_var.numpy())} "
                f"total={float(total.numpy()):.6f} Lu={float(loss_u.numpy()):.6f} "
                f"lam={float(lam.numpy()):.3f} tau={float(tau.numpy()):.3f} "
                f"keep_soft={float(keep_soft.numpy()):.3f} keep_hard={float(keep_hard.numpy()):.3f} "
                f"conf[min/mean/max]={float(conf_min.numpy()):.3f}/{float(conf_mean.numpy()):.3f}/{float(conf_max.numpy()):.3f} "
                f"uniq≈{float(uniq_count.numpy()):.0f} bg≈{float(bg_ratio.numpy()):.3f} "
                f"t_probs[min/mean/max]={float(tp_min.numpy()):.3f}/{float(tp_mean.numpy()):.3f}/{float(tp_max.numpy()):.3f} "
                f"sum_probs≈{float(sum_probs_dbg.numpy()):.3f}"
            )

    out_path = args.model_path.replace(".keras", "_ssl_pseudo.keras")
    student.save(out_path)
    print("saved:", out_path)


if __name__ == "__main__":
    main()
