from __future__ import annotations
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_ROOT = Path("runs")

RUNS = {
    0.10: "lc_f10",
    0.25: "lc_f25",
    0.50: "lc_f50",
    0.75: "lc_f75",
    1.00: "lc_f100",
}


def read_approx_train_samples(stats_path: Path):
    with stats_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("approx_train_samples:"):
                return int(line.split(":")[1].strip())
    raise RuntimeError(f"Could not find approx_train_samples in {stats_path}")


def pick_metrics(test_metrics):
    iou = test_metrics.get("mean_iou_metric") or test_metrics.get("mean_iou")
    f1 = test_metrics.get("mean_f1_metric") or test_metrics.get("mean_f1")
    return iou, f1


def main():
    results = []

    for frac, run_name in RUNS.items():
        out_dir = OUTPUT_ROOT / run_name

        stats_path = out_dir / "dataset_stats.txt"
        test_path  = out_dir / "test_metrics.json"

        approx_train = read_approx_train_samples(stats_path)

        with test_path.open("r", encoding="utf-8") as f:
            test_metrics = json.load(f)

        mean_iou, mean_f1 = pick_metrics(test_metrics)

        results.append({
            "frac": frac,
            "train_samples": approx_train,
            "mean_iou": mean_iou,
            "mean_f1": mean_f1,
        })

    # Sortieren nach Anzahl Samples
    results.sort(key=lambda x: x["train_samples"])

    xs = [r["train_samples"] for r in results]
    ys_iou = [r["mean_iou"] for r in results]
    ys_f1  = [r["mean_f1"] for r in results]

    plt.figure()
    plt.plot(xs, ys_iou, marker="o")
    plt.xlabel("Anzahl Trainingssamples")
    plt.ylabel("Test Mean IoU")
    plt.title("Learning Curve (IoU)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_ROOT / "learning_curve_iou.png")
    plt.close()

    plt.figure()
    plt.plot(xs, ys_f1, marker="o")
    plt.xlabel("Anzahl Trainingssamples")
    plt.ylabel("Test Mean F1")
    plt.title("Learning Curve (F1)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_ROOT / "learning_curve_f1.png")
    plt.close()

    print("Plots gespeichert unter:", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
