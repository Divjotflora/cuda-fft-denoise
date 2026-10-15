"""Plots the CSV results written by fft_denoise.

Usage: python3 scripts/plot_results.py [output_dir]

Produces:
  <output_dir>/example_signals.png  before/after for a few signals
  <output_dir>/timing.png           CPU vs GPU time across batch sizes
"""

import csv
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def plot_examples(out_dir):
    path = os.path.join(out_dir, "examples.csv")
    if not os.path.exists(path):
        print(f"{path} not found; skipping example plot.")
        return
    signals = defaultdict(lambda: defaultdict(list))
    for row in read_csv(path):
        s = signals[int(row["signal_id"])]
        for key in ("clean", "noisy", "gpu_denoised"):
            s[key].append(float(row[key]))

    ids = sorted(signals)
    fig, axes = plt.subplots(len(ids), 1, figsize=(10, 2.6 * len(ids)),
                             sharex=True)
    if len(ids) == 1:
        axes = [axes]
    for ax, sid in zip(axes, ids):
        s = signals[sid]
        ax.plot(s["noisy"], color="#bbbbbb", lw=0.7, label="noisy input")
        ax.plot(s["clean"], color="#1f77b4", lw=1.6, label="clean (truth)")
        ax.plot(s["gpu_denoised"], color="#d62728", lw=1.0, ls="--",
                label="GPU denoised")
        ax.set_ylabel(f"signal {sid}")
    axes[0].legend(loc="upper right", ncol=3, fontsize=8)
    axes[-1].set_xlabel("sample")
    fig.suptitle("Batched cuFFT low-pass denoising")
    fig.tight_layout()
    dest = os.path.join(out_dir, "example_signals.png")
    fig.savefig(dest, dpi=120)
    print(f"Saved {dest}")


def plot_timing(out_dir):
    path = os.path.join(out_dir, "summary.csv")
    if not os.path.exists(path):
        print(f"{path} not found; skipping timing plot.")
        return
    best = {}
    for row in read_csv(path):
        best[int(row["num_signals"])] = row  # Last run per size wins.
    sizes = sorted(best)
    cpu = [float(best[n]["cpu_ms"]) for n in sizes]
    gpu = [float(best[n]["gpu_total_ms"]) for n in sizes]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sizes, cpu, "o-", label="CPU (radix-2 FFT)")
    ax.plot(sizes, gpu, "s-", label="GPU (batched cuFFT, incl. copies)")
    for n, c, g in zip(sizes, cpu, gpu):
        ax.annotate(f"{c / g:.0f}x", (n, g), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("number of signals (1024 samples each)")
    ax.set_ylabel("time (ms)")
    ax.set_title("Denoising time vs batch size")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    dest = os.path.join(out_dir, "timing.png")
    fig.savefig(dest, dpi=120)
    print(f"Saved {dest}")


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "output"
    plot_examples(out_dir)
    plot_timing(out_dir)


if __name__ == "__main__":
    main()
