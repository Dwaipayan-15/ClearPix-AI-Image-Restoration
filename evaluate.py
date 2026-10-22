"""Evaluate a trained ClearPix model against the noisy input and classical filters.

Writes metrics (CSV/JSON), a comparison chart and a sample grid to ./results.

    python evaluate.py
    python evaluate.py --data-dir ./test_images --num-images 200
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras

from utils import add_noise, classical_denoise, load_folder_patches, psnr, ssim

RESULTS_DIR = Path("results")
TEAL, SLATE, AMBER, GREY = "#0E7C86", "#172033", "#C2410C", "#9AA3B2"


def load_test_images(args) -> np.ndarray:
    if args.data_dir:
        data = load_folder_patches(args.data_dir, args.patch_size, max_patches=args.num_images, seed=7)
    else:
        _, (x_test, _) = keras.datasets.cifar10.load_data()
        data = x_test.astype("float32") / 255.0
    return data[: args.num_images]


def evaluate(model, clean: np.ndarray, sigma: float, noise: str, seed: int) -> dict:
    noisy = add_noise(clean, noise, sigma, seed=seed)
    restored = np.clip(model.predict(noisy, batch_size=128, verbose=0), 0, 1)
    nlm = np.stack([classical_denoise(x, "nlm", sigma) for x in noisy])
    blur = np.stack([classical_denoise(x, "gaussian") for x in noisy])
    row = {"noise_level": sigma}
    for name, out in (("noisy", noisy), ("gaussian_blur", blur), ("non_local_means", nlm), ("clearpix", restored)):
        row[f"{name}_psnr"] = psnr(clean, out)
        row[f"{name}_ssim"] = ssim(clean, out)
    return row, noisy, restored


def plot_metrics(df: pd.DataFrame, path: Path):
    methods = [("noisy", "Noisy input", GREY), ("gaussian_blur", "Gaussian blur", SLATE),
               ("non_local_means", "Non-local means", AMBER), ("clearpix", "ClearPix", TEAL)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, label in zip(axes, ("psnr", "ssim"), ("PSNR (dB)", "SSIM")):
        for key, name, color in methods:
            ax.plot(df["noise_level"], df[f"{key}_{metric}"], marker="o", lw=2.2, label=name, color=color)
        ax.set_xlabel("Gaussian noise level (σ)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_samples(clean, noisy, restored, sigma, path: Path, n: int = 6):
    fig, axes = plt.subplots(3, n, figsize=(2 * n, 6.4))
    for i in range(n):
        for row, (img, title) in enumerate(((noisy[i], "Noisy"), (restored[i], "Restored"), (clean[i], "Clean"))):
            axes[row, i].imshow(img, interpolation="nearest")
            axes[row, i].axis("off")
            if i == 0:
                axes[row, i].set_title(title, loc="left", fontsize=11)
    fig.suptitle(f"Qualitative results (σ = {sigma})", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="models/clearpix_autoencoder.keras")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--patch-size", type=int, default=64)
    p.add_argument("--num-images", type=int, default=500)
    p.add_argument("--sigmas", type=float, nargs="+", default=[0.05, 0.10, 0.15, 0.20, 0.25])
    p.add_argument("--noise", default="gaussian", choices=["gaussian", "salt_pepper", "speckle"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    model = keras.models.load_model(args.model, compile=False)
    clean = load_test_images(args)
    print(f"Evaluating on {len(clean)} images, noise type: {args.noise}")

    rows, preview = [], None
    for sigma in args.sigmas:
        row, noisy, restored = evaluate(model, clean, sigma, args.noise, args.seed)
        rows.append(row)
        if sigma == args.sigmas[len(args.sigmas) // 2]:
            preview = (noisy, restored, sigma)
        print(f"σ={sigma:.2f} | noisy {row['noisy_psnr']:.2f} dB -> ClearPix {row['clearpix_psnr']:.2f} dB "
              f"| SSIM {row['noisy_ssim']:.3f} -> {row['clearpix_ssim']:.3f}")

    df = pd.DataFrame(rows).round(4)
    df.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(
        {"noise_type": args.noise, "num_images": len(clean), "results": df.to_dict("records")}, indent=2))
    plot_metrics(df, RESULTS_DIR / "metrics_comparison.png")
    plot_samples(clean, preview[0], preview[1], preview[2], RESULTS_DIR / "sample_results.png")
    print(f"\nSaved metrics and figures to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
