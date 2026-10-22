"""Train the ClearPix denoising autoencoder.

Examples
--------
    python train.py                                   # CIFAR-10, 30 epochs
    python train.py --data-dir ./my_images --patch-size 64 --epochs 50
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from model import build_autoencoder
from utils import load_folder_patches

MODELS_DIR, RESULTS_DIR = Path("models"), Path("results")


def psnr_metric(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)


def ssim_metric(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=1.0)


def load_data(args):
    if args.data_dir:
        data = load_folder_patches(args.data_dir, args.patch_size, max_patches=args.max_images)
    else:
        (x_train, _), _ = keras.datasets.cifar10.load_data()
        data = x_train.astype("float32") / 255.0
    data = data[: args.max_images]
    np.random.default_rng(args.seed).shuffle(data)
    n_val = max(1, int(len(data) * args.val_split))
    return data[n_val:], data[:n_val]


def make_dataset(clean: np.ndarray, args, training: bool) -> tf.data.Dataset:
    """Pairs of (noisy, clean) images; noise is regenerated every epoch."""
    def corrupt(img):
        sigma = tf.random.uniform([], args.sigma_min, args.sigma_max)
        noisy = img + tf.random.normal(tf.shape(img), stddev=sigma)
        return tf.clip_by_value(noisy, 0.0, 1.0), img

    ds = tf.data.Dataset.from_tensor_slices(clean)
    if training:
        ds = ds.shuffle(min(len(clean), 10000), seed=args.seed)
    return ds.map(corrupt, num_parallel_calls=tf.data.AUTOTUNE).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)


def plot_history(history: dict, path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, key, title in zip(axes, ("loss", "psnr_metric", "ssim_metric"),
                              ("Reconstruction loss (MSE)", "PSNR (dB)", "SSIM")):
        ax.plot(history[key], label="train", color="#0E7C86", lw=2)
        ax.plot(history[f"val_{key}"], label="validation", color="#C2410C", lw=2)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=None, help="Folder of images. Default: CIFAR-10 (auto-download).")
    p.add_argument("--patch-size", type=int, default=64)
    p.add_argument("--max-images", type=int, default=50000)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--base-filters", type=int, default=32)
    p.add_argument("--no-skip", action="store_true", help="Disable skip connections (plain autoencoder).")
    p.add_argument("--sigma-min", type=float, default=0.05, help="Min Gaussian noise std during training.")
    p.add_argument("--sigma-max", type=float, default=0.25, help="Max Gaussian noise std during training.")
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=str(MODELS_DIR / "clearpix_autoencoder.keras"))
    args = p.parse_args()

    keras.utils.set_random_seed(args.seed)
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    train_clean, val_clean = load_data(args)
    print(f"Training images: {len(train_clean)} | Validation images: {len(val_clean)} "
          f"| Shape: {train_clean.shape[1:]}")
    train_ds = make_dataset(train_clean, args, training=True)
    val_ds = make_dataset(val_clean, args, training=False)

    model = build_autoencoder(base_filters=args.base_filters, skip_connections=not args.no_skip)
    model.compile(optimizer=keras.optimizers.Adam(args.lr), loss="mse",
                  metrics=[psnr_metric, ssim_metric])
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(args.output, monitor="val_loss", save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1),
        keras.callbacks.CSVLogger(str(RESULTS_DIR / "training_log.csv")),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    plot_history(history.history, RESULTS_DIR / "training_curves.png")
    (RESULTS_DIR / "train_config.json").write_text(json.dumps(vars(args), indent=2))
    print(f"\nBest model saved to {args.output}")
    print(f"Training curves saved to {RESULTS_DIR / 'training_curves.png'}")


if __name__ == "__main__":
    main()
