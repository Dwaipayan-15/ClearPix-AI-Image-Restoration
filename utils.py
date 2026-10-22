"""Shared helpers: noise generation, metrics, image I/O and inference."""
from __future__ import annotations

from pathlib import Path

import numpy as np

NOISE_TYPES = ("gaussian", "salt_pepper", "speckle")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


# --------------------------------------------------------------------------- noise
def add_noise(images: np.ndarray, kind: str = "gaussian", level: float = 0.1,
              seed: int | None = None) -> np.ndarray:
    """Corrupt float images in [0, 1]. Works on a single image (H,W,C) or a batch (N,H,W,C).

    `level` is the Gaussian/speckle standard deviation, or the fraction of
    corrupted pixels for salt-and-pepper noise.
    """
    if kind not in NOISE_TYPES:
        raise ValueError(f"Unknown noise type '{kind}'. Choose from {NOISE_TYPES}.")
    rng = np.random.default_rng(seed)
    images = images.astype(np.float32)
    if kind == "gaussian":
        out = images + rng.normal(0.0, level, images.shape)
    elif kind == "speckle":
        out = images + images * rng.normal(0.0, level, images.shape)
    else:  # salt_pepper
        out = images.copy()
        mask = rng.random(images.shape[:-1])
        out[mask < level / 2] = 0.0
        out[mask > 1.0 - level / 2] = 1.0
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ------------------------------------------------------------------------- metrics
def psnr(reference: np.ndarray, test: np.ndarray) -> float:
    """Mean PSNR (dB) between float images in [0, 1]."""
    import tensorflow as tf
    ref, tst = _as_batch(reference), _as_batch(test)
    return float(tf.reduce_mean(tf.image.psnr(ref, tst, max_val=1.0)).numpy())


def ssim(reference: np.ndarray, test: np.ndarray) -> float:
    """Mean SSIM between float images in [0, 1]. Images must be at least 11x11."""
    import tensorflow as tf
    ref, tst = _as_batch(reference), _as_batch(test)
    return float(tf.reduce_mean(tf.image.ssim(ref, tst, max_val=1.0)).numpy())


def _as_batch(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x[None] if x.ndim == 3 else x


# ----------------------------------------------------------------- classical filters
def classical_denoise(image: np.ndarray, method: str = "nlm", strength: float = 0.1) -> np.ndarray:
    """Traditional baselines used for comparison: 'nlm' (non-local means) or 'gaussian'."""
    import cv2
    img8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    if method == "nlm":
        h = float(np.clip(strength * 255.0 * 0.6, 3.0, 30.0))
        out = cv2.fastNlMeansDenoisingColored(img8, None, h, h, 7, 21)
    elif method == "gaussian":
        out = cv2.GaussianBlur(img8, (5, 5), 0)
    else:
        raise ValueError("method must be 'nlm' or 'gaussian'")
    return out.astype(np.float32) / 255.0


# ----------------------------------------------------------------------------- I/O
def load_image(source, max_side: int | None = None) -> np.ndarray:
    """Load an image path / file object as float32 RGB in [0, 1], optionally downscaled."""
    from PIL import Image, ImageOps
    img = ImageOps.exif_transpose(Image.open(source)).convert("RGB")
    if max_side and max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))),
                         Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def to_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(image * 255.0 + 0.5, 0, 255).astype(np.uint8)


def load_folder_patches(folder: str, patch_size: int = 64, patches_per_image: int = 8,
                        max_patches: int = 20000, seed: int = 42) -> np.ndarray:
    """Cut random square patches from every image in a folder (recursively)."""
    rng = np.random.default_rng(seed)
    files = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise FileNotFoundError(f"No images found in '{folder}'.")
    patches = []
    for path in files:
        img = load_image(path)
        h, w = img.shape[:2]
        if min(h, w) < patch_size:
            continue
        for _ in range(patches_per_image):
            y, x = rng.integers(0, h - patch_size + 1), rng.integers(0, w - patch_size + 1)
            patches.append(img[y:y + patch_size, x:x + patch_size])
        if len(patches) >= max_patches:
            break
    if not patches:
        raise ValueError(f"No image in '{folder}' is at least {patch_size}px on both sides.")
    return np.stack(patches[:max_patches]).astype(np.float32)


# ------------------------------------------------------------------------- inference
def restore_image(model, image: np.ndarray, multiple: int = 4) -> np.ndarray:
    """Run the autoencoder on one float RGB image of any size."""
    h, w = image.shape[:2]
    pad_h, pad_w = (-h) % multiple, (-w) % multiple
    padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect") if (pad_h or pad_w) else image
    # Call the model directly (eager). model.predict() retraces with relaxed, unknown
    # spatial dims when the image size changes, which breaks bilinear UpSampling2D.
    out = np.asarray(model(padded[None].astype(np.float32), training=False))[0]
    return np.clip(out[:h, :w], 0.0, 1.0)