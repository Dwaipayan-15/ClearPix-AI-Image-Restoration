"""ClearPix — Streamlit web app for AI image restoration.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from utils import (NOISE_TYPES, add_noise, classical_denoise, load_image, psnr, restore_image,
                   ssim, to_uint8)

MODEL_PATH = Path("models/clearpix_autoencoder.keras")
MAX_SIDE = 512
NOISE_LABELS = {"gaussian": "Gaussian", "salt_pepper": "Salt & pepper", "speckle": "Speckle"}

st.set_page_config(page_title="ClearPix — AI Image Restoration", page_icon="◐", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Newsreader:opsz,wght@6..72,500;6..72,600&display=swap');
html, body, [class*="css"], .stMarkdown, button, input { font-family: 'IBM Plex Sans', sans-serif; }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { padding-top: 2.4rem; max-width: 1180px; }
h1.cp-title { font-family: 'Newsreader', serif; font-weight: 600; font-size: 3rem; letter-spacing: -0.02em;
              margin: 0; color: #172033; line-height: 1.05; }
.cp-sub { color: #5B6577; font-size: 1.05rem; margin: .5rem 0 1.6rem; max-width: 46rem; line-height: 1.55; }
.cp-rule { border: 0; border-top: 1px solid #D9DEE5; margin: 0 0 1.6rem; }
.cp-label { font-family: 'Newsreader', serif; font-size: 1.25rem; font-weight: 600; color: #172033; margin-bottom: .35rem; }
.cp-caption { color: #5B6577; font-size: .85rem; margin-top: .3rem; }
.cp-metric { background: #fff; border: 1px solid #D9DEE5; border-radius: 6px; padding: .9rem 1.1rem; }
.cp-metric .k { color: #5B6577; font-size: .8rem; }
.cp-metric .v { font-family: 'Newsreader', serif; font-size: 1.8rem; font-weight: 600; color: #172033; line-height: 1.2; }
.cp-metric .d { color: #0E7C86; font-size: .85rem; font-weight: 500; }
.cp-empty { background: #fff; border: 1px dashed #B8C0CC; border-radius: 6px; padding: 2.2rem; color: #5B6577; }
section[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #D9DEE5; }
img { border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model…")
def get_model(path: str):
    from tensorflow import keras
    return keras.models.load_model(path, compile=False)


def metric_card(label: str, value: str, delta: str = "") -> str:
    return (f'<div class="cp-metric"><div class="k">{label}</div><div class="v">{value}</div>'
            f'<div class="d">{delta}</div></div>')


def show(col, title: str, image: np.ndarray, caption: str = ""):
    with col:
        st.markdown(f'<div class="cp-label">{title}</div>', unsafe_allow_html=True)
        st.image(to_uint8(image), use_container_width=True)
        if caption:
            st.markdown(f'<div class="cp-caption">{caption}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------------------ header
st.markdown('<h1 class="cp-title">ClearPix</h1>', unsafe_allow_html=True)
st.markdown('<div class="cp-sub">Restore noisy images with a convolutional autoencoder. '
            'Upload a photo, choose how it should be corrupted — or bring one that is already noisy — '
            'and compare the result with the original.</div><hr class="cp-rule">', unsafe_allow_html=True)

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Settings")
    mode = st.radio("Input image is", ["Clean (add noise for me)", "Already noisy"],
                    help="With a clean image, ClearPix adds noise first and can measure restoration quality.")
    simulate = mode.startswith("Clean")
    if simulate:
        noise_type = st.selectbox("Noise type", NOISE_TYPES, format_func=NOISE_LABELS.get,
                                  help="The model is trained on Gaussian noise; other types are harder.")
        level = st.slider("Noise level", 0.01, 0.40, 0.15, 0.01,
                          help="Standard deviation (Gaussian, speckle) or share of corrupted pixels (salt & pepper).")
        seed = st.number_input("Random seed", 0, 99999, 42, help="Keep the seed to get the same noise again.")
    else:
        level = st.slider("Estimated noise level", 0.01, 0.40, 0.15, 0.01,
                          help="Only used to tune the classical filter shown for comparison.")
    compare_classical = st.toggle("Compare with classical filter", value=True,
                                  help="Adds a non-local means result next to the autoencoder output.")
    st.divider()
    st.caption(f"Images larger than {MAX_SIDE}px on the long side are downscaled for speed.")

# ------------------------------------------------------------------------------- model
if not MODEL_PATH.exists():
    st.markdown(f"""<div class="cp-empty"><b>No trained model found.</b><br>
    Train one first, then reload this page:<br><br>
    <code>python train.py</code><br><br>
    The best checkpoint is saved to <code>{MODEL_PATH}</code>.</div>""", unsafe_allow_html=True)
    st.stop()
model = get_model(str(MODEL_PATH))

# ------------------------------------------------------------------------------ upload
upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp", "webp"],
                          label_visibility="collapsed")
if upload is None:
    st.markdown('<div class="cp-empty"><b>Upload an image to begin.</b><br>PNG, JPG, BMP or WebP, up to 25 MB.</div>',
                unsafe_allow_html=True)
    st.stop()

try:
    original = load_image(upload, max_side=MAX_SIDE)
except Exception:
    st.error("This file could not be read as an image. Try a PNG or JPG.")
    st.stop()

if simulate:
    noisy = add_noise(original, noise_type, level, seed=int(seed))
else:
    noisy = original

with st.spinner("Restoring…"):
    restored = restore_image(model, noisy)
    classical = classical_denoise(noisy, "nlm", level) if compare_classical else None

# --------------------------------------------------------------------------- comparison
cols = st.columns(4 if (simulate and compare_classical) else 3 if (simulate or compare_classical) else 2)
i = 0
if simulate:
    show(cols[i], "Original", original); i += 1
show(cols[i], "Noisy input", noisy, f"{NOISE_LABELS[noise_type]} noise, level {level:.2f}" if simulate else ""); i += 1
show(cols[i], "ClearPix restored", restored); i += 1
if compare_classical:
    show(cols[i], "Non-local means", classical, "Classical baseline")

# ------------------------------------------------------------------------------ metrics
if simulate:
    st.markdown('<hr class="cp-rule" style="margin-top:1.8rem">', unsafe_allow_html=True)
    p_noisy, p_rest = psnr(original, noisy), psnr(original, restored)
    s_noisy, s_rest = ssim(original, noisy), ssim(original, restored)
    m = st.columns(4 if compare_classical else 4)
    m[0].markdown(metric_card("PSNR, noisy", f"{p_noisy:.2f} dB"), unsafe_allow_html=True)
    m[1].markdown(metric_card("PSNR, ClearPix", f"{p_rest:.2f} dB", f"{p_rest - p_noisy:+.2f} dB"), unsafe_allow_html=True)
    m[2].markdown(metric_card("SSIM, noisy", f"{s_noisy:.3f}"), unsafe_allow_html=True)
    m[3].markdown(metric_card("SSIM, ClearPix", f"{s_rest:.3f}", f"{s_rest - s_noisy:+.3f}"), unsafe_allow_html=True)
    if compare_classical:
        st.caption(f"Non-local means for reference: {psnr(original, classical):.2f} dB PSNR, "
                   f"{ssim(original, classical):.3f} SSIM.")
else:
    st.caption("Quality metrics need a clean reference. Choose “Clean (add noise for me)” to see PSNR and SSIM.")

# ----------------------------------------------------------------------------- download
buf = io.BytesIO()
Image.fromarray(to_uint8(restored)).save(buf, format="PNG")
st.download_button("Download restored image", buf.getvalue(), file_name="clearpix_restored.png", mime="image/png")
