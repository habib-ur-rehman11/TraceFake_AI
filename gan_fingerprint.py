"""
gan_fingerprint.py
------------------
Frequency-domain GAN fingerprint detection for TraceFake.

IMPORTANT — confidence output:
    Returns fake_prob in [0.0, 1.0]: probability the image is GAN-generated.
    0.0 = very likely real, 1.0 = very likely GAN.
    This is a heuristic — weight it lower than the trained CNN.

CALIBRATION (do this once on your dataset):
    Run detect_gan_fingerprint() on known-real and known-fake images.
    Print freq_ratio for each. Then set:
        _REAL_RATIO = 90th percentile of real-image freq_ratio values
        _FAKE_RATIO = 10th percentile of fake-image freq_ratio values
"""

import numpy as np
import cv2
from scipy.fftpack import fft2, fftshift
from PIL import Image

_REAL_RATIO = 0.60   # freq_ratio → 0% fake probability (tune this)
_FAKE_RATIO = 0.85   # freq_ratio → 100% fake probability (tune this)


def _compute_spectrum(image_path: str) -> tuple:
    """Load image, convert to grayscale, return log-magnitude FFT spectrum."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    gray      = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f_shift   = fftshift(fft2(gray))
    spectrum  = np.log(np.abs(f_shift) + 1)
    return spectrum, gray


def detect_gan_fingerprint(image_path: str) -> dict:
    """
    Returns a GAN-detection verdict based on frequency-domain analysis.

    GAN upsampling layers create periodic artifacts in high-frequency bands.
    We measure the ratio of high-frequency to total energy as the signal.

    Returns
    -------
    dict with:
        fake_prob  : float      — fake probability in [0.0, 1.0]
        freq_ratio : float      — raw energy ratio (for display/tuning)
        spectrum   : np.ndarray — log-magnitude spectrum array (for visualisation)
    """
    spectrum, _ = _compute_spectrum(image_path)

    h, w = spectrum.shape
    ch, cw = h // 2, w // 2

    # Isolate high-frequency ring by zeroing the low-frequency centre
    high_freq = spectrum.copy()
    high_freq[ch - 30: ch + 30, cw - 30: cw + 30] = 0

    freq_ratio = float(np.mean(high_freq) / (np.mean(spectrum) + 1e-6))

    # Linear interpolation, clamped to [0, 1]
    fake_prob = (freq_ratio - _REAL_RATIO) / (_FAKE_RATIO - _REAL_RATIO)
    fake_prob = float(np.clip(fake_prob, 0.0, 1.0))

    return {
        "fake_prob":  round(fake_prob, 3),
        "freq_ratio": round(freq_ratio, 4),
        "spectrum":   spectrum,
    }


def spectrum_to_pil(spectrum: np.ndarray) -> Image.Image:
    """
    Converts a log-magnitude spectrum array to an RGB PIL Image ('hot' colormap)
    so it can be displayed with st.image() in Streamlit.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import io

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(spectrum, cmap="hot")
    ax.set_title("GAN Fingerprint\n(Freq. Spectrum)", fontsize=9)
    ax.axis("off")
    plt.tight_layout(pad=0.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()
