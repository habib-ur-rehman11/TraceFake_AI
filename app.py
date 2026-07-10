import os
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import exifread

from ela_analysis    import ela_tamper_score
from gan_fingerprint import detect_gan_fingerprint, spectrum_to_pil

# ── Fusion weights (must sum to 1.0) ──────────────────────────────────────
W_CNN = 0.70   # trained classifier — high trust
W_ELA = 0.15   # heuristic — low trust
W_GAN = 0.15   # heuristic — low trust
FAKE_THRESHOLD = 0.50   # weighted_score > this → FAKE


# Page config & CSS
# -----------------------------------------------------------------------------------------
st.set_page_config(page_title="TraceFake", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button {
        background-color: #3498db; color: white;
        border-radius: 5px; width: 100%;
    }
    .result-card {
        background-color: white;
        padding: 30px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .fake-text  { color: #e74c3c; font-weight: bold; font-size: 24px; }
    .real-text  { color: #27ae60; font-weight: bold; font-size: 24px; }
    .signal-row { font-size: 15px; margin: 6px 0; }
    </style>
""", unsafe_allow_html=True)


# Model
# --------------------------------------------------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "/Users/mhabi/Desktop/HAB/Trace_Fake/TraceFake_CS619_BC220423776_FinalDeliverable/models/tracefake_model.keras"
    )

model = load_model()


# EXIF utility ()
# ---------------------------------------------------------------------------------
def get_exif(img_path: str) -> dict:
    with open(img_path, "rb") as f:
        tags = exifread.process_file(f)
    return {
        "Camera Model": str(tags.get("Image Model",           "Unknown")),
        "Software":     str(tags.get("Image Software",        "None Detected")),
        "Date Taken":   str(tags.get("EXIF DateTimeOriginal", "Not Available")),
        "GPS Info":     "None" if not tags.get("GPS GPSLatitude") else "Available",
    }


# UI – upload
# -------------------------------------------------------------------------------------------
st.markdown('<div class="result-card">', unsafe_allow_html=True)
st.title("TraceFake — AI Image Authenticity Checker")
st.write("Upload an image to to verfy.")
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"],
                                  label_visibility="collapsed")
analyze_btn = st.button("Analyze Image")
st.markdown("</div>", unsafe_allow_html=True)


# Analysis pipeline
# -----------------------------------------------------------------------------------
if uploaded_file and analyze_btn:

    TEMP_PATH = "temp_input.jpg"
    with open(TEMP_PATH, "wb") as f:
        f.write(uploaded_file.getbuffer())

    img_pil = Image.open(TEMP_PATH).resize((224, 224)).convert("RGB")
    img_arr = np.expand_dims(np.array(img_pil) / 255.0, axis=0)
    raw_pred = float(model.predict(img_arr)[0][0])   # 1 = Real
    cnn_fake_prob = 1.0 - raw_pred                   # convert to FAKE probability

    ela = ela_tamper_score(TEMP_PATH)


    gan = detect_gan_fingerprint(TEMP_PATH)


    weighted_score = (
        W_CNN * cnn_fake_prob +
        W_ELA * ela["fake_prob"] +
        W_GAN * gan["fake_prob"]
    )
    final_fake = weighted_score > FAKE_THRESHOLD

 
    # 5. EXIF (informational)
    # ------------------------------------------------------------------
    exif_data = get_exif(TEMP_PATH)

   
    # Display — main verdict
    # -----------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.header("Detection Result")

    if final_fake:
        st.markdown('<p class="fake-text">⚠️ AI-Generated / Tampered (FAKE)</p>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<p class="real-text">✅ Authentic (REAL)</p>',
                    unsafe_allow_html=True)

    # Overall confidence: distance from the 0.5 decision boundary, scaled to %
    overall_conf = abs(weighted_score - 0.5) * 200   # 0–100%
    st.write(f"**Confidence:** {overall_conf:.0f}%")
    

   


    # Display — forensic visuals
    # ------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Forensic Visuals")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("Original")
        st.image(uploaded_file)
    with col2:
        st.caption("ELA map")
        st.image(ela["ela_image"])
    with col3:
        st.caption("GAN frequency spectrum")
        st.image(spectrum_to_pil(gan["spectrum"]))
   #----------------------------------------------------------------------------------------------

    st.write(f"Weighted fake score: `{weighted_score:.3f}` (threshold: {FAKE_THRESHOLD})")
    # Signal breakdown
    st.markdown("---")
    st.markdown("**Signal breakdown** (each signal is a fake-probability 0–100%)")

    def prob_bar(p: float) -> str:
        """Simple text bar for probability display."""
        filled = round(p * 10)
        return "█" * filled + "░" * (10 - filled)

    st.markdown(
        f"<div class='signal-row'>🧠 CNN  [{prob_bar(cnn_fake_prob)}] "
        f"{cnn_fake_prob*100:.0f}%  (weight {int(W_CNN*100)}%)</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='signal-row'>🔬 ELA  [{prob_bar(ela['fake_prob'])}] "
        f"{ela['fake_prob']*100:.0f}%  (weight {int(W_ELA*100)}%, mean diff: {ela['mean_diff']})</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='signal-row'>📡 GAN  [{prob_bar(gan['fake_prob'])}] "
        f"{gan['fake_prob']*100:.0f}%  (weight {int(W_GAN*100)}%, freq ratio: {gan['freq_ratio']})</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Display — EXIF metadata ()
    # --------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("EXIF Metadata"):
        st.caption(
            "Please note that Many authentic photos lose metadata "
            "on upload, and AI tools can embed plausible-looking EXIF data."
        )
        st.table({
            "Property": list(exif_data.keys()),
            "Value":    list(exif_data.values()),
        })

    # Cleanup
    if os.path.exists(TEMP_PATH):
        os.remove(TEMP_PATH)
