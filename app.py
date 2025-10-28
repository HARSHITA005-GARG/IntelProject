import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import io

# ---------------------------
# Load model once
# ---------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("pixel_detection_final_model.keras")
    return model

model = load_model()
IMG_SIZE = 224

# ---------------------------
# Preprocessing function
# ---------------------------
def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

# ---------------------------
# Optimized pixel correction with strength control
# ---------------------------
def correct_image_optimized(image, strength=1):
    """Applies adaptive correction based on selected strength (0-3)."""
    img_cv = np.array(image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    # Bilateral filter (edge-preserving smoothing)
    d = 9 + strength * 2
    sigma = 75 + strength * 25
    img_cv = cv2.bilateralFilter(img_cv, d=d, sigmaColor=sigma, sigmaSpace=sigma)

    # Gaussian blur (to reduce pixelation)
    blur_strength = (3 + 2 * strength, 3 + 2 * strength)
    img_cv = cv2.GaussianBlur(img_cv, blur_strength, 0)

    # Sharpening (recover details)
    sharpen_intensity = 1 + 0.5 * strength
    sharpen_kernel = np.array([[0, -1, 0],
                               [-1, 4 + sharpen_intensity, -1],
                               [0, -1, 0]])
    img_cv = cv2.filter2D(img_cv, -1, sharpen_kernel)

    corrected = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    return corrected

# ---------------------------
# Streamlit Interface
# ---------------------------
st.set_page_config(page_title="AI Pixel Detector & Corrector", page_icon="🧠", layout="centered")
st.title("🧠 AI-Powered Pixel Detection & Correction")
st.write("Upload an image — the model detects if it's pixelated and auto-corrects it intelligently.")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    st.write("🔍 Analyzing pixelation...")
    processed = preprocess_image(image)
    prediction = model.predict(processed)[0][0]

    label = "Pixelated" if prediction > 0.5 else "Clean"
    confidence = float(prediction if prediction > 0.5 else 1 - prediction)

    st.markdown(f"### 🧾 Prediction: **{label}** ({confidence:.2%} confidence)")

    if label == "Pixelated":
        st.info("🛠️ Pixelation detected! Adjust correction strength below:")

        # User control for correction strength
        strength = st.slider("Correction Strength", 0, 3, 1,
                             help="0 = Light, 3 = Strong correction")

        # Apply correction
        corrected_image = correct_image_optimized(image, strength=strength)

        # Show comparison
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original (Pixelated)", use_column_width=True)
        with col2:
            st.image(corrected_image, caption=f"Corrected (Strength {strength})", use_column_width=True)

        # Download option
        buf = io.BytesIO()
        corrected_image.save(buf, format="PNG")
        st.download_button("📥 Download Corrected Image", data=buf.getvalue(),
                           file_name="corrected_image.png", mime="image/png")

    else:
        st.success("✅ Image looks clean! No correction needed.")
