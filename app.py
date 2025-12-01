import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# --- Configuration Constants ---
# *** Path to the unzipped 'plant_detector_savedmodel' folder ***
MODEL_PATH = 'plant_detector_savedmodel'
# **********************************************
IMG_SIZE = 224
CLASS_NAMES = ['ficus benjamina', 'maidenhair fern', 'oleander', 'snake plant', 'ti plant']


# --- Model Loading (Keras 3 Compatible using TFSMLayer) ---

@st.cache_resource
def load_model_fixed():
    """
    Loads the model using tf.keras.layers.TFSMLayer, which is compatible
    with the TensorFlow SavedModel format in Keras 3 environments.
    """
    if not os.path.isdir(MODEL_PATH):
        st.error(f"❌ Model folder not found at: {MODEL_PATH}.")
        st.info(
            "Please ensure the unzipped 'plant_detector_savedmodel' folder is in the same directory as this app.py file.")
        st.stop()

    try:
        # FIX: Using TFSMLayer for Keras 3 compatibility with SavedModel format
        model_layer = tf.keras.layers.TFSMLayer(
            MODEL_PATH,
            call_endpoint='serving_default',
            dtype=tf.float32  # Specify the input dtype
        )
        st.success("✅ Model loaded successfully using TFSMLayer (Keras 3 compatible)!")
        return model_layer
    except Exception as e:
        st.error(f"❌ CRITICAL ERROR LOADING MODEL: {e}")
        st.info("Ensure all dependencies are installed: `pip install streamlit tensorflow pillow`")
        st.stop()


# --- Prediction Function ---

def predict(image, model_layer):
    """
    Predicts the class and confidence score for an uploaded image.
    The TFSMLayer returns a dictionary, so we access the 'output_0' key.
    """
    # Preprocessing the image
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32)
    img_array = tf.expand_dims(img_array, 0)
    img_array = img_array / 255.0

    # Prediction (TFSMLayer returns a dictionary)
    raw_outputs = model_layer(img_array)

    # In most common SavedModel cases, the output is stored under 'output_0'
    if 'output_0' in raw_outputs:
        predictions = raw_outputs['output_0']
    else:
        # Fallback for models with different output names
        st.error("Model output key not found. Expected 'output_0'.")
        return "Prediction Error", 0.0

    # Post-processing
    score = tf.nn.softmax(predictions[0])
    predicted_class_index = np.argmax(score)

    if predicted_class_index < len(CLASS_NAMES):
        predicted_class = CLASS_NAMES[predicted_class_index]
        confidence = np.max(score) * 100
        return predicted_class, confidence
    else:
        return "Unknown Class", 0.0


# --- Streamlit App Initialization ---

# Load the TFSMLayer instance
model_layer = load_model_fixed()

st.set_page_config(
    page_title="🌿 AI Plant Detector Demo",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("🌿 AI Plant Detector Demo")
st.markdown("Upload a photo of one of the 5 plants to get an instant classification.")
st.markdown("---")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.markdown("---")

    with st.spinner('Analyzing plant...'):
        # Pass the TFSMLayer instance to the predict function
        predicted_class, confidence = predict(image, model_layer)

    st.success("✅ Analysis Complete!")

    st.markdown(f"""
        <div style="background-color: #ecfdf5; padding: 20px; border-radius: 12px; border-left: 5px solid #059669;">
            <p style="font-size: 1.1rem; margin: 0;">Prediction:</p>
            <h3 style="color: #065f46; margin-top: 5px; margin-bottom: 5px;">{predicted_class.capitalize()}</h3>
            <p style="font-size: 1.1rem; margin: 0;">Confidence: <b>{confidence:.2f}%</b></p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---")