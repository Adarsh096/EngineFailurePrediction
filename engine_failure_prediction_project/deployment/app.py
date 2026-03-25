import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
import os

# Common constants
HUGGINGFACE_USER_NAME = os.getenv('HUGGINGFACE_USER_NAME')
HUGGINGFACE_MODEL_NAME = os.getenv('HUGGINGFACE_MODEL_NAME')

# Download the model from the Model Hub
@st.cache_resource # Use caching to avoid re-downloading on every slider move
def load_remote_model():
    try:
        repo_id = f"{HUGGINGFACE_USER_NAME}/{HUGGINGFACE_MODEL_NAME}"
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename="model.joblib"
        )
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model from Hugging Face: {e}")
        st.stop()

model = load_remote_model()

# Streamlit UI Setup
st.set_page_config(page_title="Engine Failure Prediction", layout="centered")
st.title("Engine Failure Prediction")
st.write("""
This tool predicts engine health based on real-time telemetry.
Adjust the sliders below to simulate engine sensor data.
""")

st.divider()

# Create UI Layout
col1, col2 = st.columns(2)

with col1:
    engine_rpm = st.number_input("Engine RPM", min_value=20, max_value=2500, value=791, step=1, help="Rotations per minute")
    lub_oil_pressure = st.number_input("Lub Oil Pressure (bar)", min_value=0.0, max_value=8.0, value=3.3, step=0.1)
    fuel_pressure = st.number_input("Fuel Pressure (bar)", min_value=0.0, max_value=25.0, value=6.6, step=0.1)

with col2:
    coolant_pressure = st.number_input("Coolant Pressure (bar)", min_value=0.0, max_value=8.0, value=2.3, step=0.1)
    lub_oil_temp = st.number_input("Lub Oil Temp (°C)", min_value=30.0, max_value=100.0, value=77.6, step=0.1)
    coolant_temp = st.number_input("Coolant Temp (°C)", min_value=30.0, max_value=200.0, value=78.4, step=0.1)

# Prepare input data matching the exact training schema
# these keys match the 'numeric_scaling' list in our model training script
input_dict = {
    "engine_rpm": engine_rpm,
    "lub_oil_pressure": lub_oil_pressure,
    "fuel_pressure": fuel_pressure,
    "coolant_pressure": coolant_pressure,
    "lub_oil_temp": lub_oil_temp,
    "coolant_temp": coolant_temp
}

input_data = pd.DataFrame([input_dict])

# Prediction Logic
# We use 0.45 to be slightly more sensitive to failures (maximizing Recall)
classification_threshold = 0.45

st.divider()

if st.button("Generate Prediction", type="primary"):
    # The 'model' here is the Scikit-Learn Pipeline
    # It automatically runs the StandardScaler on input_data before passing to XGBoost
    prediction_proba = model.predict_proba(input_data)[0, 1]

    # Apply custom threshold
    prediction = 1 if prediction_proba >= classification_threshold else 0

    if prediction == 1:
        st.error(f"### ⚠️ CRITICAL: Engine Failure Likely\n**Probability of Failure:** {prediction_proba:.2%}")
        st.write("Immediate maintenance inspection recommended to avoid service disruption.")
    else:
        st.success(f"### ✅ NORMAL: Engine Healthy\n**Probability of Failure:** {prediction_proba:.2%}")
        st.write("Engine parameters are within safe operating margins.")

# Add technical metadata for your portfolio
with st.expander("View Model & System Details"):
    st.write(f"**Model Source:** Hugging Face Hub ({HUGGINGFACE_MODEL_NAME})")
    st.write(f"**Threshold Applied:** {classification_threshold}")
    st.write("**Architecture:** Pipeline(StandardScaler -> XGBoost)")
