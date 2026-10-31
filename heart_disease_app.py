
import streamlit as st
import pandas as pd
import joblib

# --------------------------------
# Load Trained Heart Disease Model
# --------------------------------

@st.cache_resource
def load_model():
    return joblib.load(
        r"C:\Users\CorvitsystemISB\Desktop\Saad Ai\data\heart_disease_pipeline.pkl"
    )
pipeline = load_model()

# --------------------------------
# Page Configuration
# --------------------------------

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Heart Disease Prediction")
st.subheader("Machine Learning Based Prediction System")

st.write(
    "Enter the patient's medical information below "
    "to get a model prediction."
)

st.divider()

# --------------------------------
# User Input Fields
# --------------------------------

st.header("Patient Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=63
    )

    sex = st.number_input(
        "Sex (0 or 1)",
        min_value=0,
        max_value=1,
        value=1,
        step=1
    )

    cp = st.number_input(
        "Chest Pain Type (cp)",
        min_value=0,
        value=3,
        step=1
    )

    trestbps = st.number_input(
        "Resting Blood Pressure",
        min_value=0,
        value=145
    )

    chol = st.number_input(
        "Cholesterol",
        min_value=0,
        value=233
    )

with col2:
    fbs = st.number_input(
        "Fasting Blood Sugar (fbs)",
        min_value=0,
        value=1,
        step=1
    )

    restecg = st.number_input(
        "Resting ECG (restecg)",
        min_value=0,
        value=0,
        step=1
    )

    thalach = st.number_input(
        "Maximum Heart Rate (thalach)",
        min_value=0,
        value=150
    )

    exang = st.number_input(
        "Exercise Induced Angina (exang)",
        min_value=0,
        value=0,
        step=1
    )

with col3:
    oldpeak = st.number_input(
        "Oldpeak",
        min_value=0.0,
        value=2.3,
        step=0.1
    )

    slope = st.number_input(
        "Slope",
        min_value=0,
        value=0,
        step=1
    )

    ca = st.number_input(
        "Number of Major Vessels (ca)",
        min_value=0,
        value=0,
        step=1
    )

    thal = st.number_input(
        "Thal",
        min_value=0,
        value=1,
        step=1
    )

st.divider()

# --------------------------------
# Prediction Button
# --------------------------------

if st.button("Predict Heart Disease", type="primary"):

    # Create DataFrame in original training order

    input_data = pd.DataFrame([{
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }])

    # Make Prediction

    try:
        prediction = pipeline.predict(input_data)[0]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error(
                "Heart Disease Detected by Model"
            )

        else:
            st.success(
                "No Heart Disease Detected by Model"
            )

        # Show probability if available

        if hasattr(pipeline, "predict_proba"):

            probabilities = pipeline.predict_proba(input_data)[0]

            st.write("Prediction Probabilities")

            st.write(
                f"No Disease: {probabilities[0]*100:.2f}%"
            )

            st.write(
                f"Disease: {probabilities[1]*100:.2f}%"
            )

    except Exception as e:
        st.error(f"Prediction Error: {e}")

st.divider()

st.caption(
    "Educational ML demonstration only. "
    "This prediction is not a medical diagnosis."
)
