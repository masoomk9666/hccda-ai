import streamlit as st
import pandas as pd
import pickle

# ---------------- Page Setup ----------------
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    h1 { color: #ff4b4b; text-align: center; }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 18px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.title("❤️ Heart Disease Prediction")
st.write("Logistic Regression model.")

# ---------------- Load Trained Pipeline ----------------
# NOTE: Model is NOT trained here. It was already cleaned + trained
# inside heart_disease_training.ipynb and saved as a .pkl file.
# This app only LOADS that trained pipeline and uses it to predict.

FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

@st.cache_resource
def load_pipeline():
    with open(r"C:\Users\corvit\Desktop\python\Logistic_Regression\heart_disease_pipeline.pkl", "rb") as file:
        pipeline = pickle.load(file)
    return pipeline

pipeline = load_pipeline()

st.success("✅ Trained model loaded (Test Accuracy: **85.25%**)")
st.subheader("🩺 Enter Your Details ")

# ---------------- User Input Form ----------------
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 20, 90, 50)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
    trestbps = st.slider("Resting Blood Pressure", 90, 200, 120)
    chol = st.slider("Cholesterol", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.selectbox("Resting ECG Results", [0, 1, 2])

with col2:
    thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.5, 1.0, step=0.1)
    slope = st.selectbox("Slope of ST Segment", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thal", [0, 1, 2, 3])

input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg,
                             thalach, exang, oldpeak, slope, ca, thal]],
                           columns=FEATURES)

# ---------------- Prediction ----------------
if st.button("🔍 Predict"):
    prediction = pipeline.predict(input_data)[0]
    probability = pipeline.predict_proba(input_data)[0][1]

    st.markdown("---")
    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease — Probability: {probability*100:.1f}%")
    else:
        st.success(f"✅ Low Risk of Heart Disease — Probability: {probability*100:.1f}%")

    st.progress(float(probability))
