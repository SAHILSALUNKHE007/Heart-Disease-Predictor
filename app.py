import streamlit as st
import pandas as pd
import joblib as jb

# Load artifacts
model = jb.load("Logistic_heart.pkl")
scaler = jb.load("scaler.pkl")
columns = jb.load("columns.pkl")

st.title("❤️ Heart Disease Predictor")
st.markdown("Provide the following patient details:")

# -------- INPUTS -------- #

age = st.number_input("Age", 20, 100, 40)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure", 80, 200, 120)
cholesterol = st.number_input("Cholesterol", 100, 400, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.number_input("Maximum Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise Induced Angina", ["N", "Y"])
oldpeak = st.number_input("Oldpeak (ST Depression)", 0.0, 6.0, 0.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# -------- DATAFRAME -------- #

input_data = pd.DataFrame({
    "Age": [age],
    "RestingBP": [resting_bp],
    "Cholesterol": [cholesterol],
    "FastingBS": [fasting_bs],
    "MaxHR": [max_hr],
    "Oldpeak": [oldpeak],
    "Sex_M": [1 if sex == "M" else 0],
    "ChestPainType_ATA": [1 if chest_pain == "ATA" else 0],
    "ChestPainType_NAP": [1 if chest_pain == "NAP" else 0],
    "ChestPainType_ASY": [1 if chest_pain == "ASY" else 0],
    "RestingECG_Normal": [1 if resting_ecg == "Normal" else 0],
    "RestingECG_ST": [1 if resting_ecg == "ST" else 0],
    "RestingECG_LVH": [1 if resting_ecg == "LVH" else 0],
    "ExerciseAngina_Y": [1 if exercise_angina == "Y" else 0],
    "ST_Slope_Flat": [1 if st_slope == "Flat" else 0],
    "ST_Slope_Down": [1 if st_slope == "Down" else 0],
})

# ✅ Ensure same feature order as training
input_data = input_data.reindex(columns=columns, fill_value=0)

# Scale
input_scaled = scaler.transform(input_data)

# -------- PREDICTION -------- #

if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease (Risk: {probability:.2%})")
    else:
        st.success(f"✅ Low Risk of Heart Disease (Risk: {probability:.2%})")
