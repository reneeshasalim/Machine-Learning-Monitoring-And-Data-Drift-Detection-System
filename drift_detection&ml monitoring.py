# app.py
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

st.set_page_config(page_title="ML Monitoring & Drift Detection", layout="wide")

# Load model & encoders
model = pickle.load(open("model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))
reference_df = pd.read_csv("reference_data.csv")

features = ["Pclass", "Sex", "Age", "Fare", "Embarked"]

st.title("🚢 Titanic ML Monitoring & Drift Detection")

# ---------------- USER INPUT ----------------
st.sidebar.header("Passenger Input")

pclass = st.sidebar.selectbox("Pclass", [1, 2, 3])
sex = st.sidebar.selectbox("Sex", ["male", "female"])
age = st.sidebar.slider("Age", 1, 80, 30)
fare = st.sidebar.slider("Fare", 0.0, 500.0, 50.0)
embarked = st.sidebar.selectbox("Embarked", ["S", "C", "Q"])

# Encode input
input_data = pd.DataFrame([{
    "Pclass": pclass,
    "Sex": encoders["sex"].transform([sex])[0],
    "Age": age,
    "Fare": fare,
    "Embarked": encoders["embarked"].transform([embarked])[0]
}])

# ---------------- PREDICTION ----------------
prediction = model.predict(input_data)[0]
prob = model.predict_proba(input_data)[0][1]

st.subheader("🎯 Prediction Result")
st.write("Survived" if prediction == 1 else "Not Survived")
st.write(f"Survival Probability: {prob:.2f}")

# ---------------- DRIFT DETECTION ----------------
st.subheader("📊 Feature Drift Detection (KS Test)")

drift_results = []

for col in features:
    ref = reference_df[col]
    curr = input_data[col]

    stat, p_value = ks_2samp(ref, curr)
    drift = "Yes" if p_value < 0.05 else "No"

    drift_results.append({
        "Feature": col,
        "KS Statistic": round(stat, 3),
        "P-Value": round(p_value, 5),
        "Drift Detected": drift
    })

drift_df = pd.DataFrame(drift_results)
st.dataframe(drift_df)

# ---------------- VISUALIZATION ----------------
st.subheader("📈 Feature Distribution Monitoring")

selected_feature = st.selectbox("Select feature", features)

fig, ax = plt.subplots()
ax.hist(reference_df[selected_feature], bins=30, alpha=0.6, label="Reference")
ax.axvline(input_data[selected_feature].values[0], color="red", label="Current")
ax.legend()
st.pyplot(fig)


