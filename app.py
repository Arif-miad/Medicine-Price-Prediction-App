import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved model, scaler, and encoders
model = joblib.load("medicine_price_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.title("💊 Medicine Price Prediction App")

st.markdown("""
Predict the **price of a medicine** based on its features using our trained ML model.
""")

# Input fields
medicine_name = st.selectbox(
    "Medicine Name",
    list(label_encoders["medicine_name"].classes_)
)

category = st.selectbox(
    "Category",
    list(label_encoders["category"].classes_)
)

company = st.selectbox(
    "Company",
    list(label_encoders["company"].classes_)
)

dosage_mg = st.number_input("Dosage (mg)", min_value=1, value=500)
pack_size = st.number_input("Pack Size", min_value=1, value=10)
manufacturing_cost = st.number_input("Manufacturing Cost", min_value=1.0, value=50.0)

import_status = st.selectbox(
    "Import Status",
    list(label_encoders["import_status"].classes_)
)

demand_level = st.selectbox(
    "Demand Level",
    list(label_encoders["demand_level"].classes_)
)

expiry_months = st.number_input("Months until Expiry", min_value=1, value=24)
prescription_required = st.selectbox(
    "Prescription Required",
    [0, 1]
)

# Prepare input dataframe
input_dict = {
    "medicine_name": [label_encoders["medicine_name"].transform([medicine_name])[0]],
    "category": [label_encoders["category"].transform([category])[0]],
    "company": [label_encoders["company"].transform([company])[0]],
    "dosage_mg": [dosage_mg],
    "pack_size": [pack_size],
    "manufacturing_cost": [manufacturing_cost],
    "import_status": [label_encoders["import_status"].transform([import_status])[0]],
    "demand_level": [label_encoders["demand_level"].transform([demand_level])[0]],
    "expiry_months": [expiry_months],
    "prescription_required": [prescription_required]
}

input_df = pd.DataFrame(input_dict)

# Scale input
input_scaled = scaler.transform(input_df)

# Predict price
if st.button("Predict Price"):
    predicted_price = model.predict(input_scaled)
    st.success(f"💰 Predicted Price: {predicted_price[0]:.2f}")
