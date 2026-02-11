import streamlit as st
import joblib
import numpy as np
import pandas as pd

# =========================
# Load model and encoders
# =========================
model = joblib.load("model.pkl")
ordinal_encoder = joblib.load("ordinal_encoder.pkl")
onehot_encoder = joblib.load("onehot_encoder.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# =========================
# App UI
# =========================
st.title("Bodaboda Daily Income Prediction App 🇹🇿")
st.write("This AI application predicts the daily income of a bodaboda rider based on working conditions.")

# =========================
# User Inputs
# =========================
hours_worked = st.number_input("Hours worked per day", min_value=1, max_value=24, value=8)
fuel_cost_tzs = st.number_input("Fuel cost (TZS)", min_value=0, value=7000)
distance_km = st.number_input("Distance covered (km)", min_value=0, value=80)

rainy_day_input = st.selectbox("Was it a rainy day?", ["No", "Yes"])
road_condition_input = st.selectbox("Road condition", ["Poor", "Fair", "Good"])
location_type_input = st.selectbox("Location type", ["Rural", "Urban", "Suburban"])

# =========================
# Prediction
# =========================
if st.button("Predict Daily Income"):

    # Encode rainy_day (Label Encoding)
    rainy_day_encoded = label_encoder.transform([rainy_day_input])[0]

    # Encode road_condition (Ordinal Encoding)
    road_condition_encoded = ordinal_encoder.transform([[road_condition_input]])[0][0]

    # Encode location_type (One-Hot Encoding)
    location_encoded = onehot_encoder.transform([[location_type_input]])
    
    # Convert sparse matrix to array if needed
    if hasattr(location_encoded, "toarray"):
        location_encoded = location_encoded.toarray()
    
    # Use encoder's feature names
    location_encoded_df = pd.DataFrame(
        location_encoded,
        columns=onehot_encoder.get_feature_names_out()
    )

    # Combine all inputs
    input_df = pd.DataFrame({
        "hours_worked": [hours_worked],
        "fuel_cost_tzs": [fuel_cost_tzs],
        "distance_km": [distance_km],
        "rainy_day": [rainy_day_encoded],
        "road_condition": [road_condition_encoded]
    })

    final_input = pd.concat([location_encoded_df, input_df], axis=1)

    # Predict
    prediction = model.predict(final_input)

    st.success(f"Estimated Daily Income: {int(prediction[0]):,} TZS 💰")
