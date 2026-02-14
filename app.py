
import streamlit as st

import pandas as pd
import joblib
import numpy as np


# =========================
# Load trained pipeline
# =========================
# Make sure you have a pipeline saved like in train_model.py
model = joblib.load("model.pkl")
ordinal_encoder = joblib.load("ordinal_encoder.pkl")
onehot_encoder = joblib.load("onehot_encoder.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# =========================
# App UI
# =========================
st.title("🛵 Bodaboda Daily Income Prediction App 🇹🇿")
st.write("Predict the daily income of a bodaboda rider based on working conditions.")

# =========================
# User Inputs
# =========================
hours_worked = st.number_input("Hours Worked per Day", min_value=1, max_value=24, value=8)
number_of_trips = st.number_input("Number of Trips", min_value=1, max_value=50, value=25)
fuel_cost = st.number_input("Fuel Cost (TZS)", min_value=0,  value=7000)
rainy_day = st.selectbox("Was it a Rainy Day?", ["No", "Yes"])
road_condition = st.selectbox("Road Condition", ["Poor", "Fair", "Good"])
location_type = st.selectbox("Location Type", ["Urban", "Rural"])

#Map user input to the encoded values
#label encode
rainy_day_encoded = label_encoder.transform([rainy_day])[0]
#ordinal encode
road_condition_encoded = ordinal_encoder.transform([[road_condition]])[0][0]


#One-hot encode location_type(match training column)
## Location_type (OneHotEncoder)
location_encoded = onehot_encoder.transform([[location_type]])
if hasattr(location_encoded, "toarray"):
    location_encoded = location_encoded.toarray()[0]


#combine all features in same order as training
#original preprocessing stacked one-hot first, then numeric + encoded
input_array = np.array(
    list(location_encoded) + [hours_worked, number_of_trips, fuel_cost, rainy_day_encoded, road_condition_encoded]
).reshape(1, -1)


# Predict button

if st.button("Predict Daily Income"):
    prediction = model.predict(input_array)
    st.success(f"Estimated Daily Income: {prediction.item():,.2f} TZS 💰")



# =========================
# Optional: Visualizations
# =========================
if st.checkbox("Show Dataset Visualizations"):
    df = pd.read_csv("bodaboda_income_dataset.csv")  # Make sure CSV is in project folder

    st.subheader("Income Distribution")
    st.bar_chart(df['daily_income'])

    st.subheader("Income by Road Condition")
    st.bar_chart(df.groupby('road_condition')['daily_income'].mean())

    st.subheader("Income by Location Type")
    st.bar_chart(df.groupby('location_type')['daily_income'].mean())
