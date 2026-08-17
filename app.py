# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 18:15:20 2026

@author: rgupt
"""

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "car_price_model.pkl"

model = joblib.load(MODEL_PATH)
# Load trained pipeline


st.title("Used Car Price Predictor")

st.write(
    "Enter the vehicle details below to estimate its market price."
)

make = st.selectbox(
    "Make",
    ["Audi", "BMW", "Ford", "Hyundai", "Mercedes", "Skoda", "Vauxhall"]
)

car_model = st.text_input("Model")

year = st.number_input(
    "Year",
    min_value=1996,
    max_value=2026,
    value=2018
)

transmission = st.selectbox(
    "Transmission",
    ["Manual", "Automatic", "Semi-Auto"]
)

mileage = st.number_input(
    "Mileage",
    min_value=0,
    value=30000
)

fuel_type = st.selectbox(
    "Fuel Type",
    ["Petrol", "Diesel", "Hybrid", "Electric", "Other"]
)

tax = st.number_input(
    "Tax (£)",
    min_value=0,
    value=150
)

mpg = st.number_input(
    "MPG",
    min_value=0.0,
    value=50.0
)

engine_size = st.number_input(
    "Engine Size (L)",
    min_value=0.0,
    value=2.0
)

if st.button("Predict Price"):

    input_data = pd.DataFrame({
        "model": [car_model],
        "year": [year],
        "transmission": [transmission],
        "mileage": [mileage],
        "fuelType": [fuel_type],
        "tax": [tax],
        "mpg": [mpg],
        "engineSize": [engine_size],
        "make": [make]
    })

    prediction = model.predict(input_data)[0]

    st.subheader(
        f"Estimated Price: £{prediction:,.0f}"
    )

