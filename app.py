# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# -----------------------------
# Load model and app data
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "car_price_model.pkl"
OPTIONS_PATH = BASE_DIR / "models" / "car_options.csv"

# Load trained preprocessing + XGBoost pipeline
model = joblib.load(MODEL_PATH)

# Load valid make/model combinations from the training dataset
car_options = pd.read_csv(OPTIONS_PATH)


# -----------------------------
# Page heading
# -----------------------------

st.title("Used Car Price Predictor")

st.write(
    "Enter the vehicle details below to estimate its value "
    "based on the UK used-car market represented by the training data."
)

st.info(
    "This model was trained on historical UK used-car listings up to 2020. "
    "Predictions should not be interpreted as current market valuations."
)


# -----------------------------
# Vehicle make and model
# -----------------------------

# List all makes available in the training dataset
makes = sorted(car_options["make"].unique())

make = st.selectbox(
    "Make",
    makes
)

# Filter the available models based on the selected make
models_for_make = sorted(
    car_options.loc[
        car_options["make"] == make,
        "model"
    ].unique()
)

car_model = st.selectbox(
    "Model",
    models_for_make
)


# -----------------------------
# Numerical vehicle features
# -----------------------------

year = st.slider(
    "Year",
    min_value=1996,
    max_value=2020,
    value=2018,
    step=1
)

mileage = st.slider(
    "Mileage",
    min_value=0,
    max_value=200000,
    value=30000,
    step=1000
)

tax = st.slider(
    "Road Tax (£)",
    min_value=0,
    max_value=600,
    value=150,
    step=5
)

mpg = st.slider(
    "MPG",
    min_value=0.0,
    max_value=250.0,
    value=50.0,
    step=0.5
)

engine_size = st.slider(
    "Engine Size (L)",
    min_value=0.0,
    max_value=6.5,
    value=2.0,
    step=0.1
)


# -----------------------------
# Categorical vehicle features
# -----------------------------

transmission = st.selectbox(
    "Transmission",
    ["Manual", "Automatic", "Semi-Auto"]
)

fuel_type = st.selectbox(
    "Fuel Type",
    ["Petrol", "Diesel", "Hybrid", "Electric", "Other"]
)


# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Price"):

    # Create a single-row DataFrame matching the features
    # used when training the model
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

    # The saved pipeline automatically performs preprocessing
    # before passing the data to XGBoost
    prediction = model.predict(input_data)[0]

    st.subheader(
        f"Estimated 2020-market value: £{prediction:,.0f}"
    )