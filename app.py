import streamlit as st
import pandas as pd
import pickle

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# ---------------------------
# Load Model Files
# ---------------------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_columns = pickle.load(open("features.pkl", "rb"))

# ---------------------------
# Load Dataset
# ---------------------------
df = pd.read_csv("dataset.csv")

manufacturers = sorted(df["Name"].str.split().str[0].unique())
fuel_types = sorted(df["Fuel_Type"].unique())
transmissions = sorted(df["Transmission"].unique())
owners = sorted(df["Owner_Type"].unique())

# ---------------------------
# Title
# ---------------------------
st.title("🚗 Used Car Price Prediction")
st.markdown("Predict the resale value of a used car using Machine Learning.")

st.divider()

# ---------------------------
# Input Form
# ---------------------------

col1, col2 = st.columns(2)

with col1:

    manufacturer = st.selectbox(
        "Manufacturer",
        manufacturers
    )

    km = st.number_input(
        "Kilometers Driven",
        min_value=0,
        value=50000
    )

    mileage = st.number_input(
        "Mileage (kmpl)",
        min_value=0.0,
        value=18.0
    )

    engine = st.number_input(
        "Engine (CC)",
        min_value=500,
        value=1200
    )

with col2:

    power = st.number_input(
        "Power (bhp)",
        min_value=20.0,
        value=80.0
    )

    seats = st.number_input(
        "Seats",
        min_value=2,
        max_value=10,
        value=5
    )

    fuel = st.selectbox(
        "Fuel Type",
        fuel_types
    )

    transmission = st.selectbox(
        "Transmission",
        transmissions
    )

    owner = st.selectbox(
        "Owner Type",
        owners
    )

st.divider()

# ---------------------------
# Prediction
# ---------------------------

if st.button("Predict Price"):

    input_df = pd.DataFrame({

        "Kilometers_Driven":[km],

        "Fuel_Type":[fuel],

        "Transmission":[transmission],

        "Owner_Type":[owner],

        "Mileage":[mileage],

        "Engine":[engine],

        "Power":[power],

        "Seats":[seats],

        "Manufacturer":[manufacturer]

    })

    input_df = pd.get_dummies(
        input_df,
        columns=[
            "Manufacturer",
            "Fuel_Type",
            "Transmission",
            "Owner_Type"
        ],
        drop_first=True
    )

    input_df = input_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]

    st.success(
        f"Estimated Car Price: ₹ {prediction:,.2f} Lakhs"
    )