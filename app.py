import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open('regressor_model.pkl', 'rb'))

st.title("🏆 Gold Price Prediction App")
st.write("Enter the values below to predict gold price.")

# Input fields
spx = st.number_input("SPX (S&P 500 Index)", value=1500.0)
uso = st.number_input("USO (Oil Price)", value=50.0)
slv = st.number_input("SLV (Silver Price)", value=20.0)
eur_usd = st.number_input("EUR/USD Exchange Rate", value=1.2)

# Predict button
if st.button("Predict"):
    input_data = np.array([[spx, uso, slv, eur_usd]])
    prediction = model.predict(input_data)
    st.success(f"💰 Predicted Gold Price: {prediction[0]:.2f}")
