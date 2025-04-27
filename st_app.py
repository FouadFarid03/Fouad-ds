import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Caching data and figures
@st.cache_data
def load_data_and_create_figure():
    df = pd.read_csv("climatenew.csv")
    df2 = pd.read_csv('foodnew.csv')

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Avg Temp vs Crop Yield',
            'Year vs Crop Yield',
            'Farm vs Total Emissions',
            'Food Product vs Total Emissions'
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )

    fig.add_trace(go.Scatter(x=df['Average_Temperature_C'], y=df['Crop_Yield_MT_per_HA'], mode='markers'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Year'], y=df['Crop_Yield_MT_per_HA'], mode='lines'), row=1, col=2)
    fig.add_trace(go.Scatter(x=df2['Farm'], y=df2['Total_emissions'], mode='markers'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df2['Food product'], y=df2['Total_emissions'], mode='lines'), row=2, col=2)

    fig.update_layout(height=800, width=1000, title_text="Climate and Food Visualizations")
    return df, df2, fig

df, df2, fig = load_data_and_create_figure()

st.title('🌾 Agriculture and Food Prediction System 🌍')
st.plotly_chart(fig)

# Load regression models
model_climate_reg1 = joblib.load('climate_reg_linear.pkl')
model_climate_reg2 = joblib.load('climate_reg_rf.pkl')
model_climate_reg3 = joblib.load('climate_reg_svr.pkl')

model_food_reg1 = joblib.load('food_reg_linear.pkl')
model_food_reg2 = joblib.load('food_reg_rf.pkl')
model_food_reg3 = joblib.load('food_reg_svr.pkl')

# --- Inputs Section ---

st.header('🌦️ Climate Inputs')
col1, col2, col3 = st.columns(3)

with col1:
    avg_temp = st.number_input('Average Temperature (°C)', min_value=-50, max_value=50)
    precipitation = st.number_input('Total Precipitation (mm)', min_value=0, max_value=5000)
    co2_emissions = st.number_input('CO2 Emissions (MT)', min_value=0, max_value=10000)

with col2:
    irrigation_access = st.number_input('Irrigation Access (%)', min_value=0, max_value=100)
    soil_health = st.number_input('Soil Health Index', min_value=0, max_value=100)
    year = st.number_input('Year', min_value=2000, max_value=2100)

with col3:
    # Any additional climate variables can be added here if needed
    pass

st.header('🍎 Food Inputs')
col4, col5, col6 = st.columns(3)

with col4:
    land_use_change = st.number_input('Land Use Change', min_value=0, max_value=1000)
    animal_feed = st.number_input('Animal Feed', min_value=0, max_value=1000)

with col5:
    farm = st.number_input('Farm', min_value=0, max_value=10000)
    processing = st.number_input('Processing', min_value=0, max_value=10000)

with col6:
    transport = st.number_input('Transport', min_value=0, max_value=10000)

# --- Predictions ---
if st.button('Predict 🚀'):
    # Prepare input data for prediction
    X_climate_input = np.array([[avg_temp, precipitation, co2_emissions, irrigation_access, soil_health]])
    X_food_input = np.array([[land_use_change, animal_feed, farm, processing, transport]])

    # Get predictions from the regression models
    climate_preds = [model_climate_reg1.predict(X_climate_input)[0], 
                     model_climate_reg2.predict(X_climate_input)[0], 
                     model_climate_reg3.predict(X_climate_input)[0]]

    food_preds = [model_food_reg1.predict(X_food_input)[0], 
                  model_food_reg2.predict(X_food_input)[0], 
                  model_food_reg3.predict(X_food_input)[0]]

    # Display Results
    st.subheader('🌦️ Climate Predictions')
    st.write(f"Model 1 (Linear Regression) Prediction: {climate_preds[0]:.2f} MT per HA")
    st.write(f"Model 2 (Random Forest) Prediction: {climate_preds[1]:.2f} MT per HA")
    st.write(f"Model 3 (SVR) Prediction: {climate_preds[2]:.2f} MT per HA")

    st.subheader('🍎 Food Predictions')
    st.write(f"Model 1 (Linear Regression) Prediction: {food_preds[0]:.2f} Total Emissions")
    st.write(f"Model 2 (Random Forest) Prediction: {food_preds[1]:.2f} Total Emissions")
    st.write(f"Model 3 (SVR) Prediction: {food_preds[2]:.2f} Total Emissions")
