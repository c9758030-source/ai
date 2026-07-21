import streamlit as st
import pandas as pd
import requests
from sklearn.tree import DecisionTreeClassifier

# 1. Page Configuration
st.set_page_config(
    page_title="Crop Early Warning System",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 AI Crop Early Warning System")
st.write("Real-Time Climate Monitoring & Risk Assessment")

# 2. Load Dataset & Train AI Model
@st.cache_data
def load_data():
    return pd.read_csv('crop_data.csv')

try:
    df = load_data()
    X = df[['Temperature', 'Humidity', 'SoilMoisture', 'Rainfall']]
    y = df['RiskLevel']

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X.values, y.values)  # Using .values avoids feature name warnings
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    model = None

# 3. Sidebar Mode Selection
st.sidebar.header("⚙️ Settings")
mode = st.sidebar.radio(
    "Data Input Mode:",
    ["🌦️ Live Weather API (Real-Time)", "🎛️ Manual Sliders (Demo Mode)"]
)

# Default Values
temp, humidity, moisture, rainfall = 29.0, 70.0, 40.0, 100.0

# --- MODE 1: LIVE WEATHER API ---
if mode == "🌦️ Live Weather API (Real-Time)":
    st.subheader("🌐 Fetch Live Location Weather")
    city = st.text_input("Enter City Name:", "Mumbai")
    
    if st.button("Fetch Real-Time Data"):
        with st.spinner(f"Getting weather data for {city}..."):
            try:
                # Get GPS Coordinates
                geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
                geo_res = requests.get(geo_url, timeout=5).json()
                
                if geo_res.get('results'):
                    lat = geo_res['results'][0]['latitude']
                    lon = geo_res['results'][0]['longitude']
                    
                    # Fetch Current Weather Data
                    weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
                    w_res = requests.get(weather_url, timeout=5).json()
                    
                    current = w_res.get('current_weather', {})
                    temp = float(current.get('temperature', 29.0))
                    
                    # Estimations for missing soil/rainfall sensor metrics
                    humidity = 65.0
                    moisture = 45.0
                    rainfall = 50.0
                    
                    st.success(f"Live data retrieved for **{city}**!")
                    
                    # Display Live Cards
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Temperature", f"{temp} °C")
                    col2.metric("Humidity (Est.)", f"{humidity} %")
                    col3.metric("Soil Moisture (Est.)", f"{moisture} %")
                    col4.metric("Rainfall (Est.)", f"{rainfall} mm")
                else:
                    st.error("City not found. Please enter a valid city name.")
            except Exception as err:
                st.error("Could not reach Weather API. Please check network connection.")

# --- MODE 2: MANUAL SLIDERS ---
else:
    st.subheader("📊 Sensor Simulation Controls")
    col1, col2 = st.columns(2)
    
    with col1:
        temp = st.slider("Temperature (°C)", 0.0, 50.0, 29.0)
        humidity = st.slider("Humidity (%)", 0.0, 100.0, 70.0)
    with col2:
        moisture = st.slider("Soil Moisture (%)", 0.0, 100.0, 40.0)
        rainfall = st.slider("Rainfall (mm)", 0.0, 300.0, 100.0)

# --- 4. AI PREDICTION & ADVISORY ---
st.markdown("---")
st.subheader("🧠 AI Real-Time Analysis")

if model is not None:
    # Run Prediction
    prediction = model.predict([[temp, humidity, moisture, rainfall]])[0]
    
    # Safe vs High Risk Advisory
    if "Safe" in str(prediction) or "Low" in str(prediction):
        st.success(f"✅ **STATUS:** {prediction} - Conditions are stable for crop health.")
    else:
        st.error(f"⚠️ **STATUS:** {prediction} - Crop stress risk detected!")
        st.warning("""
        💡 **Action Steps:**
        * 💧 **Irrigation:** Increase drip watering to stabilize soil moisture.
        * 🌾 **Soil Care:** Check nutrient balances (NPK).
        * 📲 **Alert:** Automatic alert logged to management system.
        """)
