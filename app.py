import pandas as pd
import streamlit as st
from sklearn.tree import DecisionTreeClassifier

# Set up the web page title
st.set_page_config(page_title="Crop Early Warning System", layout="centered")
st.title("🌾 Crop Early Warning AI Dashboard")
st.write("Adjust the environmental factors below to see the AI's risk prediction in real-time.")

# 1. Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('crop_data.csv')

df = load_data()

# 2. Train the AI Model
X = df[['Temperature', 'Humidity', 'SoilMoisture', 'Rainfall']]
y = df['RiskLevel']
model = DecisionTreeClassifier()
model.fit(X, y)

# 3. Create interactive sliders on the webpage
st.subheader("📊 Live Sensor Inputs")
temp = st.slider("Temperature (°C)", min_value=15, max_value=45, value=30)
humidity = st.slider("Humidity (%)", min_value=10, max_value=100, value=70)
moisture = st.slider("Soil Moisture (%)", min_value=5, max_value=100, value=40)
rainfall = st.slider("Rainfall (mm)", min_value=0, max_value=300, value=100)

# 4. Make live prediction based on sliders
features = [[temp, humidity, moisture, rainfall]]
prediction = model.predict(features)

# 5. Display the beautiful results panel
st.markdown("---")
st.subheader("🤖 AI Real-Time Analysis")

if prediction[0] == 1:
    st.error("⚠️ ALERT: High Risk Conditions Detected! Crops require immediate attention or irrigation.")
else:
    st.success("✅ SAFE: Low Risk Conditions. Environmental levels look stable for sustainable crop growth.")
