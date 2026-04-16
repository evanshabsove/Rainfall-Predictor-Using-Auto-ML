import streamlit as st
import joblib
import pandas as pd
import requests
from datetime import datetime
import os

# Page config
st.set_page_config(
    page_title="Rain Predictor",
    page_icon="🌧️",
    layout="centered"
)

# Load model and preprocessing info
@st.cache_resource
def load_model():
    model = joblib.load("rain_predictor_model.pkl")
    preprocessing = joblib.load("preprocessing_info.pkl")
    return model, preprocessing

model, preprocessing_info = load_model()

# Title and description
st.title("🌧️ Next-Day Rain Predictor")
st.markdown("""
This tool predicts the likelihood of rain tomorrow based on current weather conditions.
Built for event organizers to make informed decisions about outdoor activities.
""")

st.divider()

# Sidebar for API key
st.sidebar.header("⚙️ Configuration")
api_key = st.sidebar.text_input(
    "WeatherAPI.com API Key",
    type="password",
    help="Get a free API key at https://www.weatherapi.com/"
)

st.sidebar.markdown("""
### How to use:
1. Get a free API key from [WeatherAPI.com](https://www.weatherapi.com/)
2. Enter it above
3. Enter your location
4. Click 'Predict Rain Tomorrow'
""")

# Main input
location = st.text_input(
    "📍 Enter Location",
    placeholder="e.g., Sydney, Melbourne, Brisbane",
    help="Enter a city name or coordinates"
)

predict_button = st.button("🔮 Predict Rain Tomorrow", type="primary", use_container_width=True)

if predict_button:
    if not api_key:
        st.error("⚠️ Please enter your WeatherAPI.com API key in the sidebar")
    elif not location:
        st.error("⚠️ Please enter a location")
    else:
        with st.spinner("Fetching weather data..."):
            try:
                # Fetch current weather from WeatherAPI.com
                url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={location}&aqi=no"
                response = requests.get(url)
                response.raise_for_status()
                data = response.json()
                
                # Extract weather data
                current = data['current']
                location_info = data['location']
                
                # Prepare input data (matching training features)
                # Note: Some features might need estimation or default values
                input_data = {
                    'Min °C': current['temp_c'] - 5,  # Estimate (current temp - buffer)
                    'Max °C': current['temp_c'] + 5,  # Estimate (current temp + buffer)
                    'Rain(mm)': current['precip_mm'],
                    'Max wind gust Spd - km/h': current['gust_kph'],
                    'Temp °C- 9:00AM': current['temp_c'],  # Using current as proxy
                    'RH -% 9:00AM': current['humidity'],
                    'Spd - 9:00AM - km/h': current['wind_kph'],
                    'MSLP- hPa - 9:00AM': current['pressure_mb'],
                    'Temp °C- 3:00PM': current['temp_c'],
                    'RH -% 3:00PM': current['humidity'],
                    'Spd - 3:00PM - km/h': current['wind_kph'],
                    'MSLP- hPa - 3:00PM': current['pressure_mb'],
                }
                
                # Get month from location time
                month = datetime.now().strftime('%B')
                
                # Add categorical (State will be set as a default)
                categorical_data = {
                    'Month': month,
                    'State': 'NSW'  # Default - could be improved with geocoding
                }
                
                # Create DataFrame
                df_input = pd.DataFrame([{**input_data, **categorical_data}])
                
                # Fill missing with medians from training
                for col, median_val in preprocessing_info['numeric_medians'].items():
                    if col in df_input.columns and pd.isna(df_input[col].iloc[0]):
                        df_input[col] = median_val
                
                # One-hot encode categorical (same as training)
                df_encoded = pd.get_dummies(df_input, columns=['State', 'Month'], drop_first=True)
                
                # Ensure all training features are present
                for col in preprocessing_info['feature_names']:
                    if col not in df_encoded.columns:
                        df_encoded[col] = 0
                
                # Reorder columns to match training
                df_encoded = df_encoded[preprocessing_info['feature_names']]
                
                # Make prediction
                prediction = model.predict(df_encoded)[0]
                probability = model.predict_proba(df_encoded)[0, 1]
                
                # Display results
                st.success(f"✅ Weather data retrieved for **{location_info['name']}, {location_info['region']}, {location_info['country']}**")
                
                st.divider()
                
                # Create columns for layout
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Temperature", f"{current['temp_c']}°C")
                with col2:
                    st.metric("Humidity", f"{current['humidity']}%")
                with col3:
                    st.metric("Wind Speed", f"{current['wind_kph']} km/h")
                
                st.divider()
                
                # Risk assessment
                st.subheader("☔ Rain Prediction for Tomorrow")
                
                # Determine risk category
                if probability < 0.3:
                    risk_level = "LOW"
                    risk_color = "🟢"
                    recommendation = "Low risk of rain. Outdoor events are likely safe to proceed."
                elif probability < 0.6:
                    risk_level = "MODERATE"
                    risk_color = "🟡"
                    recommendation = "Moderate risk of rain. Have a backup plan ready."
                else:
                    risk_level = "HIGH"
                    risk_color = "🔴"
                    recommendation = "High risk of rain. Consider rescheduling or moving indoors."
                
                # Display prediction
                st.markdown(f"### {risk_color} {risk_level} RISK")
                st.progress(probability)
                st.metric(
                    "Probability of Rain Tomorrow",
                    f"{probability*100:.1f}%",
                    help="Based on current weather conditions"
                )
                
                # Recommendation
                st.info(f"**Recommendation:** {recommendation}")
                
                # Additional details in expander
                with st.expander("📊 Technical Details"):
                    st.write("**Model Prediction:**", "Rain" if prediction == 1 else "No Rain")
                    st.write("**Confidence Score:**", f"{probability:.3f}")
                    st.write("**Model Type:**", str(type(model).__name__))
                    
                    st.write("\n**Current Weather Conditions:**")
                    st.json({
                        "Temperature": f"{current['temp_c']}°C",
                        "Feels Like": f"{current['feelslike_c']}°C",
                        "Humidity": f"{current['humidity']}%",
                        "Pressure": f"{current['pressure_mb']} mb",
                        "Wind Speed": f"{current['wind_kph']} km/h",
                        "Wind Gust": f"{current['gust_kph']} km/h",
                        "Precipitation": f"{current['precip_mm']} mm",
                        "Conditions": current['condition']['text']
                    })
                
            except requests.exceptions.HTTPError as e:
                if response.status_code == 401:
                    st.error("❌ Invalid API key. Please check your WeatherAPI.com API key.")
                elif response.status_code == 400:
                    st.error(f"❌ Location '{location}' not found. Please try a different location.")
                else:
                    st.error(f"❌ API Error: {str(e)}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.write("Please check your inputs and try again.")

# Footer
st.divider()
st.caption("Built with TPOT AutoML | Optimized for minimizing missed rain events")
