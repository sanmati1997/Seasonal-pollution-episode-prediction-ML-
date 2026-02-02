import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Pollution Episode Predictor",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM CSS STYLING
st.markdown("""
<style>
    /* Main background and text */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #e94560 0%, #ff6b6b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        padding: 1rem 0;
        font-family: 'Segoe UI', sans-serif;
    }
    
    .sub-header {
        color: #a0a0a0;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Card styling */
    .prediction-card {
        background: linear-gradient(145deg, #1e1e30 0%, #2a2a4a 100%);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(145deg, #252545 0%, #1a1a35 100%);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #e94560;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Status badges */
    .status-severe {
        background: linear-gradient(90deg, #e94560 0%, #ff6b6b 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 50px;
        font-size: 1.5rem;
        font-weight: 700;
        text-align: center;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(233, 69, 96, 0.4);
    }
    
    .status-normal {
        background: linear-gradient(90deg, #00b894 0%, #00cec9 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 50px;
        font-size: 1.5rem;
        font-weight: 700;
        text-align: center;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(0, 184, 148, 0.4);
    }
    
    /* Section headers */
    .section-header {
        color: #e94560;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e94560;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(233, 69, 96, 0.1);
        border-left: 4px solid #e94560;
        padding: 1rem;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: rgba(255, 193, 7, 0.1);
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
    }
    
    /* Feature importance bars */
    .feature-bar {
        background: linear-gradient(90deg, #e94560 0%, #ff6b6b 100%);
        height: 20px;
        border-radius: 10px;
        margin: 5px 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Input styling */
    .stSelectbox > div > div {
        background-color: #252545;
        border: 1px solid #3a3a5a;
    }
    
    .stSlider > div > div {
        background-color: #252545;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #e94560 0%, #ff6b6b 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(233, 69, 96, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# CONSTANTS
CITIES = {
    'Delhi': {'lat': 28.6139, 'lon': 77.2090, 'state': 'Delhi'},
    'Lucknow': {'lat': 26.8467, 'lon': 80.9462, 'state': 'Uttar Pradesh'},
    'Patna': {'lat': 25.5941, 'lon': 85.1376, 'state': 'Bihar'},
    'Chandigarh': {'lat': 30.7333, 'lon': 76.7794, 'state': 'Chandigarh'},
    'Gurugram': {'lat': 28.4595, 'lon': 77.0266, 'state': 'Haryana'},
    'Jaipur': {'lat': 26.9124, 'lon': 75.7873, 'state': 'Rajasthan'},
    'Amritsar': {'lat': 31.6340, 'lon': 74.8723, 'state': 'Punjab'}
}

WINTER_MONTHS = {
    10: 'October',
    11: 'November', 
    12: 'December',
    1: 'January',
    2: 'February'
}

FEATURE_NAMES = [
    'month', 'temperature_c', 'humidity_pct', 'wind_speed_kmh',
    'wind_direction_deg', 'fire_count_punjab', 'fire_count_haryana',
    'days_from_diwali', 'previous_day_aqi', 'consecutive_poor_days',
    'weather_dispersion_index', 'stubble_impact_score', 'pollution_momentum',
    'diwali_impact_zone', 'inversion_risk'
]

# LOAD MODEL
@st.cache_resource
def load_model():
    """Load the trained model and model info"""
    model_path = 'models/best_model.pkl'
    info_path = 'models/model_info.pkl'
    
    model = None
    model_info = None
    
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    
    if os.path.exists(info_path):
        with open(info_path, 'rb') as f:
            model_info = pickle.load(f)
    
    return model, model_info

# HELPER FUNCTIONS
def engineer_features(inputs):
    """Calculate engineered features from raw inputs"""
    # Weather dispersion index
    weather_dispersion_index = round(
        inputs['wind_speed_kmh'] * (1 - inputs['humidity_pct'] / 100), 2
    )
    
    # Stubble impact score
    wind_towards_ncr = 1 if (inputs['wind_direction_deg'] >= 270 or 
                             inputs['wind_direction_deg'] <= 45) else 0
    stubble_impact_score = int(
        (inputs['fire_count_punjab'] + inputs['fire_count_haryana'] * 0.6) * 
        (1 + 0.25 * wind_towards_ncr)
    )
    
    # Pollution momentum
    pollution_momentum = int(
        inputs['previous_day_aqi'] * (1 + inputs['consecutive_poor_days'] * 0.06)
    )
    
    # Diwali impact zone
    diwali_impact_zone = 1 if abs(inputs['days_from_diwali']) <= 5 else 0
    
    # Inversion risk
    inversion_risk = 1 if (inputs['temperature_c'] < 15 and 
                           inputs['humidity_pct'] > 70 and 
                           inputs['wind_speed_kmh'] < 10) else 0
    
    return {
        'weather_dispersion_index': weather_dispersion_index,
        'stubble_impact_score': stubble_impact_score,
        'pollution_momentum': pollution_momentum,
        'diwali_impact_zone': diwali_impact_zone,
        'inversion_risk': inversion_risk
    }

def prepare_features(inputs):
    """Prepare feature vector for prediction"""
    engineered = engineer_features(inputs)
    
    features = {
        'month': inputs['month'],
        'temperature_c': inputs['temperature_c'],
        'humidity_pct': inputs['humidity_pct'],
        'wind_speed_kmh': inputs['wind_speed_kmh'],
        'wind_direction_deg': inputs['wind_direction_deg'],
        'fire_count_punjab': inputs['fire_count_punjab'],
        'fire_count_haryana': inputs['fire_count_haryana'],
        'days_from_diwali': inputs['days_from_diwali'],
        'previous_day_aqi': inputs['previous_day_aqi'],
        'consecutive_poor_days': inputs['consecutive_poor_days'],
        'weather_dispersion_index': engineered['weather_dispersion_index'],
        'stubble_impact_score': engineered['stubble_impact_score'],
        'pollution_momentum': engineered['pollution_momentum'],
        'diwali_impact_zone': engineered['diwali_impact_zone'],
        'inversion_risk': engineered['inversion_risk']
    }
    
    return pd.DataFrame([features])[FEATURE_NAMES]

def get_risk_factors(inputs, engineered):
    """Identify key risk factors based on inputs"""
    risks = []
    
    if inputs['wind_speed_kmh'] < 8:
        risks.append("🌬️ Low wind speed limiting pollutant dispersion")
    if inputs['humidity_pct'] > 75:
        risks.append("💧 High humidity trapping pollutants")
    if inputs['temperature_c'] < 12:
        risks.append("🌡️ Low temperature causing thermal inversion")
    if inputs['fire_count_punjab'] > 500:
        risks.append("🔥 High stubble burning activity in Punjab")
    if inputs['fire_count_haryana'] > 200:
        risks.append("🔥 Significant fire activity in Haryana")
    if inputs['previous_day_aqi'] > 250:
        risks.append("📊 High previous day AQI indicating pollution buildup")
    if inputs['consecutive_poor_days'] >= 3:
        risks.append("📈 Multiple consecutive poor air quality days")
    if abs(inputs['days_from_diwali']) <= 5:
        risks.append("🎆 Near Diwali period with expected firework emissions")
    if engineered['inversion_risk'] == 1:
        risks.append("⚠️ Atmospheric inversion conditions detected")
    
    return risks

def get_recommendations(is_severe, probability):
    """Get health recommendations based on prediction"""
    if is_severe:
        return [
            "😷 Wear N95 masks when outdoors",
            "🏠 Stay indoors as much as possible",
            "🚫 Avoid outdoor exercise and physical activities",
            "💨 Use air purifiers indoors",
            "🚗 Keep car windows closed while driving",
            "👶 Keep children and elderly indoors",
            "🏥 Those with respiratory conditions should have medications ready"
        ]
    else:
        if probability > 0.4:
            return [
                "😷 Consider wearing a mask during peak traffic hours",
                "🏃 Limit strenuous outdoor activities",
                "👀 Monitor air quality throughout the day",
                "💨 Keep indoor spaces well-ventilated"
            ]
        else:
            return [
                "✅ Air quality expected to be manageable",
                "🏃 Outdoor activities can proceed with caution",
                "👀 Continue monitoring air quality updates",
                "🌿 Good day for outdoor activities with basic precautions"
            ]

# MAIN APPLICATION
def main():
    # Load model
    model, model_info = load_model()
    
    # Header
    st.markdown('<h1 class="main-header">🌫️ Pollution Episode Predictor</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered prediction system for severe pollution episodes in North Indian cities</p>', 
                unsafe_allow_html=True)
    
    # Check if model is loaded
    if model is None:
        st.error("⚠️ Model not found! Please ensure `models/best_model.pkl` exists.")
        st.info("Run the training notebook (02_Model_Training.ipynb) to generate the model.")
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎛️ Configuration")
        st.markdown("---")
        
        # City selection
        st.markdown("#### 🏙️ Location")
        selected_city = st.selectbox(
            "Select City",
            options=list(CITIES.keys()),
            index=0,
            help="Choose a city for prediction"
        )
        
        city_info = CITIES[selected_city]
        st.caption(f"📍 {city_info['state']} | Lat: {city_info['lat']:.2f}, Lon: {city_info['lon']:.2f}")
        
        st.markdown("---")
        
        # Temporal inputs
        st.markdown("#### 📅 Temporal Factors")
        
        month = st.selectbox(
            "Month",
            options=list(WINTER_MONTHS.keys()),
            format_func=lambda x: WINTER_MONTHS[x],
            index=1,
            help="Select the month (Winter season: Oct-Feb)"
        )
        
        days_from_diwali = st.slider(
            "Days from Diwali",
            min_value=-30,
            max_value=60,
            value=5,
            help="Negative = before Diwali, Positive = after Diwali"
        )
        
        st.markdown("---")
        
        # About section
        st.markdown("#### ℹ️ About")
        st.markdown("""
        This application predicts severe pollution episodes 
        using a **Tuned Decision Tree** model trained on 
        weather, fire activity, and historical pollution data.
        
        **Model Performance:**
        - Accuracy: ~55%
        - Recall: ~56%
        - F1-Score: 0.53
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<p class="section-header">🌤️ Weather Conditions</p>', 
                    unsafe_allow_html=True)
        
        temperature_c = st.slider(
            "Temperature (°C)",
            min_value=0.0,
            max_value=40.0,
            value=18.0,
            step=0.5,
            help="Current temperature in Celsius"
        )
        
        humidity_pct = st.slider(
            "Humidity (%)",
            min_value=20.0,
            max_value=100.0,
            value=65.0,
            step=1.0,
            help="Relative humidity percentage"
        )
        
        wind_speed_kmh = st.slider(
            "Wind Speed (km/h)",
            min_value=0.0,
            max_value=40.0,
            value=10.0,
            step=0.5,
            help="Wind speed in kilometers per hour"
        )
        
        wind_direction_deg = st.slider(
            "Wind Direction (°)",
            min_value=0,
            max_value=360,
            value=180,
            step=5,
            help="Wind direction in degrees (0°=N, 90°=E, 180°=S, 270°=W)"
        )
    
    with col2:
        st.markdown('<p class="section-header">🔥 Fire Activity & History</p>', 
                    unsafe_allow_html=True)
        
        fire_count_punjab = st.slider(
            "Fire Count - Punjab",
            min_value=0,
            max_value=2500,
            value=400,
            step=10,
            help="Number of active fires detected in Punjab"
        )
        
        fire_count_haryana = st.slider(
            "Fire Count - Haryana",
            min_value=0,
            max_value=1200,
            value=150,
            step=10,
            help="Number of active fires detected in Haryana"
        )
        
        previous_day_aqi = st.slider(
            "Previous Day AQI",
            min_value=50,
            max_value=500,
            value=180,
            step=5,
            help="Air Quality Index of the previous day"
        )
        
        consecutive_poor_days = st.slider(
            "Consecutive Poor AQI Days",
            min_value=0,
            max_value=10,
            value=2,
            step=1,
            help="Number of consecutive days with poor air quality"
        )
    
    st.markdown("---")
    
    # Prediction button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_button = st.button("🔮 Predict Pollution Episode", use_container_width=True)
    
    if predict_button:
        # Collect inputs
        inputs = {
            'month': month,
            'temperature_c': temperature_c,
            'humidity_pct': humidity_pct,
            'wind_speed_kmh': wind_speed_kmh,
            'wind_direction_deg': wind_direction_deg,
            'fire_count_punjab': fire_count_punjab,
            'fire_count_haryana': fire_count_haryana,
            'days_from_diwali': days_from_diwali,
            'previous_day_aqi': previous_day_aqi,
            'consecutive_poor_days': consecutive_poor_days
        }
        
        # Engineer features
        engineered = engineer_features(inputs)
        
        # Prepare features
        X = prepare_features(inputs)
        
        # Make prediction
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]
        
        is_severe = prediction == 1
        prob_severe = probability[1]
        
        st.markdown("---")
        
        # Results section
        st.markdown("## 📊 Prediction Results")
        
        # Main prediction display
        result_col1, result_col2, result_col3 = st.columns([1, 2, 1])
        
        with result_col2:
            if is_severe:
                st.markdown("""
                <div style="text-align: center; padding: 2rem;">
                    <div class="status-severe">⚠️ SEVERE EPISODE PREDICTED</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="text-align: center; padding: 2rem;">
                    <div class="status-normal">✅ NORMAL CONDITIONS EXPECTED</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Metrics row
        st.markdown("### 📈 Prediction Metrics")
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric(
                label="Probability (Severe)",
                value=f"{prob_severe:.1%}",
                delta=None
            )
        
        with metric_col2:
            st.metric(
                label="Confidence Level",
                value=f"{max(prob_severe, 1-prob_severe):.1%}",
                delta=None
            )
        
        with metric_col3:
            st.metric(
                label="City",
                value=selected_city,
                delta=None
            )
        
        with metric_col4:
            risk_level = "High" if prob_severe > 0.6 else "Medium" if prob_severe > 0.4 else "Low"
            st.metric(
                label="Risk Level",
                value=risk_level,
                delta=None
            )
        
        # Two columns for details
        detail_col1, detail_col2 = st.columns(2)
        
        with detail_col1:
            st.markdown("### 🔍 Engineered Features")
            st.markdown("""
            <div class="prediction-card">
            """, unsafe_allow_html=True)
            
            eng_df = pd.DataFrame([{
                'Feature': 'Weather Dispersion Index',
                'Value': f"{engineered['weather_dispersion_index']:.2f}"
            }, {
                'Feature': 'Stubble Impact Score',
                'Value': f"{engineered['stubble_impact_score']}"
            }, {
                'Feature': 'Pollution Momentum',
                'Value': f"{engineered['pollution_momentum']}"
            }, {
                'Feature': 'Diwali Impact Zone',
                'Value': 'Yes' if engineered['diwali_impact_zone'] else 'No'
            }, {
                'Feature': 'Inversion Risk',
                'Value': 'Yes' if engineered['inversion_risk'] else 'No'
            }])
            
            st.dataframe(eng_df, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with detail_col2:
            st.markdown("### ⚠️ Risk Factors Identified")
            
            risks = get_risk_factors(inputs, engineered)
            
            if risks:
                for risk in risks:
                    st.markdown(f"• {risk}")
            else:
                st.markdown("✅ No significant risk factors identified")
        
        # Recommendations
        st.markdown("### 💡 Health Recommendations")
        
        recommendations = get_recommendations(is_severe, prob_severe)
        
        rec_cols = st.columns(2)
        for i, rec in enumerate(recommendations):
            with rec_cols[i % 2]:
                st.markdown(f"• {rec}")
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            st.markdown("### 📊 Feature Importance")
            
            importance_df = pd.DataFrame({
                'Feature': FEATURE_NAMES,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=True).tail(10)
            
            # Create horizontal bar chart
            import plotly.express as px
            
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                color='Importance',
                color_continuous_scale='Reds'
            )
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#888',
                showlegend=False,
                height=400,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            
            fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
            fig.update_yaxes(showgrid=False)
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🌫️ <strong>Seasonal Pollution Episode Prediction System</strong></p>
        <p style="font-size: 0.8rem;">
            Built with Streamlit | Model: Tuned Decision Tree | 
            Data: North Indian Cities (2019-2024)
        </p>
    </div>
    """, unsafe_allow_html=True)

# RUN APPLICATION
if __name__ == "__main__":
    main()

