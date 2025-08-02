
import pandas as pd
import streamlit as st
import joblib
import time
import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

# --- Page Configuration and Styling ---
st.set_page_config(
    page_title="Residential Energy Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS 
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styling */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main app background with image from a web URL */

    .stApp {
        background-image: url('https://cowdenelectric.com/wp-content/uploads/2017/06/residential-electric.jpg');
        background-size: cover;          /* Fit image without stretching */
        background-repeat: no-repeat;      /* Show image only once */
        background-position: top center;   /* Position it nicely */
        background-attachment: scroll;     /* Scrolls with the page */
        background-color: #ffffff;         /* Fallback or bottom color */
        color: #333;
    }



    /* Overlay to improve text readability */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(255, 255, 255, 0.95); /* Increased opacity for better text readability */
        z-index: -1;
    }

    /* Style for elements to ensure they are above the overlay */
    .modern-card, .input-group, .metric-card, .info-badge, .insight-card, .comparison-box, .app-footer, .sidebar-brand {
        background-color: rgba(255, 255, 255, 0.95) !important;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    .main-title, .page-header {
        background-color: rgba(255, 255, 255, 0.95) !important;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    
    /* Specific styling for section headers */
    h3 {
        color: #2c3e50;
        font-weight: 600;
        margin-bottom: 1.5rem;
        font-size: 1.3rem;
    }

    /* Sidebar modern styling */
    .css-1d391kg, .stSidebar {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
        border-right: 3px solid #3498db;
        color: white; /* Ensure sidebar text is readable */
    }
    
    .stSidebar .stMarkdown {
        color: white;
    }
    
    /* Custom navigation buttons */
    .nav-btn {
        background: linear-gradient(135deg, #3498db, #2980b9);
        border: none;
        border-radius: 12px;
        color: white;
        padding: 14px 20px;
        margin: 8px 0;
        font-size: 16px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
        text-align: left;
        box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);
        background-color: transparent !important;
        border: 1px solid #3498db;
    }
    
    .nav-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(52, 152, 219, 0.4);
        background: linear-gradient(135deg, #2980b9, #1f4e79);
        color: white;
    }

    .nav-btn.active {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        box-shadow: 0 6px 20px rgba(231, 76, 60, 0.4);
        color: white;
        border-color: #c0392b;
    }

    /* Main title */
    .main-title {
        background: black;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        margin: 2rem 0;
        letter-spacing: -1px;
    }

    /* Page headers */
    .page-header {
        color: #2c3e50;
        font-size: 2.5rem;
        font-weight: 600;
        text-align: center;
        margin: 2rem 0;
        position: relative;
    }

    .page-header::after {
        content: '';
        position: absolute;
        bottom: -10px;
        left: 50%;
        transform: translateX(-50%);
        width: 100px;
        height: 4px;
        background: linear-gradient(135deg, #3498db, #9b59b6);
        border-radius: 2px;
    }

    /* Modern cards */
    .modern-card {
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .modern-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(135deg, #3498db, #9b59b6);
        border-radius: 2px;
    }

    .modern-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15);
    }

    /* Input sections */
    .input-group {
        background: rgba(248, 250, 252, 0.95) !important;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        border-left: 5px solid #3498db;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
    }

    .input-group:hover {
        border-left-color: #e74c3c;
        box-shadow: 0 5px 20px rgba(52, 152, 219, 0.1);
    }

    .input-group h3 {
        color: #2c3e50;
        font-weight: 600;
        margin-bottom: 1.5rem;
        font-size: 1.3rem;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        transition: all 0.3s ease;
        transform: scale(0);
    }

    .metric-card:hover::before {
        transform: scale(1);
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);
    }

    /* Success metric variant */
    .success-metric {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        box-shadow: 0 8px 30px rgba(46, 204, 113, 0.3);
    }

    .success-metric:hover {
        box-shadow: 0 15px 40px rgba(46, 204, 113, 0.4);
    }

    /* Warning metric variant */
    .warning-metric {
        background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
        box-shadow: 0 8px 30px rgba(243, 156, 18, 0.3);
    }

    .warning-metric:hover {
        box-shadow: 0 15px 40px rgba(243, 156, 18, 0.4);
    }

    /* Predict button */
    .predict-btn {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 1rem 3rem;
        font-size: 1.2rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 8px 25px rgba(231, 76, 60, 0.3);
        width: 100%;
    }

    .predict-btn:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(231, 76, 60, 0.4);
    }

    /* Info badges */
    .info-badge {
        background: linear-gradient(135deg, #3498db, #2980b9);
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 25px;
        font-weight: 500;
        margin: 0.5rem 0;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(52, 152, 219, 0.2);
    }

    /* Dashboard insights */
    .insight-card {
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #3498db;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
    }

    .insight-card:hover {
        border-left-color: #e74c3c;
        transform: translateX(5px);
    }

    .insight-card h4 {
        color: #2c3e50;
        margin-bottom: 1rem;
        font-weight: 600;
    }

    .insight-card p, .insight-card li {
        color: #5a6c7d;
        line-height: 1.6;
    }

    /* Sidebar branding */
    .sidebar-brand {
        background: linear-gradient(135deg, #3498db, #9b59b6);
        color: white;
        padding: 1.5rem;
        text-align: center;
        border-radius: 15px;
        margin: 1rem 0;
        font-weight: 600;
    }

    /* Footer styling */
    .app-footer {
        border-radius: 15px;
        padding: 2rem;
        margin-top: 3rem;
        text-align: center;
        color: #5a6c7d;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.08);
    }

    .app-footer h4 {
        color: #2c3e50;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# --- Load Models and State Management ---
@st.cache_resource
def load_model():
    """Loads the single Random Forest machine learning model."""
    try:
        model = joblib.load('Random_forest_model (2).pkl')
        return model
    except FileNotFoundError:
        st.error("Model file 'Random_forest_model (2).pkl' not found. Please ensure it is in the same directory.")
        return None

# Load the model once and cache it
model = load_model()
if not model:
    st.stop()

# Define model columns in global scope
model_columns = [
    'num_occupants', 'house_size_sqft', 'monthly_income', 'outside_temp_celsius', 'year', 'month', 'day',
    'season', 'heating_type_Electric', 'heating_type_Gas', 'heating_type_None',
    'cooling_type_AC', 'cooling_type_Fan', 'cooling_type_None',
    'manual_override_Y', 'manual_override_N',
    'is_weekend', 'temp_above_avg', 'income_per_person', 'square_feet_per_person',
    'high_income_flag', 'low_temp_flag', 
    'season_spring', 'season_summer', 'season_fall', 'season_winter', 
    'day_of_week_0', 'day_of_week_6', 'energy_star_home'
]


# Initialize session state for multi-page navigation and history
if 'page' not in st.session_state:
    st.session_state.page = 'Home'
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []

# --- Sidebar Navigation ---
st.sidebar.markdown(f'<div class="sidebar-brand">🏠 Energy Predictor</div>', unsafe_allow_html=True)

nav_options = {
    'Home': '🏠 Home',
    'Predictor': '📈 New Prediction',
    'Dashboard': '📊 Prediction Dashboard',
    'About': 'ℹ️ About'
}

with st.sidebar:
    st.markdown("### Navigation")
    for page_name, label in nav_options.items():
        active_class = "active" if st.session_state.page == page_name else ""
        if st.button(label, key=f"nav_{page_name}", help=f"Go to {page_name}"):
            st.session_state.page = page_name
            st.rerun()

    # Add separator for spacing
    st.markdown("---")

    # 🗑️ Clear History section after navigation
    with st.expander("🗑️ Clear Prediction History", expanded=False):
        if st.session_state.predictions_history:
            confirm_clear = st.checkbox("Confirm clear history", key="sidebar_confirm")
            if confirm_clear and st.button("Clear History", key="sidebar_clear"):
                st.session_state.predictions_history.clear()
                st.success("✅ Prediction history cleared.")
                st.rerun()
        else:
            st.info("No history to clear.")

# --- Home Page ---
def show_home():
    st.markdown(f'<h1 class="main-title">Residential Energy Consumption Predictor</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="page-header">Predict, Analyze, and Optimize your Home\'s Energy Usage</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="modern-card">
            <h3>⚡ Welcome!</h3>
            <p>This application is a machine learning-powered tool for predicting residential energy consumption based on various factors. By entering key details about your residence and the environment, our model will provide a data-driven forecast.</p>
            <p>This tool is perfect for homeowners looking to understand and manage their energy usage more effectively.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="modern-card">
            <h3>📈 How It Works</h3>
            <p>Navigate to the 'New Prediction' page to input your home's details. The app will use a sophisticated 'Random Forest' model to generate a prediction.</p>
            <p>Your predictions are saved, allowing you to view them on the **Prediction Dashboard** for deeper insights and trend analysis.</p>
        </div>
        """, unsafe_allow_html=True)


# --- Predictor Page ---
def show_predictor():
    st.markdown(f'<h2 class="page-header">New Energy Consumption Prediction</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div class="modern-card">
        <p>Please enter the details below to get a prediction of your home's energy consumption.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("### Home & Occupancy Details")
        col1, col2 = st.columns(2)
        with col1:
            num_occupants = st.number_input("Number of Occupants", min_value=1, value=4, step=1, help="Number of people living in the house.")
            house_size_sqft = st.number_input("House Size (sqft)", min_value=500, value=2500, step=100, help="Total square footage of the home.")
        with col2:
            monthly_income = st.number_input("Monthly Income ($)", min_value=1000, value=7500, step=500, help="Household's approximate monthly income.")
            energy_star_home = st.checkbox("Is it an Energy Star Home?", value=False, help="Check if the home has an Energy Star rating.")

    with st.container(border=True):
        st.markdown("### Environmental Factors")
        col1, col2, col3 = st.columns(3)
        with col1:
            outside_temp_celsius = st.number_input("Outside Temperature (Celsius)", min_value=-20, max_value=40, value=20, step=1, help="Current outdoor temperature in Celsius.")
        with col2:
            heating_type = st.selectbox("Heating Type", ["Electric", "Gas", "None"], help="Primary heating source for the home.")
        with col3:
            cooling_type = st.selectbox("Cooling Type", ["AC", "Fan", "None"], help="Primary cooling source for the home.")

    with st.container(border=True):
        st.markdown("### Time & Manual Settings")
        col1, col2 = st.columns(2)
        with col1:
            date_input = st.date_input("Date", datetime.date.today(), help="The date for which you want to predict consumption.")
            manual_override = st.selectbox("Manual Override", ["No", "Yes"], help="Is there an active manual override on the thermostat?")
        with col2:
            st.empty() 
    
    # Derived parameters logic
    try:
        year = date_input.year
        month = date_input.month
        day = date_input.day
        day_of_week = date_input.weekday()
        
        if month in [12, 1, 2]: season_label = "Winter"
        elif month in [3, 4, 5]: season_label = "Summer"
        elif month in [6, 7, 8]: season_label = "Rainy"
        else: season_label = "Spring"
        
        st.markdown(f'<div class="info-badge">The day is a **{date_input.strftime("%A")}** in the **{season_label}** season.</div>', unsafe_allow_html=True)
    except ValueError:
        st.error("Invalid date entered. Please check the year, month, and day.")
        return
    
    is_weekend = int(day_of_week >= 5)
    temp_above_avg = int(outside_temp_celsius > 28)
    income_per_person = monthly_income / num_occupants
    square_feet_per_person = house_size_sqft / num_occupants
    high_income_flag = int(monthly_income > 40000)
    low_temp_flag = int(outside_temp_celsius < 28)

    season_map = {'Winter': 1, 'Summer': 4, 'Rainy': 2, 'Spring': 3}
    
    # Pre-processing input data for the model
    input_data = {
        'num_occupants': num_occupants, 'house_size_sqft': house_size_sqft, 'monthly_income': monthly_income,
        'outside_temp_celsius': outside_temp_celsius, 'year': year, 'month': month, 'day': day,
        'season': season_map[season_label], 
        'heating_type_Electric': int(heating_type == "Electric"), 'heating_type_Gas': int(heating_type == "Gas"), 
        'heating_type_None': int(heating_type == "None"), 'cooling_type_AC': int(cooling_type == "AC"), 
        'cooling_type_Fan': int(cooling_type == "Fan"), 'cooling_type_None': int(cooling_type == "None"),
        'manual_override_N': int(manual_override == "No"), 'manual_override_Y': int(manual_override == "Yes"),
        'is_weekend': is_weekend, 'temp_above_avg': temp_above_avg, 'income_per_person': income_per_person,
        'square_feet_per_person': square_feet_per_person, 'high_income_flag': high_income_flag, 
        'low_temp_flag': low_temp_flag, 
        'season_spring': int(season_label == "Spring"), 'season_summer': int(season_label == "Summer"),
        'season_fall': int(season_label == "Rainy"), 'season_winter': int(season_label == "Winter"),
        'day_of_week_0': int(day_of_week == 0), 'day_of_week_6': int(day_of_week == 6), 
        'energy_star_home': energy_star_home
    }


# Initialize state variables if not already set
    if "prediction_phase" not in st.session_state:
        st.session_state.prediction_phase = "idle"  

    if st.button("Predict Energy Consumption", key="predict_btn") and st.session_state.prediction_phase == "idle":
        st.session_state.prediction_phase = "predicting"
        st.rerun()

# Phase 1: Predicting
    if st.session_state.prediction_phase == "predicting":
        with st.spinner("Predicting..."):
            time.sleep(2)  # Simulate prediction delay
            try:
                input_df = pd.DataFrame([input_data])
                input_df = input_df[model_columns]
                prediction = model.predict(input_df)[0]

                st.session_state.predicted_value = prediction  # Store for later display
                st.session_state.prediction_timestamp = datetime.datetime.now()

            # Save prediction to history
                st.session_state.predictions_history.append({
                    'Timestamp': st.session_state.prediction_timestamp,
                    'Date': date_input,
                    'Temperature (C)': outside_temp_celsius,
                    'Occupants': num_occupants,
                    'House Size (sqft)': house_size_sqft,
                    'Heating Type': heating_type,
                    'Cooling Type': cooling_type,
                    'Prediction (kWh)': prediction
                })

                st.session_state.prediction_phase = "done"
                st.rerun()

            except Exception as e:
                st.session_state.prediction_phase = "idle"
                st.error(f"An error occurred during prediction: {e}")

# Phase 2: Show result and start navigation spinner
    elif st.session_state.prediction_phase == "done":
        st.success(f"Predicted Energy Consumption: **{st.session_state.predicted_value:.2f} kWh**")

        with st.spinner("Navigating to dashboard..."):
            time.sleep(3)
            st.session_state.page = 'Dashboard'
            st.session_state.prediction_phase = "idle"
            st.rerun()



# --- Dashboard Page ---
def show_dashboard():
    st.markdown(f'<h2 class="page-header">Prediction Dashboard & History</h2>', unsafe_allow_html=True)
    
    if not st.session_state.predictions_history:
        st.info("No predictions yet. Go to the 'New Prediction' page to start forecasting.")
        return

    history_df = pd.DataFrame(st.session_state.predictions_history)
    history_df['Date'] = pd.to_datetime(history_df['Date'])
    history_df.set_index('Date', inplace=True)
    
    st.markdown("### Recent Predictions")
    st.dataframe(history_df, use_container_width=True)

    st.markdown("---")
    
    st.markdown("### Key Metrics from History")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_consumption = history_df['Prediction (kWh)'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h4>Average Consumption</h4>
            <h2>{avg_consumption:,.2f} kWh</h2>
            <p>Mean of all past predictions</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        max_consumption = history_df['Prediction (kWh)'].max()
        st.markdown(f"""
        <div class="metric-card success-metric">
            <h4>Peak Consumption</h4>
            <h2>{max_consumption:,.2f} kWh</h2>
            <p>Highest predicted energy usage</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        avg_per_sqft = (history_df['Prediction (kWh)'] / history_df['House Size (sqft)']).mean()
        st.markdown(f"""
        <div class="metric-card warning-metric">
            <h4>Avg per SqFt</h4>
            <h2>{avg_per_sqft:,.2f} kWh/sqft</h2>
            <p>Avg consumption per square foot</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    st.markdown("### Prediction Justification & Influential Factors")
    if model and hasattr(model, 'feature_importances_'):
        # Get feature importances from the model
        feature_importances = model.feature_importances_
        feature_names = model_columns
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)
        
        # Plot feature importance
        fig_importance = px.bar(
            importance_df.head(10), 
            x='Importance', 
            y='Feature', 
            orientation='h', 
            title='Top 10 Most Important Features for Prediction'
        )
        fig_importance.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#333'),
            yaxis={'categoryorder':'total ascending'}
        )
        st.plotly_chart(fig_importance, use_container_width=True)
    else:
        st.info("Feature importance data is not available for this model.")
    
    st.markdown("---")
    
    st.markdown("### Trends & Correlations")
    col1, col2 = st.columns(2)
    with col1:
        fig_temp = px.scatter(
            history_df,
            x='Temperature (C)',
            y='Prediction (kWh)',
            title='Predicted Consumption vs. Temperature',
            trendline='ols'
        )
        fig_temp.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#333')
        )
        st.plotly_chart(fig_temp, use_container_width=True)
    
    with col2:
        fig_size = px.scatter(
            history_df,
            x='House Size (sqft)',
            y='Prediction (kWh)',
            title='Predicted Consumption vs. House Size',
            trendline='ols'
        )
        fig_size.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#333')
        )
        st.plotly_chart(fig_size, use_container_width=True)

# --- About Page ---
def show_about():
    st.markdown(f'<h2 class="page-header">About This Project</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div class="modern-card">
        <h3>Project Overview</h3>
        <p>This application is a machine learning-powered tool for predicting residential energy consumption. The goal is to provide homeowners and analysts with an intuitive way to forecast energy usage and understand the key drivers behind it.</p>
        <p>By leveraging a variety of features, including household characteristics, environmental data, and time-based factors, the app offers a robust forecasting solution.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="modern-card">
        <h3>The Model: Random Forest</h3>
        <p>The core of this application is a "Random Forest Regressor" model. Here's why it's a great choice for this task:</p>
        <ul>
            <li><strong>Accuracy:</strong> It's an ensemble model that combines the predictions of many decision trees, which often leads to highly accurate results.</li>
            <li><strong>Robustness:</strong> It is less prone to overfitting and can handle complex, non-linear relationships between features.</li>
            <li><strong>Feature Importance:</strong> The model can calculate which features are most influential in its predictions, providing valuable insights, which you can see in the Dashboard.</li>
        </ul>
        <p>The model was trained on a comprehensive dataset of residential energy consumption to learn the patterns and relationships between the input features and the resulting energy usage.</p>
    </div>
    """, unsafe_allow_html=True)


# --- Main App Logic ---
if st.session_state.page == 'Home':
    show_home()
elif st.session_state.page == 'Predictor':
    show_predictor()
elif st.session_state.page == 'Dashboard':
    show_dashboard()
elif st.session_state.page == 'About':
    show_about()

# Footer
st.markdown("---")
st.markdown(f"""
<div class="app-footer">
    <h4>Residential Energy Predictor</h4>
    <p>Developed with Streamlit and powered by Machine Learning.</p>
</div>
""", unsafe_allow_html=True)
