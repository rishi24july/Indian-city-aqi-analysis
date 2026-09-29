"""
Interactive Streamlit Dashboard: Air Quality & Root Cause Analysis
Run via: streamlit run air_quality_analysis/app.py
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="India City AQI & Root Cause Analysis",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants & Paths
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, 'city_day.csv')
CHARTS_DIR = os.path.join(BASE_DIR, 'charts')

CPCB_SAFE_LIMITS = {
    'PM2.5': 60.0,
    'PM10': 100.0,
    'NO2': 80.0,
    'SO2': 80.0,
    'CO': 2.0,
    'O3': 100.0,
    'NH3': 400.0
}

AQI_COLORS = {
    'Good': '#009966',          # 0-50
    'Satisfactory': '#84cf33',  # 51-100
    'Moderate': '#ffde33',      # 101-200
    'Poor': '#ff9933',          # 201-300
    'Very Poor': '#cc0033',     # 301-400
    'Severe': '#7e0023'         # 401+
}

CITY_DIAGNOSTICS = {
    'Ahmedabad': {
        'status': 'Severe Industrial & Chemical Load',
        'causes': [
            "Vatva, Naroda aur Odhav industrial estates ke chemical aur textile manufacturing units.",
            "Thermal coal power plants aur high sulfur diesel combustion.",
            "Heavy vehicular freight traffic on state highways."
        ],
        'primary': 'SO2, CO & PM'
    },
    'Delhi': {
        'status': 'Chronic Severe Smog & Vehicular Load',
        'causes': [
            "10+ million registered vehicles se non-stop NO2 aur particulate emissions.",
            "October-November stubble (parali) burning in neighboring states (Punjab, Haryana).",
            "Winter meteorological inversion: Landlocked terrain hawa ko trap kar leta hai.",
            "Extensive construction & unpaved road dust."
        ],
        'primary': 'PM2.5, PM10 & NO2'
    },
    'Patna': {
        'status': 'Basin Dust & Biomass Trapping',
        'causes': [
            "Indo-Gangetic Basin topography: River Ganga basin moisture aur dust ko trap karti hai.",
            "Unpaved roads aur non-mechanized construction dust.",
            "Domestic chulha (wood/coal/cow-dung) cooking emissions in peri-urban areas."
        ],
        'primary': 'PM2.5 & PM10'
    },
    'Gurugram': {
        'status': 'Rapid Urbanization & Transit Corridors',
        'causes': [
            "Massive residential & commercial real-estate construction dust.",
            "NH-48 express transit trucks passing overnight.",
            "Proximity to Aravalli belt soil and semi-arid dust storms."
        ],
        'primary': 'PM10 & NO2'
    },
    'Lucknow': {
        'status': 'High Density Vehicular Congestion',
        'causes': [
            "Narrow arterial roads aur high vehicle-to-road ratio leading to persistent idling.",
            "Biomass waste burning during winter months.",
            "Regional transboundary smog from Western UP."
        ],
        'primary': 'PM2.5 & CO'
    },
    'Kolkata': {
        'status': 'Diesel Fleet & Riverine Humidity',
        'causes': [
            "Older commercial diesel transport (taxis, buses, delivery trucks).",
            "High humidity particulate pollutants ko heavy bana kar ground level pe rokti hai.",
            "Kolkata Port cargo handling & heavy logistics."
        ],
        'primary': 'NO2 & PM2.5'
    },
    'Mumbai': {
        'status': 'Coastal Buffer with Construction Spikes',
        'causes': [
            "Sea breeze (land-sea circulation) pollution ko disperse karne mein madad karti hai.",
            "Lekin massive ongoing infrastructure/metro construction dust PM10 badhati hai.",
            "High vehicular density on Western/Eastern Express highways."
        ],
        'primary': 'NO2 & PM10'
    },
    'Bengaluru': {
        'status': 'Traffic Bottlenecks in IT Corridors',
        'causes': [
            "High greenery aur 900m elevation air quality ko moderate rakhte hain.",
            "Lekin Silk Board, Whitefield aur ORR jaise tech corridors mein severe traffic idling se NO2 spike hota hai.",
            "Road dust resuspension during dry winter months."
        ],
        'primary': 'NO2 & PM10'
    },
    'Aizawl': {
        'status': 'Clean Air Benchmark',
        'causes': [
            "High forest cover (>85%) acting as natural carbon and particulate sink.",
            "Hilly terrain, absence of heavy polluting industries.",
            "Strict vehicle fitness and low vehicular density."
        ],
        'primary': 'Clean (All pollutants under CPCB limits)'
    },
    'Shillong': {
        'status': 'Clean Air Benchmark',
        'causes': [
            "Abundant rainfall (Pine belt) which washes down suspended particles.",
            "Low industrialization and high green cover."
        ],
        'primary': 'Clean'
    }
}

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Month_Name'] = df['Date'].dt.strftime('%b')
    return df

df = load_data()

# Header
st.title("🇮🇳 Indian Cities Air Quality & Root Cause Analysis")
st.markdown("""
**Objective:** Kis city me kitna air pollution hai aur kis vajah se hai?  
*Data Source: Central Pollution Control Board (CPCB) | 26 Major Indian Cities (2015-2020)*
""")
st.divider()

# Sidebar
st.sidebar.header("🔍 Filters & Navigation")
all_cities = sorted(df['City'].unique())
selected_city = st.sidebar.selectbox("Select a City for Detailed Diagnosis:", ["-- All Cities Overview --"] + all_cities)

year_list = ["All Years"] + sorted(df['Year'].dropna().unique().astype(int).tolist())
selected_year = st.sidebar.selectbox("Filter by Year:", year_list)

filtered_df = df.copy()
if selected_year != "All Years":
    filtered_df = filtered_df[filtered_df['Year'] == selected_year]

# Helper function
def get_aqi_category(aqi):
    if pd.isna(aqi): return "Unknown", "#888888"
    if aqi <= 50: return "Good", AQI_COLORS['Good']
    elif aqi <= 100: return "Satisfactory", AQI_COLORS['Satisfactory']
    elif aqi <= 200: return "Moderate", AQI_COLORS['Moderate']
    elif aqi <= 300: return "Poor", AQI_COLORS['Poor']
    elif aqi <= 400: return "Very Poor", AQI_COLORS['Very Poor']
    else: return "Severe", AQI_COLORS['Severe']

# MAIN VIEW
if selected_city == "-- All Cities Overview --":
    st.subheader("📊 National Air Quality Overview Across 26 Major Cities")
    
    # Overview KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    avg_nat_aqi = filtered_df['AQI'].mean()
    cat, color = get_aqi_category(avg_nat_aqi)
    
    col1.metric("All-India Average AQI", f"{avg_nat_aqi:.1f}", delta=f"Status: {cat}")
    
    city_group = filtered_df.groupby('City')['AQI'].mean().sort_values(ascending=False)
    col2.metric("Most Polluted City", f"{city_group.index[0]}", f"AQI: {city_group.iloc[0]:.1f}")
    col3.metric("Cleanest City", f"{city_group.index[-1]}", f"AQI: {city_group.iloc[-1]:.1f}")
    col4.metric("Total Monitored Days", f"{len(filtered_df.dropna(subset=['AQI'])):,}")
    
    tab1, tab2, tab3 = st.tabs(["🏆 City Rankings", "🔥 Root Causes & Correlations", "🤖 ML Driver Model"])
    
    with tab1:
        st.markdown("### Top 10 Most Polluted vs 10 Cleanest Cities")
        chart1_file = os.path.join(CHARTS_DIR, '1_city_aqi_ranking.png')
        if os.path.exists(chart1_file):
            st.image(chart1_file, use_container_width=True)
            
        st.markdown("### Proportion of Days in Each AQI Category (Severe vs Clean)")
        chart2_file = os.path.join(CHARTS_DIR, '2_aqi_category_distribution.png')
        if os.path.exists(chart2_file):
            st.image(chart2_file, use_container_width=True)
            
    with tab2:
        st.markdown("### Kis Vajah Se Hai? (Pollutants Correlation Heatmap)")
        chart3_file = os.path.join(CHARTS_DIR, '3_pollutant_correlation_heatmap.png')
        if os.path.exists(chart3_file):
            st.image(chart3_file, use_container_width=True)
            
        st.markdown("### Major Cities Emission Signatures (Times Above Safe Limit)")
        chart4_file = os.path.join(CHARTS_DIR, '4_city_pollutant_fingerprint.png')
        if os.path.exists(chart4_file):
            st.image(chart4_file, use_container_width=True)
            
        st.markdown("### Seasonal Trends: Why Does Smog Spike in Winter?")
        chart5_file = os.path.join(CHARTS_DIR, '5_seasonal_trend_monthly.png')
        if os.path.exists(chart5_file):
            st.image(chart5_file, use_container_width=True)

    with tab3:
        st.markdown("### Machine Learning Driver Analysis (Random Forest)")
        st.info("Humne 17,000+ daily data samples par Random Forest Regressor train kiya. Model ne **91.9% variance ($R^2 = 0.919$)** explain kiya.")
        chart6_file = os.path.join(CHARTS_DIR, '6_ml_feature_importance.png')
        if os.path.exists(chart6_file):
            st.image(chart6_file, use_container_width=True)
            
else:
    # SINGLE CITY DIAGNOSTIC VIEW
    city_df = filtered_df[filtered_df['City'] == selected_city].copy()
    city_avg_aqi = city_df['AQI'].mean()
    cat, col_hex = get_aqi_category(city_avg_aqi)
    
    st.subheader(f"📍 City Diagnostic Profile: {selected_city}")
    
    # Metric cards
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Average AQI", f"{city_avg_aqi:.1f}", f"Category: {cat}")
    m2.metric("Peak AQI Recorded", f"{city_df['AQI'].max():.0f}")
    m3.metric("Monitored Records", f"{len(city_df.dropna(subset=['AQI'])):,}")
    
    # Dominant pollutant calculation
    poll_means = city_df[list(CPCB_SAFE_LIMITS.keys())].mean()
    ratios = {p: poll_means[p] / CPCB_SAFE_LIMITS[p] for p in CPCB_SAFE_LIMITS if not pd.isna(poll_means.get(p, np.nan))}
    if ratios:
        worst_poll = max(ratios, key=ratios.get)
        m4.metric("Main Violator Pollutant", f"{worst_poll}", f"{ratios[worst_poll]:.1f}x Safe Limit")
    else:
        m4.metric("Main Violator Pollutant", "N/A")
        
    st.divider()
    
    # Diagnosis Box
    st.markdown("### ❓ Kis Vajah Se Hai Is City Me Pollution?")
    if selected_city in CITY_DIAGNOSTICS:
        diag = CITY_DIAGNOSTICS[selected_city]
        st.warning(f"**Primary Driver:** {diag['primary']} ({diag['status']})")
        st.markdown("**Major Root Causes:**")
        for c in diag['causes']:
            st.markdown(f"- 🔴 {c}")
    else:
        st.info(f"Is city ke dominant pollutants: **{worst_poll}** hain jo safe limit se **{ratios.get(worst_poll, 1.0):.1f}x guna** zyada paye gaye.")
        
    # Charts for selected city
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown(f"#### Pollutants vs CPCB Safe Limits ({selected_city})")
        fig, ax = plt.subplots(figsize=(8, 5))
        safe_pollutants = [p for p in CPCB_SAFE_LIMITS.keys() if p in poll_means.index and not pd.isna(poll_means[p])]
        exceed_vals = [poll_means[p] / CPCB_SAFE_LIMITS[p] for p in safe_pollutants]
        
        bar_colors = ['#cc0033' if v > 1.0 else '#009966' for v in exceed_vals]
        bars = ax.bar(safe_pollutants, exceed_vals, color=bar_colors, edgecolor='black', alpha=0.85)
        ax.axhline(1.0, color='red', linestyle='--', linewidth=1.5, label='Safe Standard Limit (1.0x)')
        
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2., h + 0.05, f"{h:.1f}x", ha='center', va='bottom', fontsize=9, fontweight='bold')
            
        ax.set_ylabel("Times Above Safe Limit")
        ax.legend()
        st.pyplot(fig)
        
    with c2:
        st.markdown(f"#### Monthly AQI Pattern ({selected_city})")
        month_avg = city_df.groupby('Month')['AQI'].mean()
        months = np.arange(1, 13)
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.plot(month_avg.index, month_avg.values, marker='o', color='#d95f02', linewidth=2.5)
        ax2.set_xticks(months)
        ax2.set_xticklabels(month_names)
        ax2.set_ylabel("Mean AQI")
        ax2.axvspan(10.5, 12.5, color='red', alpha=0.15, label='Winter Smog Period')
        ax2.legend()
        st.pyplot(fig2)
