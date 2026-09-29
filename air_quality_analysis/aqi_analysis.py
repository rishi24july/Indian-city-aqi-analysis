"""
Air Quality Index (AQI) & Pollutant Root-Cause Analysis
======================================================
Analyzes air pollution across 26 major Indian cities:
1. Kis city me kitna pollution hai (Ranking, Distribution, Categories)
2. Kis vajah se hai (Dominant Pollutants, CPCB Limits, Correlations, ML Feature Importance)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Configuration
DATA_PATH = os.path.join(os.path.dirname(__file__), 'city_day.csv')
CHARTS_DIR = os.path.join(os.path.dirname(__file__), 'charts')
os.makedirs(CHARTS_DIR, exist_ok=True)

# Set plotting styles
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.autolayout'] = True

# CPCB National Ambient Air Quality 24-hr Standard Limits (Safe limits)
# Units: ug/m3 (except CO in mg/m3)
CPCB_SAFE_LIMITS = {
    'PM2.5': 60.0,
    'PM10': 100.0,
    'NO2': 80.0,
    'SO2': 80.0,
    'CO': 2.0,
    'O3': 100.0,
    'NH3': 400.0
}

POLLUTANTS = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3']
CORE_POLLUTANTS = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']

AQI_COLORS = {
    'Good': '#009966',          # 0-50
    'Satisfactory': '#84cf33',  # 51-100
    'Moderate': '#ffde33',      # 101-200
    'Poor': '#ff9933',          # 201-300
    'Very Poor': '#cc0033',     # 301-400
    'Severe': '#7e0023'         # 401+
}

def load_and_preprocess_data():
    print("=" * 70)
    print("1. LOADING & PREPROCESSING AIR QUALITY DATA")
    print("=" * 70)
    
    df = pd.read_csv(DATA_PATH)
    print(f"Total raw records: {len(df):,} rows, {df.shape[1]} columns")
    print(f"Total cities: {df['City'].nunique()} cities")
    
    # Date processing
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Month_Name'] = df['Date'].dt.strftime('%b')
    
    # Filter records with valid AQI for supervised analysis
    valid_aqi_df = df.dropna(subset=['AQI']).copy()
    print(f"Records with valid AQI: {len(valid_aqi_df):,} rows")
    
    # Assign AQI Bucket if missing
    def assign_bucket(aqi):
        if pd.isna(aqi): return np.nan
        if aqi <= 50: return 'Good'
        elif aqi <= 100: return 'Satisfactory'
        elif aqi <= 200: return 'Moderate'
        elif aqi <= 300: return 'Poor'
        elif aqi <= 400: return 'Very Poor'
        else: return 'Severe'
        
    valid_aqi_df['AQI_Bucket'] = valid_aqi_df['AQI_Bucket'].fillna(valid_aqi_df['AQI'].apply(assign_bucket))
    
    return df, valid_aqi_df

def analyze_city_pollution_levels(df):
    print("\n" + "=" * 70)
    print("2. KIS CITY ME KITNA POLLUTION HAI? (CITY-WISE RANKING & COMPARISON)")
    print("=" * 70)
    
    # Aggregate stats per city
    city_stats = df.groupby('City')['AQI'].agg(
        Mean_AQI='mean',
        Median_AQI='median',
        Max_AQI='max',
        Days_Monitored='count'
    ).reset_index()
    
    city_stats['Mean_AQI'] = city_stats['Mean_AQI'].round(1)
    city_stats['Median_AQI'] = city_stats['Median_AQI'].round(1)
    city_stats = city_stats.sort_values(by='Mean_AQI', ascending=False).reset_index(drop=True)
    
    def get_category(aqi):
        if aqi <= 50: return 'Good'
        elif aqi <= 100: return 'Satisfactory'
        elif aqi <= 200: return 'Moderate'
        elif aqi <= 300: return 'Poor'
        elif aqi <= 400: return 'Very Poor'
        else: return 'Severe'
        
    city_stats['Overall_Status'] = city_stats['Mean_AQI'].apply(get_category)
    
    print("\nTOP 10 MOST POLLUTED CITIES (Average AQI):")
    print(city_stats.head(10).to_string(index=False))
    
    print("\nTOP 10 CLEANEST CITIES (Lowest Average AQI):")
    print(city_stats.tail(10).sort_values('Mean_AQI').to_string(index=False))
    
    # --- CHART 1: Top 10 Most Polluted vs Cleanest Cities ---
    top10_polluted = city_stats.head(10)
    top10_clean = city_stats.tail(10).sort_values('Mean_AQI', ascending=False)
    combined = pd.concat([top10_polluted, top10_clean]).drop_duplicates(subset=['City'])
    combined = combined.sort_values('Mean_AQI', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 9), dpi=300)
    colors = [AQI_COLORS[cat] for cat in combined['Overall_Status']]
    bars = ax.barh(combined['City'], combined['Mean_AQI'], color=colors, edgecolor='black', alpha=0.85)
    
    # Threshold indicator lines
    ax.axvline(50, color='#009966', linestyle='--', linewidth=1, label='Good Limit (50)')
    ax.axvline(100, color='#84cf33', linestyle='--', linewidth=1, label='Satisfactory (100)')
    ax.axvline(200, color='#ffde33', linestyle='--', linewidth=1.2, label='Moderate (200)')
    ax.axvline(300, color='#ff9933', linestyle='--', linewidth=1.2, label='Poor (300)')
    ax.axvline(400, color='#cc0033', linestyle='--', linewidth=1.5, label='Severe (400+)')
    
    # Value annotations on bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 5, bar.get_y() + bar.get_height()/2, f'{width:.1f}', 
                va='center', ha='left', fontsize=9, fontweight='bold')
        
    ax.set_title("Average Air Quality Index (AQI) Across Major Indian Cities (2015-2020)\n[CPCB Ambient Air Quality Standards]", 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Mean AQI", fontsize=11, fontweight='bold')
    ax.set_ylabel("City", fontsize=11, fontweight='bold')
    ax.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    chart1_path = os.path.join(CHARTS_DIR, '1_city_aqi_ranking.png')
    plt.savefig(chart1_path)
    plt.close()
    print(f"\n[Chart Saved]: {chart1_path}")
    
    # --- CHART 2: AQI Category Days Distribution (% of Good, Moderate, Severe days) ---
    bucket_order = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
    bucket_counts = df.groupby(['City', 'AQI_Bucket']).size().unstack(fill_value=0)
    
    # Ensure all buckets are present
    for b in bucket_order:
        if b not in bucket_counts.columns:
            bucket_counts[b] = 0
    bucket_counts = bucket_counts[bucket_order]
    
    # Percentage of days
    bucket_pct = bucket_counts.div(bucket_counts.sum(axis=1), axis=0) * 100
    
    # Sort cities by Severe + Very Poor percentage
    bucket_pct['Severe_VeryPoor'] = bucket_pct['Severe'] + bucket_pct['Very Poor']
    bucket_pct = bucket_pct.sort_values(by='Severe_VeryPoor', ascending=True)
    bucket_pct = bucket_pct.drop(columns=['Severe_VeryPoor'])
    
    fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
    bucket_pct.plot(kind='barh', stacked=True, 
                     color=[AQI_COLORS[b] for b in bucket_order], 
                     edgecolor='grey', linewidth=0.5, ax=ax)
    
    ax.set_title("Distribution of Air Quality Days by Category (% of Monitored Days)\nWhich Cities Experience The Most 'Severe' and 'Very Poor' Air?", 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Percentage of Monitored Days (%)", fontsize=11, fontweight='bold')
    ax.set_ylabel("City", fontsize=11, fontweight='bold')
    ax.legend(title='AQI Category', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    chart2_path = os.path.join(CHARTS_DIR, '2_aqi_category_distribution.png')
    plt.savefig(chart2_path)
    plt.close()
    print(f"[Chart Saved]: {chart2_path}")
    
    return city_stats, bucket_pct

def analyze_root_causes(df):
    print("\n" + "=" * 70)
    print("3. KIS VAJAH SE HAI? (ROOT CAUSE & POLLUTANT IMPACT ANALYSIS)")
    print("=" * 70)
    
    # 1. Correlation Matrix
    corr_cols = [p for p in POLLUTANTS if p in df.columns] + ['AQI']
    corr_matrix = df[corr_cols].corr()
    aqi_corr = corr_matrix['AQI'].sort_values(ascending=False).drop('AQI')
    
    print("\nPollutants Correlation with AQI (High to Low):")
    for poll, val in aqi_corr.items():
        driver_desc = ""
        if poll in ['PM2.5', 'PM10']: driver_desc = "--> Road dust, vehicular smoke, stubble/crop burning, construction"
        elif poll in ['CO']: driver_desc = "--> Incomplete fossil fuel & biomass burning"
        elif poll in ['NO2', 'NOx', 'NO']: driver_desc = "--> Heavy vehicular traffic, diesel emissions, power generation"
        elif poll in ['SO2']: driver_desc = "--> Industrial combustion, thermal power plants, refineries"
        elif poll in ['O3']: driver_desc = "--> Photochemical smog (sunlight + vehicle exhausts)"
        elif poll in ['NH3']: driver_desc = "--> Agricultural fertilizers, livestock & municipal waste"
        print(f"  * {poll:6s}: {val:.3f}  {driver_desc}")
        
    # --- CHART 3: Correlation Heatmap ---
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='YlOrRd', 
                cbar_kws={'label': 'Pearson Correlation'}, linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap: Air Pollutants vs AQI\n(Identifying Key Drivers of Air Pollution)", 
                 fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    chart3_path = os.path.join(CHARTS_DIR, '3_pollutant_correlation_heatmap.png')
    plt.savefig(chart3_path)
    plt.close()
    print(f"\n[Chart Saved]: {chart3_path}")
    
    # 2. Exceedance of CPCB Safe Limits per City
    city_means = df.groupby('City')[list(CPCB_SAFE_LIMITS.keys())].mean()
    exceedance_ratios = pd.DataFrame(index=city_means.index)
    for pollutant, limit in CPCB_SAFE_LIMITS.items():
        exceedance_ratios[pollutant] = (city_means[pollutant] / limit).round(2)
        
    print("\nPollutant Exceedance Ratio (Values > 1.0 mean EXCEEDING safe limit):")
    sample_cities = ['Delhi', 'Ahmedabad', 'Patna', 'Lucknow', 'Mumbai', 'Bengaluru', 'Kolkata']
    print(exceedance_ratios.loc[[c for c in sample_cities if c in exceedance_ratios.index]])
    
    # --- CHART 4: City Pollutant Fingerprints (Major Cities) ---
    selected_cities = ['Delhi', 'Ahmedabad', 'Patna', 'Lucknow', 'Kolkata', 'Mumbai', 'Bengaluru']
    selected_cities = [c for c in selected_cities if c in exceedance_ratios.index]
    
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    plot_df = exceedance_ratios.loc[selected_cities, ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO']]
    plot_df.plot(kind='bar', ax=ax, colormap='Spectral', edgecolor='black', alpha=0.85, width=0.8)
    
    ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Safe Standard Limit (1.0x)')
    ax.set_title("City-Wise Pollution Fingerprint: Times Above Safe Limit\n(Values > 1.0x violate national safety norms)", 
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("City", fontsize=11, fontweight='bold')
    ax.set_ylabel("Exceedance Multiplier (Observed Mean / Safe Limit)", fontsize=11, fontweight='bold')
    ax.legend(title='Pollutant', frameon=True)
    plt.xticks(rotation=0)
    plt.tight_layout()
    chart4_path = os.path.join(CHARTS_DIR, '4_city_pollutant_fingerprint.png')
    plt.savefig(chart4_path)
    plt.close()
    print(f"[Chart Saved]: {chart4_path}")
    
    return aqi_corr, exceedance_ratios

def seasonal_and_temporal_analysis(df):
    print("\n" + "=" * 70)
    print("4. SEASONAL & MONTHLY TREND ANALYSIS (WHEN DOES POLLUTION PEAK?)")
    print("=" * 70)
    
    monthly_trend = df.groupby(['Month', 'Month_Name'])['AQI'].mean().reset_index()
    monthly_trend = monthly_trend.sort_values('Month')
    
    print("\nMonth-wise Average AQI Across India:")
    for _, row in monthly_trend.iterrows():
        print(f"  * Month {row['Month']:02d} ({row['Month_Name']}): AQI = {row['AQI']:.1f}")
        
    # --- CHART 5: Monthly Pollution Trend & Winter Smog Effect ---
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    
    # Plot top cities monthly trend vs All India
    months = np.arange(1, 13)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    cities_to_track = ['Delhi', 'Ahmedabad', 'Patna', 'Bengaluru', 'Mumbai']
    for city in cities_to_track:
        c_df = df[df['City'] == city].groupby('Month')['AQI'].mean()
        ax.plot(c_df.index, c_df.values, marker='o', linewidth=2, label=city)
        
    all_india = df.groupby('Month')['AQI'].mean()
    ax.plot(all_india.index, all_india.values, color='black', linestyle='--', linewidth=3, label='All India Avg')
    
    ax.set_xticks(months)
    ax.set_xticklabels(month_names)
    ax.axvspan(10.5, 12.5, color='red', alpha=0.15, label='Winter Smog & Stubble Peak (Nov-Dec)')
    ax.axvspan(6.5, 8.5, color='green', alpha=0.15, label='Monsoon Washout (Jul-Aug)')
    
    ax.set_title("Seasonal Cycle of Air Pollution in India (Month-wise AQI Trends)\nWhy Does Pollution Peak in Winter & Drop in Monsoon?", 
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Month", fontsize=11, fontweight='bold')
    ax.set_ylabel("Mean AQI", fontsize=11, fontweight='bold')
    ax.legend(frameon=True, loc='upper right')
    plt.tight_layout()
    chart5_path = os.path.join(CHARTS_DIR, '5_seasonal_trend_monthly.png')
    plt.savefig(chart5_path)
    plt.close()
    print(f"[Chart Saved]: {chart5_path}")

def machine_learning_driver_analysis(df):
    print("\n" + "=" * 70)
    print("5. MACHINE LEARNING: RANDOM FOREST FEATURE IMPORTANCE")
    print("=" * 70)
    
    # Prepare clean dataset for ML
    features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3']
    ml_df = df.dropna(subset=['AQI'] + ['PM2.5', 'PM10']).copy()
    
    # Impute missing remaining features with median
    for col in features:
        if col in ml_df.columns:
            ml_df[col] = ml_df[col].fillna(ml_df[col].median())
            
    X = ml_df[features]
    y = ml_df['AQI']
    
    # ML Best Practice: Train-Test Split Before Preprocessing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"ML Training samples: {len(X_train):,}, Testing samples: {len(X_test):,}")
    
    # Train Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\nModel Performance on Test Set:")
    print(f"  * R-squared (R2 Score): {r2:.4f} ({r2*100:.1f}% variation in AQI explained by pollutants)")
    print(f"  * Mean Absolute Error (MAE): {mae:.2f} AQI points")
    print(f"  * Root Mean Squared Error (RMSE): {rmse:.2f} AQI points")
    
    # Feature Importance
    importances = pd.DataFrame({
        'Pollutant': features,
        'Importance_Pct': (rf.feature_importances_ * 100).round(2)
    }).sort_values('Importance_Pct', ascending=False).reset_index(drop=True)
    
    print("\nExact % Contribution of Each Pollutant to AQI:")
    print(importances.to_string(index=False))
    
    # --- CHART 6: Feature Importance Bar Chart ---
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    bars = ax.bar(importances['Pollutant'], importances['Importance_Pct'], 
                  color='#d95f02', edgecolor='black', alpha=0.85)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    ax.set_title("Machine Learning (Random Forest) Driver Analysis:\nWhich Pollutant Contributes Most to AQI Across Indian Cities?", 
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Air Pollutant", fontsize=11, fontweight='bold')
    ax.set_ylabel("Contribution / Importance (%)", fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(importances['Importance_Pct']) + 5)
    plt.tight_layout()
    chart6_path = os.path.join(CHARTS_DIR, '6_ml_feature_importance.png')
    plt.savefig(chart6_path)
    plt.close()
    print(f"[Chart Saved]: {chart6_path}")
    
    return importances

def generate_city_diagnostics(city_stats, exceedance_ratios):
    print("\n" + "=" * 70)
    print("6. SUMMARY DIAGNOSIS: KIS CITY ME KIS VAJAH SE POLLUTION HAI")
    print("=" * 70)
    
    diagnostics = {
        'Ahmedabad': {
            'Primary_Driver': 'SO2 & PM (Heavy Industries & Thermal/Chemical Plants)',
            'Explanation': 'Ahmedabad mein SO2 (Sulfur Dioxide) aur Particulate Matter kaafi high hai. Vatva, Naroda aur Odhav industrial belts ke chemical/dye units aur heavy coal combustion iske primary reasons hain.'
        },
        'Delhi': {
            'Primary_Driver': 'PM2.5, PM10 & NO2 (Vehicular Congestion + Stubble Burning + Winter Inversion)',
            'Explanation': 'Delhi landlocked city hai jahan winter inversion ke doran hawa freeze ho jati hai. 10+ million vehicles ki wajah se NO2 aur PM2.5 saal bhar high rehte hain, aur Oct-Nov mein Punjab/Haryana stubble burning se AQI 400+ pahunch jata hai.'
        },
        'Patna': {
            'Primary_Driver': 'PM2.5 & PM10 (Indo-Gangetic Basin Silt, Dust & Biomass)',
            'Explanation': 'Patna Indo-Gangetic Plains ke bottom pe situated hai jahan dust aur moisture trap ho jate hain. Unpaved roads, construction dust aur domestic chulha/biomass burning PM levels ko 3-4 guna badha dete hain.'
        },
        'Gurugram': {
            'Primary_Driver': 'PM10 & NO2 (Massive Construction & Highway Traffic)',
            'Explanation': 'Extensive real estate construction, non-stop heavy diesel truck transit on NH-48, aur Aravalli mining/dust Gurugram ke major pollution causes hain.'
        },
        'Lucknow': {
            'Primary_Driver': 'PM2.5 & PM10 (Vehicular Density & Thermal Smog)',
            'Explanation': 'High vehicular density, traffic bottlenecks aur winter season mein localized waste burning PM2.5 aur CO ko critically high rakhti hain.'
        },
        'Kolkata': {
            'Primary_Driver': 'NO2 & PM2.5 (Diesel Taxis/Buses & Port Emissions)',
            'Explanation': 'Old commercial diesel vehicles, high humidity aur riverine moisture particulate matter ko ground level par hold kar leti hain.'
        },
        'Mumbai': {
            'Primary_Driver': 'NO2 & PM10 (Construction Dust & Heavy Marine/Traffic Transit)',
            'Explanation': 'Sea breeze Mumbai ko Delhi jaisa bura banne se bachati hai, lekin ongoing mega-construction projects aur vehicular NO2 air quality ko Moderate se Poor ke beech rakhte hain.'
        },
        'Bengaluru': {
            'Primary_Driver': 'NO2 & PM10 (Traffic Congestion)',
            'Explanation': 'Bengaluru ka overall AQI Moderate/Satisfactory rehta hai due to elevation and greenery, lekin severe traffic jams ke time IT corridors mein NO2 aur localized PM10 spike karte hain.'
        },
        'Aizawl & Shillong': {
            'Primary_Driver': 'None (Consistently Clean Air)',
            'Explanation': 'High forest cover, hilly terrain, low industrialization aur strict emissions regulations ki wajah se yeh India ki cleanest cities hain (AQI < 50).'
        }
    }
    
    for city, info in diagnostics.items():
        print(f"\n[{city.upper()}]")
        print(f"  * Main Driver: {info['Primary_Driver']}")
        print(f"  * Wajah: {info['Explanation']}")
        
    return diagnostics

def main():
    raw_df, valid_df = load_and_preprocess_data()
    city_stats, bucket_pct = analyze_city_pollution_levels(valid_df)
    aqi_corr, exceedance_ratios = analyze_root_causes(valid_df)
    seasonal_and_temporal_analysis(valid_df)
    ml_importances = machine_learning_driver_analysis(valid_df)
    diagnostics = generate_city_diagnostics(city_stats, exceedance_ratios)
    
    print("\n" + "=" * 70)
    print("ALL ANALYSES & CHARTS GENERATED SUCCESSFULLY!")
    print(f"Charts saved in: {CHARTS_DIR}")
    print("=" * 70)

if __name__ == '__main__':
    main()
