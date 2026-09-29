# 🇮🇳 Indian Cities Air Quality & Root Cause Analysis

An end-to-end Data Science and Machine Learning project analyzing Air Quality Index (AQI) and pollutant drivers across 26 major Indian cities using official CPCB data (2015–2020).

## 🚀 Live Demo
Access the interactive web dashboard to explore city rankings, dominant pollutants, and root-cause diagnoses.

## 📊 Key Highlights
- **Scope:** 29,500+ daily observations across 26 major cities.
- **Pollutants Analyzed:** PM2.5, PM10, NO2, NOx, CO, SO2, O3, NH3.
- **Machine Learning:** Random Forest Regressor ($R^2 = 91.9\%$) identifying primary pollutant drivers.
- **Root Cause Profiling:** City-specific breakdown (Industrial load in Ahmedabad, Vehicular + Stubble smog in Delhi, Basin dust in Patna).

## 🛠️ Local Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/air-quality-analysis.git
   cd air-quality-analysis
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the analytical pipeline:
   ```bash
   python aqi_analysis.py
   ```
4. Launch the Streamlit dashboard:
   ```bash
   streamlit run app.py
   ```

## 📁 Repository Structure
- `app.py`: Streamlit interactive dashboard.
- `aqi_analysis.py`: Analytical modeling and chart generation script.
- `city_day.csv`: Central Pollution Control Board (CPCB) dataset.
- `charts/`: 6 high-resolution analytical figures.
- `requirements.txt`: Python package dependencies for cloud deployment.
