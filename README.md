# 🌫️ Seasonal Pollution Episode Prediction

An AI-powered prediction system for forecasting severe pollution episodes in North Indian cities during winter months. This project uses machine learning to analyze weather conditions, stubble burning activity, and historical pollution data to predict hazardous air quality events.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Information](#-model-information)
- [Dataset](#-dataset)

---

## 🎯 Overview

During winter months (October-February), North Indian cities experience severe air pollution episodes due to:
- **Stubble burning** in Punjab and Haryana
- **Meteorological conditions** (low wind, high humidity, temperature inversions)
- **Festival emissions** (Diwali fireworks)
- **Accumulated pollution** from previous days

This project predicts whether a given day will have a **severe pollution episode** based on these factors.

---

## ✨ Features

- **Machine Learning Prediction**: Tuned Decision Tree model with 55% accuracy and 56% recall
- **Interactive Web Application**: User-friendly Streamlit interface
- **Real-time Analysis**: Input current conditions and get instant predictions
- **Risk Assessment**: Identifies key risk factors contributing to pollution
- **Health Recommendations**: Provides actionable health advice based on predictions
- **Feature Importance**: Visualizes which factors most influence predictions

---

## 📁 Project Structure

```
Seasonal-pollution-episode-prediction/
│
├── 01_Data_Collection.ipynb      # Data collection & preprocessing
├── 02_Model_Training.ipynb       # Model training & evaluation
├── 03_Application.py             # Streamlit web application
├── 04_Prediction_Demo.ipynb      # Prediction demonstration
├── README.md                     # This file
│
├── data/
│   ├── raw/
│   │   └── city_day.csv          # Raw pollution data
│   └── final/
│       ├── pollution_dataset_with_features.csv
│       └── pollution_episode_dataset.csv
│
└── models/
    ├── best_model.pkl            # Trained Decision Tree model
    └── model_info.pkl            # Model metadata
```

---

## 💻 Requirements

### System Requirements
- Python 3.9 or higher
- 4GB RAM minimum
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Python Dependencies
```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.18.0
```

---

## 🚀 Installation

### Step 1: Clone or Download the Project

```bash
cd ~/Desktop
# If you have the project folder, navigate to it:
cd Seasonal-pollution-episode-prediction
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install streamlit pandas numpy scikit-learn xgboost matplotlib seaborn plotly
```

Or if you have a requirements.txt:
```bash
pip install -r requirements.txt
```

---

## 🎮 Usage

### Running the Streamlit Application

```bash
# Navigate to project directory
cd /Users/nishalsave/Desktop/Seasonal-pollution-episode-prediction

# Run the Streamlit app
streamlit run 03_Application.py
```

The application will open automatically in your browser at `http://localhost:8501`

### Using the Application

1. **Select City**: Choose from 7 North Indian cities (Delhi, Lucknow, Patna, etc.)
2. **Set Temporal Factors**: Select month and days from Diwali
3. **Input Weather Conditions**: 
   - Temperature (°C)
   - Humidity (%)
   - Wind Speed (km/h)
   - Wind Direction (°)
4. **Input Fire Activity**:
   - Punjab fire count
   - Haryana fire count
5. **Input Historical Data**:
   - Previous day AQI
   - Consecutive poor air quality days
6. **Click "Predict"**: Get instant prediction with risk assessment

### Running Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook

# Then open the notebooks in order:
# 1. 01_Data_Collection.ipynb
# 2. 02_Model_Training.ipynb
# 3. 04_Prediction_Demo.ipynb
```

---

## 🤖 Model Information

### Model Type
**Tuned Decision Tree Classifier**

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| max_depth | 7 |
| min_samples_split | 30 |
| min_samples_leaf | 5 |
| random_state | 42 |

### Performance Metrics
| Metric | Value |
|--------|-------|
| Test Accuracy | 55.0% |
| Recall (Severe Episodes) | 56% |
| Precision (Severe Episodes) | 50% |
| F1-Score | 0.526 |
| AUC | 0.521 |

### Features Used (15 total)
1. `month` - Winter month (Oct-Feb)
2. `temperature_c` - Temperature in Celsius
3. `humidity_pct` - Relative humidity percentage
4. `wind_speed_kmh` - Wind speed in km/h
5. `wind_direction_deg` - Wind direction in degrees
6. `fire_count_punjab` - Active fires in Punjab
7. `fire_count_haryana` - Active fires in Haryana
8. `days_from_diwali` - Days before/after Diwali
9. `previous_day_aqi` - Previous day's AQI
10. `consecutive_poor_days` - Consecutive poor AQI days
11. `weather_dispersion_index` - Engineered feature
12. `stubble_impact_score` - Engineered feature
13. `pollution_momentum` - Engineered feature
14. `diwali_impact_zone` - Binary: within 5 days of Diwali
15. `inversion_risk` - Binary: atmospheric inversion conditions

---

## 📊 Dataset

### Cities Covered
| City | State | Coordinates |
|------|-------|-------------|
| Delhi | Delhi | 28.61°N, 77.21°E |
| Lucknow | Uttar Pradesh | 26.85°N, 80.95°E |
| Patna | Bihar | 25.59°N, 85.14°E |
| Chandigarh | Chandigarh | 30.73°N, 76.78°E |
| Gurugram | Haryana | 28.46°N, 77.03°E |
| Jaipur | Rajasthan | 26.91°N, 75.79°E |
| Amritsar | Punjab | 31.63°N, 74.87°E |

### Data Period
- Winter seasons from 2019-2024
- Months: October, November, December, January, February

### Target Variable
- `severe_episode`: Binary (0 = Normal, 1 = Severe)
- Distribution: 55.5% Normal, 44.5% Severe

---


## 👥 Contributors

- Nishal Save
- Shantanu Ghaisas
- Sanmati Sawalwade 

---

<div align="center">
  <p>🌫️ <strong>Stay Safe, Breathe Clean</strong> 🌫️</p>
  <p><em>Predicting pollution to protect public health</em></p>
</div>

