# ☀️ Anomaly Detection in Solar Power Generation Using Machine Learning

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B.svg)](https://streamlit.io/)
[![Course](https://img.shields.io/badge/Course-AD124PC503-green.svg)](#)
[![Institute](https://img.shields.io/badge/Institute-DYPCOE%20Akurdi-purple.svg)](https://www.dypcoe.ac.in)

> **Academic Mini-Project for Third Year Artificial Intelligence & Data Science**  
> **Course:** AD124PC503 (Continuous Concurrent Evaluation - CCE IV)  
> **Department:** Artificial Intelligence and Data Science  
> **Institute:** D. Y. Patil College of Engineering, Akurdi, Pune - 411044  
> **Subject Guide:** Dr. Suvarna G.  
> **Project Members:** Damini, Soukhya, Srushti  

---

## 📌 Executive Summary & Scientific Disclaimer

This machine learning project develops an automated health monitoring system that detects unusual operational patterns and potential anomalies in solar photovoltaic (PV) generation and weather sensor telemetry using **Unsupervised Machine Learning Algorithms**.

> [!IMPORTANT]
> **Scientific & Academic Clarification on Anomaly Semantics:**  
> **Anomalies are unusual observations, NOT confirmed equipment faults.** In unsupervised machine learning, an anomaly represents a statistical or spatial outlier in multi-sensor space (e.g., low AC power output despite peak solar irradiation, or unexpected power clipping). While such deviations can indicate inverter failure, sensor calibration drift, or panel shading, they can also stem from sudden cloud transits, high module temperatures, or intentional grid curtailment. Domain expertise is required to verify actual hardware breakdowns.
> 
> Furthermore, because the Kaggle Solar Power Generation benchmark dataset contains no verified ground-truth labels of hardware faults, **supervised evaluation metrics (Accuracy, Precision, Recall, F1-Score) are not scientifically applicable and are deliberately excluded.**

---

## 📑 Table of Contents
1. [Project Overview & Objectives](#-project-overview--objectives)
2. [Dataset Description & Schema](#-dataset-description--schema)
3. [System Architecture & Methodology](#-system-architecture--methodology)
4. [Machine Learning Algorithms](#-machine-learning-algorithms)
   - [Isolation Forest (Primary Model)](#1-isolation-forest-primary-model)
   - [DBSCAN (Comparison Model)](#2-dbscan-comparison-model)
5. [Comparative Evaluation & Experimental Results](#-comparative-evaluation--experimental-results)
6. [Project File Structure](#-project-file-structure)
7. [Installation & Execution Guide](#-installation--execution-guide)
   - [Local Setup & Training](#local-setup--training)
   - [Launching the Streamlit Web Application](#launching-the-streamlit-web-application)
   - [Running the Google Colab Notebook](#running-the-google-colab-notebook)
8. [Streamlit Interactive Web Application Features](#-streamlit-interactive-web-application-features)
9. [Limitations & Future Scope](#-limitations--future-scope)
10. [Viva Voce Questions & Answers](#-viva-voce-questions--answers)

---

## 🎯 Project Overview & Objectives

### Problem Statement
Large-scale utility solar power plants consist of thousands of PV modules feeding multiple central inverters. Identifying operational underperformance in real-time is challenging due to diurnal day-night cycles, cloud intermittency, and weather fluctuations. Manual inspection of continuous 15-minute multi-sensor feeds across 22 inverters is impractical.

### Core Objectives
1. **Automated Ingestion & Preprocessing:** Ingest and synchronize inverter generation data with ambient weather monitoring station data on timestamps.
2. **Feature Engineering & Selection:** Focus on six physically meaningful telemetry features representing electrical conversion and environmental exposure.
3. **Unsupervised Outlier Detection:** Train an **Isolation Forest** ensemble as the primary anomaly detection algorithm ($n_{estimators}=100, contamination=0.05$).
4. **Density-Based Comparison:** Implement **DBSCAN** ($\epsilon=0.8, min\_samples=10$) on standardized features to contrast tree isolation against spatial density clustering.
5. **Interactive Demonstration:** Build a full-featured **Streamlit web application** for telemetry exploration, model diagnostics, and single/batch inference.
6. **Academic Reproducibility:** Deliver an end-to-end Google Colab-compatible Jupyter notebook.

---

## 📊 Dataset Description & Schema

The system analyzes the benchmark **Kaggle Solar Power Generation Dataset** (Plant 1):

### 1. Generation Data (`Plant_1_Generation_Data.csv`)
- **Total Records:** 68,778 rows logged at 15-minute intervals across 34 days (May 15, 2020 – June 18, 2020).
- **Inverters Monitored:** 22 unique inverters identified by `SOURCE_KEY`.
- **Fields:** `DATE_TIME`, `PLANT_ID`, `SOURCE_KEY`, `DC_POWER`, `AC_POWER`, `DAILY_YIELD`, `TOTAL_YIELD`.

### 2. Weather Sensor Data (`Plant_1_Weather_Sensor_Data.csv`)
- **Total Records:** 3,182 rows logged at 15-minute intervals by the plant's weather station.
- **Fields:** `DATE_TIME`, `PLANT_ID`, `SOURCE_KEY`, `AMBIENT_TEMPERATURE`, `MODULE_TEMPERATURE`, `IRRADIATION`.

### 3. Merging Strategy & Selected Features
- **Join Strategy:** Inner join on `DATE_TIME` after parsing `DD-MM-YYYY HH:MM` (generation) and `YYYY-MM-DD HH:MM:SS` (weather).
- **Synchronized Observations:** Exactly **68,774** observations after null/infinity filtering.
- **Selected Numerical Features (6 Dimensions):**
  1. `DC_POWER` (kW): Direct current generated by the PV array.
  2. `AC_POWER` (kW): Alternating current converted by the inverter.
  3. `DAILY_YIELD` (kWh): Cumulative energy generated so far on the given calendar day.
  4. `AMBIENT_TEMPERATURE` (°C): Surrounding air temperature.
  5. `MODULE_TEMPERATURE` (°C): Surface temperature of the solar panels.
  6. `IRRADIATION` ($W/m^2$ or normalized index): Solar radiation captured by pyranometer.

---

## 🏗️ System Architecture & Methodology

```
+----------------------------+     +-------------------------------+
| Generation Telemetry (CSV) |     | Weather Sensor Data (CSV)     |
| (68,778 rows, 22 Inverters)|     | (3,182 rows, Weather Station) |
+--------------+-------------+     +---------------+---------------+
               |                                   |
               +-----------------+-----------------+
                                 |
                                 v
               +-----------------------------------+
               |  DATE_TIME Parsing & Harmonization|
               |  (Inner Join on 15-min Intervals) |
               +-----------------+-----------------+
                                 |
                                 v
               +-----------------------------------+
               | Clean Synchronized Telemetry      |
               | (68,774 Rows, 6 Sensor Features)  |
               +-----------------+-----------------+
                                 |
         +-----------------------+-----------------------+
         |                                               |
         v                                               v
+-------------------------------+             +-------------------------------+
| Feature Matrix X              |             | Feature Matrix X              |
| (Raw Scale Invariant)         |             | (StandardScaler: Mean=0,Var=1)|
+---------------+---------------+             +---------------+---------------+
                |                                             |
                v                                             v
+-------------------------------+             +-------------------------------+
| Model 1: Isolation Forest     |             | Model 2: DBSCAN               |
| - n_estimators = 100          |             | - eps = 0.8                   |
| - contamination = 0.05        |             | - min_samples = 10            |
+---------------+---------------+             +---------------+---------------+
                |                                             |
                v                                             v
+-------------------------------+             +-------------------------------+
| 3,439 Flagged Anomalies       |             | 6 Spatial Noise Points        |
| (5.0004% of observations)     |             | (0.0087% of observations)     |
+---------------+---------------+             +---------------+---------------+
                |                                             |
                +-----------------------+---------------------+
                                        |
                                        v
                    +---------------------------------------+
                    | Comparative Analysis & Diagnostics    |
                    | - DC vs AC Conversion Discrepancies   |
                    | - Irradiation vs AC Power Deviations  |
                    | - Streamlit Dashboard & Live API      |
                    +---------------------------------------+
```

---

## 🤖 Machine Learning Algorithms

### 1. Isolation Forest (Primary Model)
- **Concept:** Isolation Forest is an unsupervised tree ensemble based on the principle that anomalies are "few and different". It constructs Isolation Trees (iTrees) by recursively selecting a random feature and a random split value between its minimum and maximum.
- **Mathematical Basis:** Anomalies require significantly fewer random splits to isolate into terminal nodes. The anomaly score is defined as:
  $$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$
  where $E(h(x))$ is the average path length of observation $x$ across all trees, and $c(n)$ is the average path length of unsuccessful searches in a Binary Search Tree of $n$ instances:
  $$c(n) = 2 \ln(n - 1) + 0.5772156649 - \frac{2(n - 1)}{n}$$
- **Hyperparameters:**
  - `n_estimators = 100`: Stable ensemble convergence.
  - `contamination = 0.05`: Expectation that ~5% of observations display anomalous patterns.
  - `random_state = 42`: Deterministic reproducibility.
- **Scaling:** **Not required.** Tree partitioning is monotonic and invariant to numerical scale.

### 2. DBSCAN (Comparison Model)
- **Concept:** Density-Based Spatial Clustering of Applications with Noise (DBSCAN) identifies clusters as contiguous regions of high point density separated by sparse regions.
- **Points Classification:**
  - **Core Point:** Contains at least $min\_samples$ points within distance $\epsilon$.
  - **Border Point:** Resides within distance $\epsilon$ of a core point but has fewer than $min\_samples$ neighbors.
  - **Noise Point (Anomaly):** Neither core nor border point (assigned cluster label `-1`).
- **Hyperparameters:**
  - `eps = 0.8`: Euclidean neighborhood radius.
  - `min_samples = 10`: Minimum observations required to form a dense core.
- **Scaling:** **Strictly Required.** DBSCAN calculates Euclidean distance in $\mathbb{R}^6$:
  $$d(\mathbf{u}, \mathbf{v}) = \sqrt{\sum_{i=1}^6 (u_i - v_i)^2}$$
  Without `StandardScaler` ($z = \frac{x - \mu}{\sigma}$), $DC\_POWER$ ($\approx 10,000$) completely suppresses $IRRADIATION$ ($\approx 0.8$).

---

## 📈 Comparative Evaluation & Experimental Results

All results are calculated directly from the merged dataset of **68,774 observations**:

| Evaluation Dimension | Isolation Forest (Primary) | DBSCAN (Comparison) |
| :--- | :--- | :--- |
| **Algorithmic Category** | Tree-Based Partitioning Ensemble | Density-Based Spatial Clustering |
| **Key Hyperparameters** | `n_estimators=100, contamination=0.05` | `eps=0.8, min_samples=10` |
| **Feature Scaling** | Not required (scale-invariant) | Mandatory (`StandardScaler`) |
| **Total Observations** | 68,774 | 68,774 |
| **Normal Observations** | **65,335** (94.9996%) | **68,768** (99.9913%) |
| **Detected Anomalies / Noise** | **3,439** | **6** |
| **Anomaly Percentage** | **5.0004%** | **0.0087%** |
| **Key Advantage** | High computational efficiency $O(n \log n)$; robust across multi-sensor regimes. | Discovers non-linear cluster geometry; does not enforce preset anomaly quota. |
| **Key Limitation** | Anomaly count depends on predefined `contamination` parameter. | Sensitive to $\epsilon$; vulnerable to non-uniform density across day/night cycles. |

### Why do the two algorithms produce different anomaly counts?
1. **Contamination Enforcement vs. Natural Density:** Isolation Forest is constrained by `contamination=0.05`, designating the top 5% shortest-path points as anomalies (~3,439 rows). In contrast, DBSCAN explores density reachability without an outlier quota; only 6 observations were truly isolated from all dense operational manifolds at $\epsilon=0.8$.
2. **Subspace vs. Full Metric Space:** Isolation Forest isolates points along individual orthogonal feature projections, identifying localized anomalies (e.g. zero AC power despite peak irradiation). DBSCAN requires distance divergence across all 6 standardized dimensions simultaneously.

---

## 📁 Project File Structure

```
solar_anomaly_detection/
│
├── server.py                  # Full-Stack Flask Web Server & REST API backend
├── app.py                     # Interactive Streamlit Web Application
├── train_model.py             # Complete training & evaluation pipeline script
├── requirements.txt           # Project dependencies
├── README.md                  # Comprehensive documentation & viva guide
│
├── frontend/                  # Modern Interactive Web Dashboard (Single Page App)
│   ├── index.html             # Responsive TailwindCSS & Chart.js dashboard
│   ├── style.css              # Custom styling, sliders & status card animations
│   └── app.js                 # Asynchronous REST API client & Chart.js controller
│
├── data/
│   ├── Plant_1_Generation_Data.csv      # Raw Kaggle Generation Data
│   ├── Plant_1_Weather_Sensor_Data.csv  # Raw Kaggle Weather Sensor Data
│   ├── generation_data.csv              # Standardized alias
│   └── weather_data.csv                 # Standardized alias
│
├── models/
│   ├── isolation_forest.pkl             # Trained Isolation Forest model
│   ├── dbscan.pkl                       # Trained DBSCAN model
│   └── scaler.pkl                       # Fitted StandardScaler
│
├── notebooks/
│   └── Solar_Anomaly_Detection.ipynb    # 16-section Colab-compatible notebook
│
├── outputs/
│   ├── anomaly_results.csv              # Full dataset with anomaly labels & scores
│   ├── model_comparison.csv             # Tabular model comparison metrics
│   └── graphs/
│       ├── missing_values.png           # Missing values inspection
│       ├── feature_distributions.png    # 6-panel feature distribution plot
│       ├── correlation_heatmap.png      # Sensor correlation matrix
│       ├── ambient_vs_module_temp.png   # Temperature dynamics
│       ├── iforest_dc_vs_ac.png         # DC vs AC with anomalies highlighted
│       ├── iforest_irradiation_vs_ac.png# Irradiation vs AC with anomalies highlighted
│       ├── iforest_timeseries_anomalies.png # Time-series trace of anomalies
│       ├── iforest_distribution_pie.png # Normal vs Anomaly pie chart
│       ├── dbscan_clusters.png          # DBSCAN dense clusters and noise points
│       └── model_comparison_bar.png     # Comparative count & percentage bar charts
│
└── utils/
    ├── __init__.py                      # Package initialization
    ├── preprocessing.py                 # Ingestion, datetime parsing & standardization
    ├── visualization.py                 # Matplotlib & Seaborn charting functions
    └── prediction.py                    # Inference pipeline for single & batch scoring
```

---

## 🚀 Installation & Execution Guide

### Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14
- Recommended: Virtual environment (`venv` or `conda`)

### Step 1: Clone or Navigate to Directory
```powershell
cd "c:\Users\wsouk\OneDrive\Desktop\ml cce"
```

### Step 2: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Run the Training Pipeline
Execute the full automated training script to preprocess data, train both algorithms, and generate all graphs:
```powershell
python train_model.py
```
*Output:*
- Merges 68,774 observations.
- Trains and saves `isolation_forest.pkl`, `dbscan.pkl`, and `scaler.pkl` to `models/`.
- Exports `anomaly_results.csv` and `model_comparison.csv` to `outputs/`.
- Saves 10 high-resolution charts to `outputs/graphs/`.

### Step 4A: Launch the React Frontend & Flask Backend

#### 1. Start the Flask Backend Server (Port 5000)
In your first terminal:
```powershell
python server.py
```
*The backend loads the trained models and serves the REST API at `http://localhost:5000` (and serves the pre-built React app at `http://localhost:5000`).*

#### 2. Start the Vite React Development Server (Port 5173)
In your second terminal:
```powershell
cd frontend
npm run dev
```
*Vite launches with instant Hot Module Replacement (HMR) at `http://localhost:5173` and automatically proxies all `/api/*` calls to the Flask backend on port 5000.*

#### 3. Build React for Production (Optional)
```powershell
cd frontend
npm run build
```
*Builds the optimized production bundle into `frontend/dist/`. Flask automatically detects and serves this bundle directly at `http://localhost:5000`.*

### Step 4B: Launch the Streamlit Web Application (Alternative)
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### REST API Documentation
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the interactive frontend Single Page Application dashboard |
| `GET` | `/api/status` | System health check and model loading status |
| `GET` | `/api/kpis` | Real-time observation counts, normal/anomaly percentages |
| `GET` | `/api/presets` | Preset demonstration scenarios (Nominal midday, Inverter trip, Clipping, Night idle) |
| `GET` | `/api/charts/data` | Scatter, pie, and time-series data for Chart.js rendering |
| `POST` | `/api/predict` | Single telemetry observation inference returning status, score & diagnosis |
| `POST` | `/api/predict/batch` | Batch scoring endpoint for CSV uploads or JSON records |

### Step 5: Running the Google Colab Notebook
1. Open [Google Colab](https://colab.research.google.com).
2. Upload `notebooks/Solar_Anomaly_Detection.ipynb`.
3. Upload `Plant_1_Generation_Data.csv` and `Plant_1_Weather_Sensor_Data.csv` to the session files.
4. Execute cells sequentially (`Runtime -> Run all`).

---

## 🖥️ Streamlit Interactive Web Application Features

The interactive dashboard includes 6 specialized navigation pages:
1. **📊 Dashboard:** High-level executive KPIs (total records: 68,774, normal count: 65,335, IF anomalies: 3,439, DBSCAN noise: 6, anomaly pie chart, and project summary).
2. **📂 Dataset Explorer:** Inspect raw telemetry, verify schemas and non-null counts, view descriptive statistics ($mean, std, min, max$), and test custom dataset uploads.
3. **🔍 Anomaly Detection:** Switch between Isolation Forest and DBSCAN, filter observations by status (Normal / Anomaly), and download complete prediction CSVs.
4. **📈 Visual Analytics:** Interactive visualizations including DC vs AC conversion curve, Irradiation vs AC solar curve, 15-minute time-series with anomaly flags, and DBSCAN clustering.
5. **🧠 Model Information:** Complete mathematical formulations (tree path length formula, DBSCAN density equations), hyperparameter rationale, comparison matrix, and viva defense guide.
6. **⚡ New Prediction:**
   - **Single-Observation Diagnostic:** Real-time sliders for the 6 sensor inputs predicting status and continuous anomaly score.
   - **Batch CSV Scoring:** Upload custom CSV files with sensor readings to score hundreds of rows instantly and download annotated predictions.

---

## ⚠️ Limitations & Future Scope

### Current Limitations
1. **Unlabeled Benchmark:** Without ground-truth failure logs from plant maintenance crews, model performance cannot be validated using supervised metrics (Precision/Recall).
2. **Global vs Inverter-Specific Dynamics:** All 22 inverters are pooled into a unified feature space. Inverters located at different corners of a multi-megawatt farm experience localized shading and microclimates.
3. **Static Contamination Assumption:** Isolation Forest requires a preset contamination fraction ($0.05$), which may vary across seasons.

### Future Scope
1. **Semi-Supervised & Active Learning:** Incorporate maintenance work orders to refine anomaly thresholds.
2. **Deep Sequence Modeling:** Train Temporal Convolutional Networks (TCN) or LSTM Autoencoders to capture sequential ramp rates across consecutive 15-minute time intervals.
3. **Edge Deployment:** Deploy lightweight ONNX-quantized models onto SCADA edge gateways for sub-second inverter health alerting.

---

## 🎓 Viva Voce Questions & Answers

**Q1: What is the primary objective of this project?**  
*Answer:* To develop an unsupervised machine learning system capable of detecting unusual patterns and potential anomalies in solar power generation and weather sensor data without requiring prior failure labels.

**Q2: Why are anomalies described as "unusual observations" rather than "confirmed equipment faults"?**  
*Answer:* Unsupervised algorithms detect mathematical outliers—observations that deviate from the majority data distribution. While such deviations may indicate inverter failure or sensor drift, they could also arise from extreme weather, sudden cloud cover, or scheduled curtailment. Verification requires physical inspection.

**Q3: Why did you not compute Accuracy, Precision, Recall, or F1-Score?**  
*Answer:* Supervised metrics require verified ground-truth labels indicating which observations were actual faults. Because the Kaggle Solar Power dataset does not include verified fault labels, calculating accuracy or F1 would require fabricating labels, which is scientifically invalid.

**Q4: Why was feature scaling required for DBSCAN but not for Isolation Forest?**  
*Answer:* DBSCAN relies on multi-dimensional Euclidean distance ($d = \sqrt{\sum (u_i - v_i)^2}$). Because features have drastically different magnitudes ($DC\_POWER \approx 10,000$ vs $IRRADIATION \approx 0.8$), unscaled distances would be completely dominated by DC Power. Isolation Forest relies on tree-based monotonic splits, making it invariant to feature scale.

**Q5: How does Isolation Forest calculate its anomaly score?**  
*Answer:* Isolation Forest measures the path length $h(x)$ required to isolate an observation in an ensemble of random decision trees. Anomalies require fewer splits (shorter path lengths) than normal clustered points. The score is $s(x, n) = 2^{-E(h(x))/c(n)}$, where values approaching 1 denote anomalies.

**Q6: Why did Isolation Forest detect 3,439 anomalies while DBSCAN detected only 6?**  
*Answer:* Isolation Forest was configured with `contamination=0.05`, mathematically designating the 5% most isolated instances as anomalies. DBSCAN evaluates continuous density in 6D space without a preset quota; only 6 observations were isolated from all dense clusters under $\epsilon=0.8$ and $min\_samples=10$.
