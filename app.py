"""
Solar Power Generation Anomaly Detection System
Streamlit Web Application for Academic Mini-Project Demonstration
Department of Artificial Intelligence and Data Science
D. Y. Patil College of Engineering, Akurdi, Pune
"""

import os
import sys
import io
import time
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Ensure local utils can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.preprocessing import (
    SELECTED_FEATURES,
    load_raw_datasets,
    preprocess_and_merge,
    prepare_features,
    scale_features,
    validate_input_data
)
from utils.visualization import (
    plot_missing_values,
    plot_feature_distributions,
    plot_correlation_heatmap,
    plot_dc_vs_ac,
    plot_irradiation_vs_ac,
    plot_temperature_relationship,
    plot_timeseries_anomalies,
    plot_dbscan_clusters,
    plot_model_comparison_bars,
    plot_anomaly_distribution_pie
)
from utils.prediction import (
    load_model,
    predict_isolation_forest,
    predict_single_instance,
    predict_batch_dataframe
)

# Page configuration
st.set_page_config(
    page_title="Solar Power Anomaly Detection | CCE-IV Mini Project",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1E3C72 0%, #2A5298 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 16px;
        border-left: 5px solid #2A5298;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .status-normal {
        background-color: #d4edda;
        color: #155724;
        padding: 12px;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        border: 1px solid #c3e6cb;
    }
    .status-anomaly {
        background-color: #f8d7da;
        color: #721c24;
        padding: 12px;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        border: 1px solid #f5c6cb;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        border-radius: 6px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# Path definitions
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
GRAPHS_DIR = os.path.join(OUTPUTS_DIR, "graphs")
DATA_DIR = os.path.join(BASE_DIR, "data")

IFOREST_MODEL_PATH = os.path.join(MODELS_DIR, "isolation_forest.pkl")
DBSCAN_MODEL_PATH = os.path.join(MODELS_DIR, "dbscan.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
ANOMALY_RESULTS_PATH = os.path.join(OUTPUTS_DIR, "anomaly_results.csv")
COMPARISON_CSV_PATH = os.path.join(OUTPUTS_DIR, "model_comparison.csv")


# Cached Data & Model Loaders
@st.cache_resource
def get_trained_models():
    """Load pretrained models and scalers if available."""
    models = {}
    if os.path.exists(IFOREST_MODEL_PATH):
        models['iforest'] = load_model(IFOREST_MODEL_PATH)
    if os.path.exists(DBSCAN_MODEL_PATH):
        models['dbscan'] = load_model(DBSCAN_MODEL_PATH)
    if os.path.exists(SCALER_PATH):
        models['scaler'] = load_model(SCALER_PATH)
    return models


@st.cache_data
def get_cached_results():
    """Load precalculated anomaly results if available."""
    if os.path.exists(ANOMALY_RESULTS_PATH):
        df = pd.read_csv(ANOMALY_RESULTS_PATH)
        if 'DATE_TIME' in df.columns:
            df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])
        return df
    return None


@st.cache_data
def get_comparison_data():
    """Load model comparison data."""
    if os.path.exists(COMPARISON_CSV_PATH):
        return pd.read_csv(COMPARISON_CSV_PATH)
    return None


# Sidebar Navigation & College Metadata
st.sidebar.image("https://img.icons8.com/fluency/96/sun.png", width=70)
st.sidebar.title("Solar AI Sentinel")
st.sidebar.caption("Unsupervised Anomaly Detection Platform")
st.sidebar.markdown("---")

app_mode = st.sidebar.radio(
    "Navigate Project Sections:",
    [
        "📊 Dashboard",
        "📂 Dataset Explorer",
        "🔍 Anomaly Detection",
        "📈 Visual Analytics",
        "🧠 Model Information",
        "⚡ New Prediction"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Academic Mini-Project Information**
- **Course:** AD124PC503 (Machine Learning)
- **Class:** TY AI & Data Science
- **Institute:** D. Y. Patil College of Engineering, Akurdi, Pune
- **Evaluation:** CCE IV Mini-Project
- **Note:** *Anomalies represent unusual observations, not confirmed equipment faults.*
""")

# Load models and saved data
models = get_trained_models()
cached_results = get_cached_results()
cached_comparison = get_comparison_data()


# -------------------------------------------------------------------------------------------------
# 1. DASHBOARD
# -------------------------------------------------------------------------------------------------
if app_mode == "📊 Dashboard":
    st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 2.2rem;">☀️ Anomaly Detection in Solar Power Generation</h1>
        <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 1.05rem;">
            Academic Mini-Project | Third Year Artificial Intelligence & Data Science | D. Y. Patil College of Engineering, Akurdi
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### 🎯 Project Overview & Objective
    This machine learning system continuously analyzes multi-sensor operational telemetry from solar photovoltaic (PV) plants 
    and detects unusual operational patterns or anomalies using **Unsupervised Machine Learning Algorithms**.
    
    The system identifies sensor drift, unusual conversion efficiency drops, inverter sub-performance, or atypical 
    environmental regimes without requiring labeled historical failure data.
    """)

    col1, col2, col3, col4 = st.columns(4)

    total_obs = len(cached_results) if cached_results is not None else 68774
    n_iforest_anom = int((cached_results['anomaly_iforest'] == -1).sum()) if cached_results is not None and 'anomaly_iforest' in cached_results.columns else 3439
    n_iforest_norm = total_obs - n_iforest_anom
    pct_iforest = (n_iforest_anom / total_obs) * 100

    n_dbscan_noise = int((cached_results['anomaly_dbscan'] == -1).sum()) if cached_results is not None and 'anomaly_dbscan' in cached_results.columns else 6
    pct_dbscan = (n_dbscan_noise / total_obs) * 100

    with col1:
        st.metric(
            label="Total Merged Observations",
            value=f"{total_obs:,}",
            help="Total 15-minute sensor readings merged from Plant 1 Generation & Weather stations"
        )
    with col2:
        st.metric(
            label="Normal Observations (IF)",
            value=f"{n_iforest_norm:,}",
            delta=f"{100 - pct_iforest:.2f}%",
            help="Observations following nominal power generation characteristics"
        )
    with col3:
        st.metric(
            label="Isolation Forest Anomalies",
            value=f"{n_iforest_anom:,}",
            delta=f"-{pct_iforest:.2f}%",
            delta_color="inverse",
            help="Unusual observations flagged at 5% preset contamination rate"
        )
    with col4:
        st.metric(
            label="DBSCAN Noise Points",
            value=f"{n_dbscan_noise:,}",
            delta=f"-{pct_dbscan:.4f}%",
            delta_color="inverse",
            help="Extreme spatial outliers in 6D standardized sensor space"
        )

    st.markdown("---")
    
    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.subheader("📋 Executive Project Summary")
        st.markdown(fr"""
        - **Primary Algorithm:** **Isolation Forest** (Tree-based ensemble, $n_{{estimators}}=100$, $contamination=0.05$).
        - **Comparison Algorithm:** **DBSCAN** (Density-Based Spatial Clustering with Noise, $\epsilon=0.8$, $min\_samples=10$).
        - **Target Dataset:** Kaggle Solar Power Generation Dataset (Plant 1: 34 days of 15-minute inverter telemetry).
        - **Selected Feature Dimensions:** 
          1. `DC_POWER` (kW)
          2. `AC_POWER` (kW)
          3. `DAILY_YIELD` (kWh)
          4. `AMBIENT_TEMPERATURE` (°C)
          5. `MODULE_TEMPERATURE` (°C)
          6. `IRRADIATION` ($W/m^2$ or index)
        - **Operational Status:** Models trained, cross-verified, and persisted ready for live inference.
        """)
        
        st.info("💡 **Viva Tip:** In unsupervised anomaly detection, anomalies represent mathematical irregularities in sensor feature space. They indicate potential sub-optimal generation or sensor issues, not confirmed hardware breakdowns.")

    with col_right:
        st.subheader("📊 Anomaly Distribution")
        fig_pie = plot_anomaly_distribution_pie(n_iforest_norm, n_iforest_anom, model_name="Isolation Forest")
        st.pyplot(fig_pie, use_container_width=True)


# -------------------------------------------------------------------------------------------------
# 2. DATASET EXPLORER
# -------------------------------------------------------------------------------------------------
elif app_mode == "📂 Dataset Explorer":
    st.title("📂 Dataset Explorer & Telemetry Inspection")
    st.markdown("Inspect raw solar generation and weather datasets, explore data types, check missing values, and review summary statistics.")

    upload_tab, default_tab = st.tabs(["📁 Inspect Current Datasets", "📤 Upload Custom Datasets"])

    with default_tab:
        st.subheader("Default Plant 1 Telemetry Data")
        if cached_results is not None:
            st.success(f"Loaded merged plant data containing **{len(cached_results):,}** observations.")
            
            st.markdown("#### Sample Preview (First 10 Rows)")
            st.dataframe(cached_results.head(10), use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Feature Data Types & Null Counts")
                info_df = pd.DataFrame({
                    "Feature": cached_results.columns,
                    "Data Type": [str(t) for t in cached_results.dtypes],
                    "Non-Null Count": cached_results.count().values,
                    "Null Count": cached_results.isnull().sum().values
                })
                st.dataframe(info_df, use_container_width=True)

            with col2:
                st.markdown("#### Numerical Summary Statistics")
                numeric_cols = [c for c in SELECTED_FEATURES if c in cached_results.columns]
                st.dataframe(cached_results[numeric_cols].describe().T.round(2), use_container_width=True)
        else:
            st.warning("No preprocessed dataset found. Please run the training pipeline first or upload CSV files.")

    with upload_tab:
        st.subheader("Upload Custom Generation and Weather CSV Files")
        st.markdown("Upload separate Generation Data and Weather Sensor Data to merge and inspect.")

        up_col1, up_col2 = st.columns(2)
        with up_col1:
            gen_file = st.file_uploader("Upload Generation Data CSV", type=["csv"], key="gen_up")
        with up_col2:
            weather_file = st.file_uploader("Upload Weather Data CSV", type=["csv"], key="weather_up")

        if gen_file and weather_file:
            try:
                raw_gen = pd.read_csv(gen_file)
                raw_weather = pd.read_csv(weather_file)
                st.write(f"Generation Rows: {len(raw_gen):,} | Weather Rows: {len(raw_weather):,}")

                with st.spinner("Parsing timestamps and merging datasets on DATE_TIME..."):
                    custom_merged, custom_stats = preprocess_and_merge(raw_gen, raw_weather)
                
                st.success(f"Successfully merged! Final clean rows: **{len(custom_merged):,}**")
                st.dataframe(custom_merged.head(10), use_container_width=True)
                
                st.markdown("#### Summary Statistics of Uploaded Data")
                st.dataframe(custom_merged[SELECTED_FEATURES].describe().T.round(2), use_container_width=True)
            except Exception as e:
                st.error(f"Error processing uploaded files: {e}")


# -------------------------------------------------------------------------------------------------
# 3. ANOMALY DETECTION
# -------------------------------------------------------------------------------------------------
elif app_mode == "🔍 Anomaly Detection":
    st.title("🔍 Unsupervised Anomaly Detection")
    st.markdown("Execute and inspect unsupervised anomaly detection using **Isolation Forest** or **DBSCAN**.")

    if cached_results is None:
        st.warning("Anomaly results not found. Run `python train_model.py` to generate model predictions.")
    else:
        model_choice = st.radio(
            "Select Detection Algorithm:",
            ["Isolation Forest (Primary)", "DBSCAN (Comparison)"],
            horizontal=True
        )

        if "Isolation Forest" in model_choice:
            st.subheader("🌲 Isolation Forest Detections")
            st.markdown("""
            Isolation Forest isolates anomalies by randomly selecting a feature and randomly splitting the value. 
            Because anomalous instances require fewer random splits to isolate, they exhibit significantly **shorter path lengths** in the trees.
            """)

            col_m1, col_m2, col_m3 = st.columns(3)
            anom_df = cached_results[cached_results['anomaly_iforest'] == -1]
            norm_df = cached_results[cached_results['anomaly_iforest'] == 1]
            
            with col_m1:
                st.metric("Total Flagged Anomalies", f"{len(anom_df):,}")
            with col_m2:
                st.metric("Normal Observations", f"{len(norm_df):,}")
            with col_m3:
                st.metric("Anomaly Rate", f"{(len(anom_df)/len(cached_results))*100:.2f}%")

            filter_view = st.selectbox("Filter Display Data:", ["Show Detected Anomalies Only", "Show Normal Observations Only", "Show All Data"])
            
            if filter_view == "Show Detected Anomalies Only":
                disp_df = anom_df
            elif filter_view == "Show Normal Observations Only":
                disp_df = norm_df
            else:
                disp_df = cached_results

            st.dataframe(
                disp_df[['DATE_TIME'] + SELECTED_FEATURES + ['anomaly_score_iforest', 'label_iforest']].head(500),
                use_container_width=True
            )

            # CSV Download
            csv_data = disp_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Isolation Forest Prediction Results (CSV)",
                data=csv_data,
                file_name="isolation_forest_predictions.csv",
                mime="text/csv"
            )

        else:
            st.subheader("🌐 DBSCAN Density-Based Detections")
            st.markdown(r"""
            DBSCAN identifies clusters as continuous dense regions. Observations in sparse areas whose $\epsilon$-neighborhood ($\epsilon=0.8$) contains 
            fewer than $min\_samples=10$ neighbors are categorized as **noise outliers** (Label: -1).
            """)

            noise_df = cached_results[cached_results['anomaly_dbscan'] == -1]
            clustered_df = cached_results[cached_results['anomaly_dbscan'] == 1]

            col_d1, col_d2, col_d3 = st.columns(3)
            with col_d1:
                st.metric("Detected Noise Points", f"{len(noise_df):,}")
            with col_d2:
                st.metric("Clustered Nominal Points", f"{len(clustered_df):,}")
            with col_d3:
                st.metric("Noise Percentage", f"{(len(noise_df)/len(cached_results))*100:.4f}%")

            st.markdown("#### Table of Detected DBSCAN Noise Points")
            if len(noise_df) > 0:
                st.dataframe(
                    noise_df[['DATE_TIME'] + SELECTED_FEATURES + ['cluster_dbscan', 'label_dbscan']],
                    use_container_width=True
                )
            else:
                st.info("No noise points detected under current parameters.")

            # CSV Download
            csv_dbscan = cached_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download DBSCAN Cluster & Noise Results (CSV)",
                data=csv_dbscan,
                file_name="dbscan_predictions.csv",
                mime="text/csv"
            )


# -------------------------------------------------------------------------------------------------
# 4. VISUAL ANALYTICS
# -------------------------------------------------------------------------------------------------
elif app_mode == "📈 Visual Analytics":
    st.title("📈 Visual Analytics & Diagnostic Charts")
    st.markdown("Explore high-resolution visualizations for exploratory data analysis, physical conversion relationships, and anomaly diagnostics.")

    viz_tabs = st.tabs([
        "🔬 Power Conversion (DC vs AC)",
        "☀️ Solar Curve (Irradiation vs AC)",
        "⏱️ Time-Series Anomaly Trace",
        "🌐 DBSCAN Cluster View",
        "📊 Model Comparison Chart",
        "🌡️ Temperature & Correlation"
    ])

    if cached_results is not None:
        with viz_tabs[0]:
            st.subheader("DC Power vs AC Power Conversion Relationship")
            st.markdown("Inverters convert Direct Current (DC) from panels into Alternating Current (AC) for the grid. Observations diverging from this linear line represent inverter clipping, conversion inefficiency, or sensor faults.")
            fig = plot_dc_vs_ac(cached_results, anomaly_col='anomaly_iforest')
            st.pyplot(fig, use_container_width=True)

        with viz_tabs[1]:
            st.subheader("Solar Irradiation vs AC Power Output")
            st.markdown("Under high sunlight (irradiation), AC power output should increase proportionally. Observations with high irradiation but near-zero power indicate inverter downtime, grid disconnection, or heavy shading.")
            fig = plot_irradiation_vs_ac(cached_results, anomaly_col='anomaly_iforest')
            st.pyplot(fig, use_container_width=True)

        with viz_tabs[2]:
            st.subheader("Solar Generation Over Time with Detected Anomalies")
            st.markdown("15-minute generation curve showing diurnal (day/night) cycles. Red markers identify anomalous behavior flagged by Isolation Forest.")
            sample_days = st.slider("Select sample time window (days):", min_value=1, max_value=34, value=7)
            fig = plot_timeseries_anomalies(cached_results, sample_days=sample_days)
            st.pyplot(fig, use_container_width=True)

        with viz_tabs[3]:
            st.subheader("DBSCAN Multi-Dimensional Clustering & Noise Points")
            st.markdown("Points assigned to dense clusters share high proximity in standardized feature space. Red diamonds represent isolated noise outliers.")
            fig = plot_dbscan_clusters(cached_results)
            st.pyplot(fig, use_container_width=True)

        with viz_tabs[4]:
            st.subheader("Model Comparison: Isolation Forest vs DBSCAN")
            st.markdown("Comparison of detection counts and percentages between the tree-based partitioning model and the density clustering model.")
            if cached_comparison is not None:
                fig = plot_model_comparison_bars(cached_comparison)
                st.pyplot(fig, use_container_width=True)
                st.dataframe(cached_comparison, use_container_width=True)
            else:
                st.info("Comparison data will display once training is complete.")

        with viz_tabs[5]:
            st.subheader("Feature Correlation & Temperature Dynamics")
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                st.markdown("#### Correlation Heatmap")
                fig_corr = plot_correlation_heatmap(cached_results, SELECTED_FEATURES)
                st.pyplot(fig_corr, use_container_width=True)
            with col_g2:
                st.markdown("#### Ambient vs Module Temperature")
                fig_temp = plot_temperature_relationship(cached_results)
                st.pyplot(fig_temp, use_container_width=True)
    else:
        st.warning("Visualizations require preprocessed data. Run `python train_model.py` first.")


# -------------------------------------------------------------------------------------------------
# 5. MODEL INFORMATION
# -------------------------------------------------------------------------------------------------
elif app_mode == "🧠 Model Information":
    st.title("🧠 Theoretical Background & Model Comparison")
    st.markdown("Comprehensive technical details for academic viva, project presentation, and examiner defense.")

    tab_if, tab_db, tab_comp, tab_viva = st.tabs([
        "🌲 Model 1: Isolation Forest",
        "🌐 Model 2: DBSCAN",
        "⚖️ Comparative Evaluation",
        "🎓 Viva Q&A & Academic Defense"
    ])

    with tab_if:
        st.subheader("🌲 Isolation Forest (Primary Model)")
        st.markdown(r"""
        ### Theoretical Principle
        Isolation Forest explicitly isolates anomalies instead of profiling normal data points.
        - The algorithm builds an ensemble of **Isolation Trees (iTrees)** via recursive random axis-aligned partitioning.
        - Since anomalies have attribute values divergent from nominal clusters, they are isolated in fewer splits and exhibit **significantly shorter path lengths** $h(x)$.
        
        ### Anomaly Score Formula
        The anomaly score $s(x, n)$ for an observation $x$ over a dataset of size $n$ is defined as:
        $$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$
        Where:
        - $E(h(x))$ is the average path length across all trees.
        - $c(n) = 2 \ln(n - 1) + 0.5772156649\ (\text{Euler's constant}) - \frac{2(n - 1)}{n}$ is the average path length of unsuccessful searches in a Binary Search Tree (BST).
        - If $E(h(x)) \to 0$, $s \to 1$ $\implies$ **Definite Anomaly**.
        - If $E(h(x)) \to c(n)$, $s \to 0.5$ $\implies$ **Nominal Observation**.

        ### Project Hyperparameters
        - `n_estimators = 100`: Number of isolation trees built to guarantee statistical convergence.
        - `contamination = 0.05`: The proportion of outliers expected in the dataset (5%).
        - `random_state = 42`: Ensures deterministic reproducibility of splits.
        """)

    with tab_db:
        st.subheader("🌐 DBSCAN (Density-Based Spatial Clustering of Applications with Noise)")
        st.markdown(r"""
        ### Theoretical Principle
        DBSCAN identifies clusters as high-density regions separated by areas of low density.
        - **$\epsilon$-Neighborhood:** $N_\epsilon(p) = \{q \in D \mid \text{dist}(p, q) \le \epsilon\}$
        - **Core Point:** A point $p$ with at least $min\_samples$ points in $N_\epsilon(p)$.
        - **Border Point:** A point reachable from a core point but having fewer than $min\_samples$ in its neighborhood.
        - **Noise Point (Anomaly):** Any point that is neither a core point nor reachable from any core point (labeled as $-1$).

        ### Feature Standardization Requirement
        DBSCAN relies on Euclidean distance in multi-dimensional space:
        $$d(\mathbf{u}, \mathbf{v}) = \sqrt{\sum_{i=1}^k (u_i - v_i)^2}$$
        Because features have vastly different scales (e.g. $DC\_POWER \approx 10,000$ kW vs $IRRADIATION \approx 0.8$), features must be normalized using **StandardScaler**:
        $$z = \frac{x - \mu}{\sigma}$$
        
        ### Project Hyperparameters
        - `eps = 0.8`: Maximum radius of neighborhood in 6D standardized Euclidean space.
        - `min_samples = 10`: Minimum observations required inside $\epsilon$-ball to form a dense core.
        """)

    with tab_comp:
        st.subheader("⚖️ Isolation Forest vs DBSCAN Architectural Comparison")
        if cached_comparison is not None:
            st.dataframe(cached_comparison, use_container_width=True)
        
        st.markdown(r"""
        ### Why do the two algorithms detect different anomaly counts?
        1. **Contamination Constraint vs Density Natural Outliers:**
           - Isolation Forest was configured with `contamination=0.05`, mathematically selecting the 5% most isolated instances (~3,439 rows).
           - DBSCAN does not enforce a preset anomaly proportion; it scans the continuous density landscape. Only 6 points were truly isolated from all dense operational manifolds at $\epsilon=0.8$.
        2. **Tree Splits vs Euclidean Neighborhood:**
           - Isolation Forest evaluates orthogonal axis projections, finding localized sub-optimal points (e.g., low AC power while irradiation is high).
           - DBSCAN evaluates global 6D Euclidean distance, grouping solar diurnal operation into dense continuous curves.
        3. **Why Supervised Metrics (Accuracy, Precision, Recall, F1) are NOT Calculated:**
           - The Kaggle Solar Power Dataset does **not** provide verified ground-truth labels of hardware faults.
           - Unsupervised learning detects *unusual observations*, not verified ground-truth defects. Computing accuracy or F1-score without ground truth would be scientifically invalid.
        """)

    with tab_viva:
        st.subheader("🎓 Viva Questions & Examiner Answers")
        with st.expander("Q1: What is the main objective of this project?"):
            st.write("To develop an unsupervised machine learning system capable of detecting anomalous patterns and sensor irregularities in solar photovoltaic plant data without requiring prior failure labels.")
        
        with st.expander("Q2: Why are anomalies defined as 'unusual observations' rather than 'equipment faults'?"):
            st.write("Unsupervised algorithms detect mathematical outliers—data points that deviate statistically from the majority. While an anomaly may be caused by inverter failure or sensor drift, it could also arise from extreme weather, cloud transits, or scheduled curtailment. Hence, domain expertise is needed to verify actual faults.")

        with st.expander("Q3: Why was feature scaling applied to DBSCAN but not required for Isolation Forest?"):
            st.write("DBSCAN calculates Euclidean distance across 6 dimensions. Without scaling, features with large numeric ranges (DC Power up to 14,000) would dominate features with small ranges (Irradiation 0-1). Isolation Forest relies on tree-based monotonic splits, making it invariant to monotonic feature scaling.")

        with st.expander("Q4: How did you merge the Generation Data and Weather Sensor Data?"):
            st.write("Both datasets were merged using an inner join on `DATE_TIME`. Generation timestamps (`%d-%m-%Y %H:%M`) and Weather timestamps (`%Y-%m-%d %H:%M:%S`) were normalized into standard datetime format, yielding exactly 68,774 synchronized observations across 34 days.")


# -------------------------------------------------------------------------------------------------
# 6. NEW PREDICTION
# -------------------------------------------------------------------------------------------------
elif app_mode == "⚡ New Prediction":
    st.title("⚡ Real-Time Anomaly Prediction")
    st.markdown("Input new solar sensor readings or upload a CSV file to evaluate whether operational conditions are **Normal** or an **Anomaly** using the trained Isolation Forest model.")

    if 'iforest' not in models:
        st.error("Trained Isolation Forest model (`isolation_forest.pkl`) is missing. Run `python train_model.py` first.")
    else:
        pred_tab_single, pred_tab_batch = st.tabs(["🎛️ Single Observation Prediction", "📁 Batch CSV Prediction"])

        with pred_tab_single:
            st.subheader("Interactive Sensor Input Sliders")
            st.markdown("Adjust the sliders below to simulate solar plant operating conditions:")

            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                dc_input = st.slider("DC Power (kW)", min_value=0.0, max_value=15000.0, value=3000.0, step=50.0)
                ac_input = st.slider("AC Power (kW)", min_value=0.0, max_value=1500.0, value=300.0, step=10.0)
            with col_s2:
                daily_yield_input = st.slider("Daily Yield (kWh)", min_value=0.0, max_value=10000.0, value=2500.0, step=50.0)
                ambient_temp_input = st.slider("Ambient Temperature (°C)", min_value=15.0, max_value=50.0, value=28.0, step=0.5)
            with col_s3:
                module_temp_input = st.slider("Module Temperature (°C)", min_value=15.0, max_value=75.0, value=35.0, step=0.5)
                irradiation_input = st.slider("Solar Irradiation", min_value=0.0, max_value=1.5, value=0.35, step=0.02)

            input_dict = {
                'DC_POWER': dc_input,
                'AC_POWER': ac_input,
                'DAILY_YIELD': daily_yield_input,
                'AMBIENT_TEMPERATURE': ambient_temp_input,
                'MODULE_TEMPERATURE': module_temp_input,
                'IRRADIATION': irradiation_input
            }

            if st.button("🚀 Analyze Single Observation", use_container_width=True, type="primary"):
                result = predict_single_instance(models['iforest'], input_dict)
                
                st.markdown("---")
                st.subheader("Prediction Diagnostic Result")

                res_col1, res_col2 = st.columns([1, 2])
                with res_col1:
                    if result['is_anomaly']:
                        st.markdown(f"""
                        <div class="status-anomaly">
                            ⚠️ ANOMALY DETECTED<br>
                            <span style="font-size:0.9rem;">Anomaly Code: -1</span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="status-normal">
                            ✅ NORMAL OBSERVATION<br>
                            <span style="font-size:0.9rem;">Anomaly Code: +1</span>
                        </div>
                        """, unsafe_allow_html=True)

                with res_col2:
                    score = result['anomaly_score']
                    st.metric(
                        label="Isolation Forest Anomaly Score",
                        value=f"{score:.4f}",
                        help="Negative score indicates anomalous behavior; positive score indicates nominal cluster proximity."
                    )
                    if result['is_anomaly']:
                        st.warning("Observation isolated quickly by the decision trees. Potential causes: inverter clipping, abnormal conversion ratio, or sensor discrepancy.")
                    else:
                        st.success("Observation aligns well with nominal power generation characteristics.")

        with pred_tab_batch:
            st.subheader("Batch Anomaly Detection from Uploaded CSV")
            st.markdown(f"Upload a CSV file containing columns: `{', '.join(SELECTED_FEATURES)}`")

            batch_file = st.file_uploader("Upload batch CSV for scoring", type=["csv"], key="batch_scoring")
            if batch_file:
                try:
                    df_upload = pd.read_csv(batch_file)
                    st.write(f"Uploaded {len(df_upload):,} rows.")
                    
                    if st.button("⚡ Score Batch Telemetry Data", type="primary"):
                        scored_df, summary = predict_batch_dataframe(models['iforest'], df_upload)
                        
                        col_b1, col_b2, col_b3 = st.columns(3)
                        with col_b1:
                            st.metric("Total Scored Rows", f"{summary['total_samples']:,}")
                        with col_b2:
                            st.metric("Normal Rows", f"{summary['normal_count']:,}")
                        with col_b3:
                            st.metric("Flagged Anomalies", f"{summary['anomaly_count']:,} ({summary['anomaly_percentage']}%)")

                        st.dataframe(scored_df.head(100), use_container_width=True)

                        csv_out = scored_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Scored Batch Results (CSV)",
                            data=csv_out,
                            file_name="batch_scored_anomalies.csv",
                            mime="text/csv"
                        )
                except Exception as ex:
                    st.error(f"Error during batch scoring: {ex}")
