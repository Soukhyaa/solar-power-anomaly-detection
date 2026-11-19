"""
Solar Power Generation Anomaly Detection - Model Training & Evaluation Pipeline
Academic Mini-Project for TY Artificial Intelligence and Data Science (DYPCOE Akurdi)
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN

# Add local path to import utilities
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.preprocessing import (
    SELECTED_FEATURES,
    load_raw_datasets,
    preprocess_and_merge,
    prepare_features,
    scale_features
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


def run_training_pipeline(
    data_dir="data",
    models_dir="models",
    outputs_dir="outputs",
    graphs_dir="outputs/graphs"
):
    print("=" * 80)
    print(" SOLAR POWER GENERATION ANOMALY DETECTION - TRAINING PIPELINE")
    print("=" * 80)

    # 1. Verify Directories
    for d in [models_dir, outputs_dir, graphs_dir]:
        os.makedirs(d, exist_ok=True)

    # 2. Locate Data Files
    gen_candidates = [
        os.path.join(data_dir, "generation_data.csv"),
        os.path.join(data_dir, "Plant_1_Generation_Data.csv")
    ]
    weather_candidates = [
        os.path.join(data_dir, "weather_data.csv"),
        os.path.join(data_dir, "Plant_1_Weather_Sensor_Data.csv")
    ]

    gen_path = next((p for p in gen_candidates if os.path.exists(p)), None)
    weather_path = next((p for p in weather_candidates if os.path.exists(p)), None)

    if not gen_path or not weather_path:
        raise FileNotFoundError(f"Could not locate dataset files in '{data_dir}'. Ensure generation and weather CSVs are present.")

    print(f"\n[1/7] Loading datasets:")
    print(f"  - Generation file: {gen_path}")
    print(f"  - Weather file:    {weather_path}")

    df_gen, df_weather = load_raw_datasets(gen_path, weather_path)
    print(f"  - Raw Generation Shape: {df_gen.shape}")
    print(f"  - Raw Weather Shape:    {df_weather.shape}")

    # 3. Preprocessing & Merging
    print("\n[2/7] Preprocessing and merging on DATE_TIME...")
    merged_df, stats = preprocess_and_merge(df_gen, df_weather)
    print(f"  - Merged Clean Shape:   {merged_df.shape}")
    print(f"  - Total Observations:   {len(merged_df):,}")
    print(f"  - Selected Features ({len(SELECTED_FEATURES)}): {SELECTED_FEATURES}")

    # Extract feature matrix X
    X = prepare_features(merged_df, SELECTED_FEATURES)

    # 4. Exploratory Data Analysis & Visualization
    print("\n[3/7] Generating EDA visualizations...")
    plot_missing_values(merged_df[SELECTED_FEATURES], save_path=os.path.join(graphs_dir, "missing_values.png"))
    plot_feature_distributions(merged_df, SELECTED_FEATURES, save_path=os.path.join(graphs_dir, "feature_distributions.png"))
    plot_correlation_heatmap(merged_df, SELECTED_FEATURES, save_path=os.path.join(graphs_dir, "correlation_heatmap.png"))
    plot_temperature_relationship(merged_df, save_path=os.path.join(graphs_dir, "ambient_vs_module_temp.png"))
    print(f"  - Saved EDA graphs to: {graphs_dir}")

    # 5. Train Model 1: Isolation Forest
    print("\n[4/7] Training Primary Model: Isolation Forest...")
    iforest = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42,
        n_jobs=-1
    )
    t0 = time.time()
    iforest.fit(X)
    t_iforest = time.time() - t0

    iforest_preds = iforest.predict(X)
    iforest_scores = iforest.decision_function(X)

    merged_df['anomaly_iforest'] = iforest_preds
    merged_df['anomaly_score_iforest'] = np.round(iforest_scores, 4)
    merged_df['label_iforest'] = ["Anomaly" if p == -1 else "Normal" for p in iforest_preds]

    n_iforest_anom = int((iforest_preds == -1).sum())
    n_iforest_norm = int((iforest_preds == 1).sum())
    pct_iforest_anom = (n_iforest_anom / len(merged_df)) * 100

    print(f"  - Fit time:               {t_iforest:.2f}s")
    print(f"  - Normal Observations:    {n_iforest_norm:,} ({100 - pct_iforest_anom:.2f}%)")
    print(f"  - Detected Anomalies:     {n_iforest_anom:,} ({pct_iforest_anom:.2f}%)")

    # Save Isolation Forest Model
    iforest_model_path = os.path.join(models_dir, "isolation_forest.pkl")
    joblib.dump(iforest, iforest_model_path)
    print(f"  - Saved model to:         {iforest_model_path}")

    # 6. Train Model 2: DBSCAN
    print("\n[5/7] Training Comparison Model: DBSCAN...")
    print("  - Standardizing features using StandardScaler...")
    X_scaled, scaler = scale_features(X, fit=True)
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"  - Saved scaler to:        {scaler_path}")

    dbscan = DBSCAN(
        eps=0.8,
        min_samples=10,
        n_jobs=-1
    )
    t0 = time.time()
    dbscan_clusters = dbscan.fit_predict(X_scaled)
    t_dbscan = time.time() - t0

    merged_df['cluster_dbscan'] = dbscan_clusters
    merged_df['anomaly_dbscan'] = np.where(dbscan_clusters == -1, -1, 1)
    merged_df['label_dbscan'] = np.where(dbscan_clusters == -1, "Anomaly (Noise)", "Normal")

    n_dbscan_noise = int((dbscan_clusters == -1).sum())
    n_dbscan_clustered = int((dbscan_clusters != -1).sum())
    pct_dbscan_noise = (n_dbscan_noise / len(merged_df)) * 100

    print(f"  - Fit time:               {t_dbscan:.2f}s")
    print(f"  - Clustered Points:       {n_dbscan_clustered:,} ({100 - pct_dbscan_noise:.4f}%)")
    print(f"  - Noise Points (Outliers):{n_dbscan_noise:,} ({pct_dbscan_noise:.4f}%)")
    print(f"  - Number of Clusters:     {len(set(dbscan_clusters)) - (1 if -1 in dbscan_clusters else 0)}")

    # Save DBSCAN Model
    dbscan_model_path = os.path.join(models_dir, "dbscan.pkl")
    joblib.dump(dbscan, dbscan_model_path)
    print(f"  - Saved model to:         {dbscan_model_path}")

    # 7. Model Visualizations
    print("\n[6/7] Generating result visualizations...")
    plot_dc_vs_ac(merged_df, anomaly_col='anomaly_iforest', save_path=os.path.join(graphs_dir, "iforest_dc_vs_ac.png"))
    plot_irradiation_vs_ac(merged_df, anomaly_col='anomaly_iforest', save_path=os.path.join(graphs_dir, "iforest_irradiation_vs_ac.png"))
    plot_timeseries_anomalies(merged_df, date_col='DATE_TIME', value_col='AC_POWER', anomaly_col='anomaly_iforest', sample_days=7, save_path=os.path.join(graphs_dir, "iforest_timeseries_anomalies.png"))
    plot_anomaly_distribution_pie(n_iforest_norm, n_iforest_anom, model_name="Isolation Forest", save_path=os.path.join(graphs_dir, "iforest_distribution_pie.png"))
    plot_dbscan_clusters(merged_df, x_col='DC_POWER', y_col='AC_POWER', cluster_col='cluster_dbscan', save_path=os.path.join(graphs_dir, "dbscan_clusters.png"))

    # 8. Model Comparison Table and Chart
    comparison_data = [
        {
            "Algorithm": "Isolation Forest",
            "Category": "Tree-based Ensembles (Partitioning)",
            "Parameters": "n_estimators=100, contamination=0.05",
            "Scaling Required": "No (Tree Partition Invariant)",
            "Total Observations": len(merged_df),
            "Normal Count": n_iforest_norm,
            "Anomaly Count": n_iforest_anom,
            "Anomaly Percentage (%)": round(pct_iforest_anom, 4),
            "Detection Rationale": "Isolates anomalies quickly due to short average path length in binary trees.",
            "Key Advantage": "Scales gracefully to large multidimensional datasets, handles global outliers well.",
            "Key Limitation": "Contamination parameter must be preset prior to detection."
        },
        {
            "Algorithm": "DBSCAN",
            "Category": "Density-Based Spatial Clustering (Noise)",
            "Parameters": "eps=0.8, min_samples=10",
            "Scaling Required": "Yes (Euclidean Distance Sensitive)",
            "Total Observations": len(merged_df),
            "Normal Count": n_dbscan_clustered,
            "Anomaly Count": n_dbscan_noise,
            "Anomaly Percentage (%)": round(pct_dbscan_noise, 4),
            "Detection Rationale": "Flags points residing in sparse regions as noise (-1) failing min_samples in eps-neighborhood.",
            "Key Advantage": "Does not require specifying anomaly percentage beforehand; forms arbitrary shaped clusters.",
            "Key Limitation": "Sensitive to scale differences and uniform density assumptions across multi-sensor regimes."
        }
    ]
    comparison_df = pd.DataFrame(comparison_data)
    comparison_csv_path = os.path.join(outputs_dir, "model_comparison.csv")
    comparison_df.to_csv(comparison_csv_path, index=False)
    print(f"\n[7/7] Saved model comparison table to: {comparison_csv_path}")

    plot_model_comparison_bars(comparison_df, save_path=os.path.join(graphs_dir, "model_comparison_bar.png"))

    # Save full annotated dataset
    results_csv_path = os.path.join(outputs_dir, "anomaly_results.csv")
    # Export relevant subset to keep CSV size practical and fast to download
    export_cols = [
        'DATE_TIME', 'DC_POWER', 'AC_POWER', 'DAILY_YIELD',
        'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION',
        'anomaly_iforest', 'anomaly_score_iforest', 'label_iforest',
        'cluster_dbscan', 'anomaly_dbscan', 'label_dbscan'
    ]
    # Keep only available columns
    available_cols = [c for c in export_cols if c in merged_df.columns]
    merged_df[available_cols].to_csv(results_csv_path, index=False)
    print(f"  - Saved full anomaly predictions to: {results_csv_path}")

    print("\n" + "=" * 80)
    print(" TRAINING & EVALUATION COMPLETED SUCCESSFULLY")
    print(f" Total Rows Processed:        {len(merged_df):,}")
    print(f" Isolation Forest Anomalies:  {n_iforest_anom:,} ({pct_iforest_anom:.2f}%)")
    print(f" DBSCAN Noise Points:         {n_dbscan_noise:,} ({pct_dbscan_noise:.4f}%)")
    print("=" * 80)

    return {
        'merged_df': merged_df,
        'comparison_df': comparison_df,
        'iforest': iforest,
        'dbscan': dbscan,
        'scaler': scaler
    }


if __name__ == "__main__":
    run_training_pipeline()
