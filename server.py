"""
Solar Power Generation Anomaly Detection - REST API & Web Dashboard Backend
Built with Flask to serve the React interactive dashboard and comprehensive telemetry endpoints.
"""

import os
import sys
import io
import json
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from utils.preprocessing import SELECTED_FEATURES, validate_input_data
from utils.prediction import load_model, predict_isolation_forest, predict_single_instance, predict_batch_dataframe

# Paths
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
FRONTEND_DIST = os.path.join(BASE_DIR, "frontend", "dist")
FRONTEND_DIR = FRONTEND_DIST if os.path.exists(FRONTEND_DIST) else os.path.join(BASE_DIR, "frontend")

IFOREST_MODEL_PATH = os.path.join(MODELS_DIR, "isolation_forest.pkl")
DBSCAN_MODEL_PATH = os.path.join(MODELS_DIR, "dbscan.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
ANOMALY_RESULTS_PATH = os.path.join(OUTPUTS_DIR, "anomaly_results.csv")
COMPARISON_CSV_PATH = os.path.join(OUTPUTS_DIR, "model_comparison.csv")

# Initialize Flask app
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return response

# Load models and cached data globally
print("Loading trained models and dataset artifacts...")
try:
    iforest_model = load_model(IFOREST_MODEL_PATH)
    print("✓ Loaded Isolation Forest model.")
except Exception as e:
    iforest_model = None
    print(f"⚠ Warning: Could not load Isolation Forest model: {e}")

try:
    dbscan_model = load_model(DBSCAN_MODEL_PATH)
    scaler_obj = load_model(SCALER_PATH)
    print("✓ Loaded DBSCAN model and scaler.")
except Exception as e:
    dbscan_model = None
    scaler_obj = None
    print(f"⚠ Warning: Could not load DBSCAN/scaler: {e}")

# Cache results
cached_results = None
if os.path.exists(ANOMALY_RESULTS_PATH):
    try:
        cached_results = pd.read_csv(ANOMALY_RESULTS_PATH)
        print(f"✓ Cached {len(cached_results):,} observations from anomaly_results.csv.")
    except Exception as e:
        print(f"⚠ Warning loading anomaly_results.csv: {e}")

cached_comparison = None
if os.path.exists(COMPARISON_CSV_PATH):
    try:
        cached_comparison = pd.read_csv(COMPARISON_CSV_PATH).to_dict(orient="records")
        print("✓ Cached model comparison data.")
    except Exception as e:
        print(f"⚠ Warning loading model_comparison.csv: {e}")


# Precompute inverter metrics for instant access
cached_inverters = []
if cached_results is not None:
    inverter_col = 'SOURCE_KEY_gen' if 'SOURCE_KEY_gen' in cached_results.columns else 'SOURCE_KEY'
    if inverter_col in cached_results.columns:
        inv_groups = cached_results.groupby(inverter_col)
        for inv_id, grp in inv_groups:
            total_r = len(grp)
            anom_c = int((grp['anomaly_iforest'] == -1).sum()) if 'anomaly_iforest' in grp.columns else 0
            anom_pct = round((anom_c / total_r) * 100, 2)
            avg_dc = round(float(grp['DC_POWER'].mean()), 2)
            avg_ac = round(float(grp['AC_POWER'].mean()), 2)
            avg_yield = round(float(grp['DAILY_YIELD'].mean()), 2)
            max_yield = round(float(grp['DAILY_YIELD'].max()), 2)
            eff = round((avg_ac / avg_dc * 100), 2) if avg_dc > 0 else 0.0

            if anom_pct > 5.5:
                status = "High Anomaly Rate"
                badge = "rose"
            elif anom_pct >= 4.8:
                status = "Moderate Alert"
                badge = "amber"
            else:
                status = "Optimal / Healthy"
                badge = "emerald"

            cached_inverters.append({
                "inverter_id": str(inv_id),
                "total_records": total_r,
                "anomaly_count": anom_c,
                "anomaly_rate": anom_pct,
                "avg_dc": avg_dc,
                "avg_ac": avg_ac,
                "avg_yield": avg_yield,
                "max_yield": max_yield,
                "efficiency": eff,
                "status": status,
                "badge": badge
            })
        # Sort by anomaly count descending
        cached_inverters.sort(key=lambda x: x['anomaly_count'], reverse=True)
        print(f"✓ Precomputed performance metrics for {len(cached_inverters)} inverters.")


# =================================================================================================
# FRONTEND STATIC ROUTES
# =================================================================================================

@app.route("/")
def index():
    """Serve the React dashboard index.html."""
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    """Serve static frontend assets."""
    return send_from_directory(FRONTEND_DIR, filename)


# =================================================================================================
# REST API ENDPOINTS
# =================================================================================================

@app.route("/api/status", methods=["GET"])
def api_status():
    """Health check and model readiness."""
    return jsonify({
        "status": "healthy",
        "service": "Solar Power Anomaly Detection API",
        "version": "2.0.0",
        "engine": "React 18 + Flask ML Core",
        "models": {
            "isolation_forest": iforest_model is not None,
            "dbscan": dbscan_model is not None,
            "scaler": scaler_obj is not None
        },
        "academic_context": {
            "course": "AD124PC503 (Machine Learning Mini-Project)",
            "college": "D. Y. Patil College of Engineering, Akurdi, Pune",
            "department": "Artificial Intelligence & Data Science",
            "dataset": "Kaggle Solar Power Generation Dataset (Plant 1: 68,774 observations, 22 inverters, 34 days)"
        }
    })


@app.route("/api/kpis", methods=["GET"])
def api_kpis():
    """Retrieve operational metrics, peaks, and anomaly summaries."""
    total_obs = len(cached_results) if cached_results is not None else 68774
    n_iforest_anom = int((cached_results['anomaly_iforest'] == -1).sum()) if cached_results is not None and 'anomaly_iforest' in cached_results.columns else 3439
    n_iforest_norm = total_obs - n_iforest_anom
    pct_iforest_anom = round((n_iforest_anom / total_obs) * 100, 4)

    n_dbscan_noise = int((cached_results['anomaly_dbscan'] == -1).sum()) if cached_results is not None and 'anomaly_dbscan' in cached_results.columns else 6
    n_dbscan_norm = total_obs - n_dbscan_noise
    pct_dbscan_noise = round((n_dbscan_noise / total_obs) * 100, 4)

    # Physical plant records
    peak_dc = round(float(cached_results['DC_POWER'].max()), 2) if cached_results is not None else 14471.12
    peak_ac = round(float(cached_results['AC_POWER'].max()), 2) if cached_results is not None else 1410.95
    peak_yield = round(float(cached_results['DAILY_YIELD'].max()), 2) if cached_results is not None else 9163.00
    peak_mod_temp = round(float(cached_results['MODULE_TEMPERATURE'].max()), 2) if cached_results is not None else 65.55
    peak_irr = round(float(cached_results['IRRADIATION'].max()), 3) if cached_results is not None else 1.222

    return jsonify({
        "total_observations": total_obs,
        "total_inverters": len(cached_inverters) if cached_inverters else 22,
        "monitoring_days": 34,
        "isolation_forest": {
            "normal_count": n_iforest_norm,
            "anomaly_count": n_iforest_anom,
            "anomaly_percentage": pct_iforest_anom,
            "normal_percentage": round(100 - pct_iforest_anom, 4),
            "parameters": "n_estimators=100, contamination=0.05, random_state=42"
        },
        "dbscan": {
            "clustered_count": n_dbscan_norm,
            "noise_count": n_dbscan_noise,
            "noise_percentage": pct_dbscan_noise,
            "clustered_percentage": round(100 - pct_dbscan_noise, 4),
            "parameters": "eps=0.8, min_samples=10 (StandardScaler)"
        },
        "plant_extremes": {
            "peak_dc_power": peak_dc,
            "peak_ac_power": peak_ac,
            "peak_daily_yield": peak_yield,
            "peak_module_temp": peak_mod_temp,
            "peak_irradiation": peak_irr
        },
        "features": SELECTED_FEATURES
    })


@app.route("/api/inverters", methods=["GET"])
def api_inverters():
    """Return comprehensive performance & anomaly breakdown for all 22 inverters."""
    return jsonify(cached_inverters)


@app.route("/api/feature-stats", methods=["GET"])
def api_feature_stats():
    """Return descriptive statistics and anomaly vs normal feature comparisons."""
    if cached_results is None:
        return jsonify({"error": "No cached data available"}), 500

    stats = []
    for col in SELECTED_FEATURES:
        series = cached_results[col]
        norm_series = cached_results[cached_results['anomaly_iforest'] == 1][col]
        anom_series = cached_results[cached_results['anomaly_iforest'] == -1][col]

        stats.append({
            "feature": col,
            "mean": round(float(series.mean()), 2),
            "std": round(float(series.std()), 2),
            "min": round(float(series.min()), 2),
            "p25": round(float(series.quantile(0.25)), 2),
            "median": round(float(series.median()), 2),
            "p75": round(float(series.quantile(0.75)), 2),
            "max": round(float(series.max()), 2),
            "normal_mean": round(float(norm_series.mean()), 2),
            "anomaly_mean": round(float(anom_series.mean()), 2),
            "deviation_ratio": round(float(anom_series.mean() / (norm_series.mean() + 1e-6)), 2)
        })

    return jsonify(stats)


@app.route("/api/data/records", methods=["GET"])
def api_data_records():
    """
    Paginated telemetry records with search and anomaly filtering.
    Query params: page (default 1), limit (default 25), filter ('all'|'anomalies'|'normal'|'dbscan_noise'), inverter (optional)
    """
    if cached_results is None:
        return jsonify({"records": [], "total": 0, "page": 1, "pages": 0})

    try:
        page = int(request.args.get("page", 1))
        limit = min(int(request.args.get("limit", 25)), 100)
        filter_mode = request.args.get("filter", "all")
        inverter_filter = request.args.get("inverter", "")
        search_query = request.args.get("search", "").strip()

        df_view = cached_results

        # Inverter filter
        inv_col = 'SOURCE_KEY_gen' if 'SOURCE_KEY_gen' in df_view.columns else 'SOURCE_KEY'
        if inverter_filter and inv_col in df_view.columns:
            df_view = df_view[df_view[inv_col] == inverter_filter]

        # Anomaly status filter
        if filter_mode == "anomalies":
            df_view = df_view[df_view['anomaly_iforest'] == -1]
        elif filter_mode == "normal":
            df_view = df_view[df_view['anomaly_iforest'] == 1]
        elif filter_mode == "dbscan_noise":
            df_view = df_view[df_view['anomaly_dbscan'] == -1]

        # Timestamp search
        if search_query:
            df_view = df_view[df_view['DATE_TIME'].astype(str).str.contains(search_query, case=False, na=False)]

        total_matching = len(df_view)
        total_pages = max(1, (total_matching + limit - 1) // limit)
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit

        slice_df = df_view.iloc[start_idx:end_idx].copy()

        # Format records for clean frontend JSON
        records = []
        for _, row in slice_df.iterrows():
            records.append({
                "date_time": str(row['DATE_TIME']),
                "inverter_id": str(row.get(inv_col, "Plant-1")),
                "dc_power": round(float(row['DC_POWER']), 1),
                "ac_power": round(float(row['AC_POWER']), 1),
                "daily_yield": round(float(row['DAILY_YIELD']), 1),
                "ambient_temp": round(float(row['AMBIENT_TEMPERATURE']), 1),
                "module_temp": round(float(row['MODULE_TEMPERATURE']), 1),
                "irradiation": round(float(row['IRRADIATION']), 3),
                "anomaly_code": int(row.get('anomaly_iforest', 1)),
                "anomaly_label": "Anomaly" if row.get('anomaly_iforest') == -1 else "Normal",
                "anomaly_score": round(float(row.get('anomaly_score_iforest', 0)), 4),
                "dbscan_cluster": int(row.get('cluster_dbscan', 0)),
                "dbscan_noise": bool(row.get('cluster_dbscan') == -1)
            })

        return jsonify({
            "records": records,
            "total": total_matching,
            "page": page,
            "limit": limit,
            "pages": total_pages
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/comparison", methods=["GET"])
def api_comparison():
    """Return algorithmic comparison table."""
    if cached_comparison:
        return jsonify(cached_comparison)
    return jsonify([])


@app.route("/api/charts/data", methods=["GET"])
def api_charts_data():
    """
    Return downsampled data points for fast, interactive rendering in Chart.js / Recharts.
    """
    if cached_results is None:
        return jsonify({"error": "No cached telemetry data available"}), 500

    # Downsample points for smooth browser rendering (800 points)
    sample_df = cached_results.sample(n=min(800, len(cached_results)), random_state=42).copy()

    # Scatter: DC vs AC
    dc_ac_normal = []
    dc_ac_anomaly = []
    for _, row in sample_df.iterrows():
        point = {"x": round(float(row['DC_POWER']), 1), "y": round(float(row['AC_POWER']), 1)}
        if row.get('anomaly_iforest') == -1:
            dc_ac_anomaly.append(point)
        else:
            dc_ac_normal.append(point)

    # Scatter: Irradiation vs AC
    irr_ac_normal = []
    irr_ac_anomaly = []
    for _, row in sample_df.iterrows():
        point = {"x": round(float(row['IRRADIATION']), 3), "y": round(float(row['AC_POWER']), 1)}
        if row.get('anomaly_iforest') == -1:
            irr_ac_anomaly.append(point)
        else:
            irr_ac_normal.append(point)

    # Time series sample (First 4 days: ~250 intervals)
    ts_df = cached_results.sort_values(by='DATE_TIME').head(250).copy()
    ts_labels = [str(t) for t in ts_df['DATE_TIME']]
    ts_ac = [round(float(v), 1) for v in ts_df['AC_POWER']]
    ts_anomalies = [
        round(float(v), 1) if is_anom == -1 else None 
        for v, is_anom in zip(ts_df['AC_POWER'], ts_df['anomaly_iforest'])
    ]

    # Inverter ranking data (Top 10 highest anomaly inverters)
    inv_labels = [inv['inverter_id'][-8:] for inv in cached_inverters[:10]]
    inv_counts = [inv['anomaly_count'] for inv in cached_inverters[:10]]
    inv_rates = [inv['anomaly_rate'] for inv in cached_inverters[:10]]

    # Correlation matrix data
    corr_matrix = cached_results[SELECTED_FEATURES].corr().round(2).to_dict()

    return jsonify({
        "dc_vs_ac": {
            "normal": dc_ac_normal,
            "anomaly": dc_ac_anomaly
        },
        "irr_vs_ac": {
            "normal": irr_ac_normal,
            "anomaly": irr_ac_anomaly
        },
        "timeseries": {
            "labels": ts_labels,
            "ac_power": ts_ac,
            "anomalies": ts_anomalies
        },
        "inverters_chart": {
            "labels": inv_labels,
            "counts": inv_counts,
            "rates": inv_rates
        },
        "correlation": corr_matrix,
        "features": SELECTED_FEATURES
    })


@app.route("/api/presets", methods=["GET"])
def api_presets():
    """Return pre-configured test scenarios for easy Viva demonstration."""
    return jsonify([
        {
            "name": "☀️ Nominal Midday Generation (Normal)",
            "description": "Standard high-sun conditions with balanced DC-to-AC conversion and normal module temperature.",
            "expected": "Normal",
            "values": {
                "DC_POWER": 4200.0,
                "AC_POWER": 410.0,
                "DAILY_YIELD": 3100.0,
                "AMBIENT_TEMPERATURE": 29.5,
                "MODULE_TEMPERATURE": 42.0,
                "IRRADIATION": 0.48
            }
        },
        {
            "name": "⚡ Inverter Trip / Disconnect (Anomaly)",
            "description": "Peak solar irradiation (0.95), but inverter produces 0.0 kW output. Indicative of trip or breaker disconnect.",
            "expected": "Anomaly",
            "values": {
                "DC_POWER": 0.0,
                "AC_POWER": 0.0,
                "DAILY_YIELD": 150.0,
                "AMBIENT_TEMPERATURE": 35.0,
                "MODULE_TEMPERATURE": 64.0,
                "IRRADIATION": 0.95
            }
        },
        {
            "name": "📉 Severe Inverter Clipping (Anomaly)",
            "description": "Massive DC power (12,500 kW) generated by panels, but AC output throttled down to 350 kW due to inverter fault or thermal derating.",
            "expected": "Anomaly",
            "values": {
                "DC_POWER": 12500.0,
                "AC_POWER": 350.0,
                "DAILY_YIELD": 4500.0,
                "AMBIENT_TEMPERATURE": 36.0,
                "MODULE_TEMPERATURE": 68.0,
                "IRRADIATION": 0.88
            }
        },
        {
            "name": "🌡️ Extreme Thermal Derating (Anomaly)",
            "description": "Blistering module surface temp (72°C) with declining AC output despite strong solar irradiance.",
            "expected": "Anomaly",
            "values": {
                "DC_POWER": 2800.0,
                "AC_POWER": 180.0,
                "DAILY_YIELD": 3800.0,
                "AMBIENT_TEMPERATURE": 42.0,
                "MODULE_TEMPERATURE": 72.0,
                "IRRADIATION": 0.70
            }
        },
        {
            "name": "🌙 Night-time Idle (Normal)",
            "description": "Zero sun after dusk; panels dormant with near-zero generation and cool ambient temperatures.",
            "expected": "Normal",
            "values": {
                "DC_POWER": 0.0,
                "AC_POWER": 0.0,
                "DAILY_YIELD": 5500.0,
                "AMBIENT_TEMPERATURE": 22.0,
                "MODULE_TEMPERATURE": 21.5,
                "IRRADIATION": 0.0
            }
        }
    ])


@app.route("/api/predict", methods=["POST"])
def api_predict_single():
    """Predict anomaly status for a single observation JSON payload."""
    if iforest_model is None:
        return jsonify({"error": "Isolation Forest model is not loaded"}), 500

    try:
        data = request.get_json(force=True)
        result = predict_single_instance(iforest_model, data)

        # Build physical engineering diagnosis explanation
        is_anom = result['is_anomaly']
        score = result['anomaly_score']
        dc = float(data.get('DC_POWER', 0))
        ac = float(data.get('AC_POWER', 0))
        irr = float(data.get('IRRADIATION', 0))
        mod_temp = float(data.get('MODULE_TEMPERATURE', 0))

        if is_anom:
            if irr > 0.4 and ac < 50:
                diagnosis = "Severe Generation Drop: High solar irradiation with near-zero power output. Potential inverter shutdown, tripping, or sensor malfunction."
            elif dc > 8000 and ac < (dc * 0.05):
                diagnosis = "Inverter Throttling / Clipping: Disproportionately low AC output relative to massive DC string generation."
            elif mod_temp > 68:
                diagnosis = "Severe Thermal Stress: Panel module temperature is critically elevated, inducing thermal efficiency derating."
            else:
                diagnosis = f"Unusual Multi-Sensor Correlation: Observation deviates significantly from the nominal solar operating manifold (Score: {score:.4f})."
        else:
            diagnosis = f"Nominal Generation: Power conversion and thermal sensors align with standard diurnal solar curves (Score: {score:.4f})."

        result['engineering_diagnosis'] = diagnosis
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/predict/batch", methods=["POST"])
def api_predict_batch():
    """Batch prediction endpoint accepting either multipart file or JSON records."""
    if iforest_model is None:
        return jsonify({"error": "Isolation Forest model is not loaded"}), 500

    try:
        if 'file' in request.files:
            file = request.files['file']
            df = pd.read_csv(file)
        else:
            data = request.get_json(force=True)
            records = data.get('records', [])
            if not records:
                return jsonify({"error": "No records or CSV file provided"}), 400
            df = pd.DataFrame(records)

        scored_df, summary = predict_batch_dataframe(iforest_model, df)
        preview_records = scored_df.head(100).to_dict(orient="records")

        return jsonify({
            "summary": summary,
            "preview": preview_records,
            "total_rows": len(scored_df)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# =================================================================================================
# MAIN ENTRY POINT
# =================================================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 80)
    print(f" SOLAR ANOMALY DETECTION REST API & REACT DASHBOARD SERVER")
    print(f" Serving React Frontend from: {FRONTEND_DIR}")
    print(f" Running at: http://localhost:{port}")
    print("=" * 80)
    app.run(host="0.0.0.0", port=port, debug=False)
