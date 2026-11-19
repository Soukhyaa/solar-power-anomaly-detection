"""
Test the Flask REST API & React Web Dashboard Endpoints
"""

import json
from server import app

client = app.test_client()

print("=" * 80)
print(" TESTING FLASK REST API & REACT DASHBOARD ENDPOINTS")
print("=" * 80)

# 1. Test Static Index Route
res = client.get("/")
print(f"GET / -> Status Code: {res.status_code}")
assert res.status_code == 200, "Frontend index.html failed to serve"
assert b"Solar Power Anomaly Detection" in res.data or b"Solar" in res.data, "Index page title missing"
assert b"root" in res.data, "React root element missing"
print("✓ React Frontend index.html served successfully.")

# 2. Test API Status
res = client.get("/api/status")
print(f"GET /api/status -> Status Code: {res.status_code}")
data = json.loads(res.data)
assert data["status"] == "healthy", "API not healthy"
assert data["models"]["isolation_forest"] is True, "Isolation Forest not loaded"
print("✓ /api/status passed.")

# 3. Test API KPIs
res = client.get("/api/kpis")
print(f"GET /api/kpis -> Status Code: {res.status_code}")
kpis = json.loads(res.data)
print(f"KPIs: Total={kpis['total_observations']}, IF Anomalies={kpis['isolation_forest']['anomaly_count']}, DBSCAN Noise={kpis['dbscan']['noise_count']}")
assert kpis["total_observations"] == 68774
assert kpis["isolation_forest"]["anomaly_count"] == 3439
assert kpis["dbscan"]["noise_count"] == 6
print("✓ /api/kpis passed.")

# 4. Test Inverters Matrix Endpoint (22 Inverters)
res = client.get("/api/inverters")
print(f"GET /api/inverters -> Status Code: {res.status_code}")
inverters = json.loads(res.data)
print(f"Loaded {len(inverters)} inverters. Top inverter: {inverters[0]['inverter_id']} with {inverters[0]['anomaly_count']} anomalies ({inverters[0]['anomaly_rate']}%)")
assert len(inverters) == 22, f"Expected 22 inverters, found {len(inverters)}"
print("✓ /api/inverters passed (all 22 inverters accounted for).")

# 5. Test Feature Stats Endpoint
res = client.get("/api/feature-stats")
print(f"GET /api/feature-stats -> Status Code: {res.status_code}")
fstats = json.loads(res.data)
assert len(fstats) == 6, f"Expected 6 features, got {len(fstats)}"
print(f"Features: {[f['feature'] for f in fstats]}")
print("✓ /api/feature-stats passed.")

# 6. Test Data Records (Pagination & Filter)
res = client.get("/api/data/records?page=1&limit=10&filter=anomalies")
print(f"GET /api/data/records?filter=anomalies -> Status Code: {res.status_code}")
recs = json.loads(res.data)
print(f"Total matching anomalies: {recs['total']}, Returned preview: {len(recs['records'])}")
assert recs['total'] == 3439, f"Expected 3,439 anomalies in data explorer, got {recs['total']}"
assert len(recs['records']) == 10
print("✓ /api/data/records passed.")

# 7. Test Single Prediction (Normal)
test_normal = {
    "DC_POWER": 4200.0,
    "AC_POWER": 410.0,
    "DAILY_YIELD": 3100.0,
    "AMBIENT_TEMPERATURE": 29.5,
    "MODULE_TEMPERATURE": 42.0,
    "IRRADIATION": 0.48
}
res = client.post("/api/predict", data=json.dumps(test_normal), content_type="application/json")
print(f"POST /api/predict (Normal) -> Status Code: {res.status_code}")
pred_res = json.loads(res.data)
print(f"Prediction: {pred_res['prediction']} (Code: {pred_res['anomaly_code']}), Score: {pred_res['anomaly_score']}")
assert pred_res["prediction"] == "Normal"
print("✓ Single normal prediction passed.")

# 8. Test Single Prediction (Anomaly)
test_anomaly = {
    "DC_POWER": 0.0,
    "AC_POWER": 0.0,
    "DAILY_YIELD": 150.0,
    "AMBIENT_TEMPERATURE": 35.0,
    "MODULE_TEMPERATURE": 64.0,
    "IRRADIATION": 0.95
}
res = client.post("/api/predict", data=json.dumps(test_anomaly), content_type="application/json")
print(f"POST /api/predict (Anomaly) -> Status Code: {res.status_code}")
pred_res = json.loads(res.data)
print(f"Prediction: {pred_res['prediction']} (Code: {pred_res['anomaly_code']}), Score: {pred_res['anomaly_score']}")
print(f"Diagnosis: {pred_res['engineering_diagnosis']}")
assert pred_res["prediction"] == "Anomaly"
print("✓ Single anomaly prediction passed.")

# 9. Test Chart Data
res = client.get("/api/charts/data")
print(f"GET /api/charts/data -> Status Code: {res.status_code}")
charts = json.loads(res.data)
assert "dc_vs_ac" in charts and "irr_vs_ac" in charts and "timeseries" in charts and "inverters_chart" in charts
print("✓ /api/charts/data passed.")

print("\n" + "=" * 80)
print(" ALL 9 REACT REST API ENDPOINTS TESTED & PASSED FLAWLESSLY!")
print("=" * 80)
