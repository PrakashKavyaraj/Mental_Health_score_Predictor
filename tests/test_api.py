import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

SAMPLE_PAYLOAD = {
    "age": 21,
    "gender": "Male",
    "country": "India",
    "academic_level": "Undergraduate",
    "most_used_platform": "Instagram",
    "purpose_of_use": "Entertainment",
    "avg_daily_usage_hours": 6.0,
    "daily_unlocks": 180,
    "study_hours": 3.0,
    "physical_activity_hours": 1.0,
    "sleep_hours_per_night": 6.5,
    "stress_level": "High"
}

def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "LightGBM" in data["model_name"]
    assert data["test_r2"] > 0.80

def test_predict_endpoint():
    response = client.post("/api/v1/predict", json=SAMPLE_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert 1.0 <= data["predicted_score"] <= 10.0
    assert "confidence_interval" in data
    assert data["confidence_interval"]["lower"] < data["confidence_interval"]["upper"]
    assert "wellness_band" in data

def test_explain_endpoint():
    response = client.post("/api/v1/explain", json=SAMPLE_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert "positive_drivers" in data
    assert "negative_detractors" in data
    assert "actionable_recommendations" in data
    assert len(data["actionable_recommendations"]) > 0

def test_simulate_endpoint():
    sim_payload = {
        "baseline": SAMPLE_PAYLOAD,
        "changes": {
            "avg_daily_usage_hours": 3.0,
            "sleep_hours_per_night": 8.0
        }
    }
    response = client.post("/api/v1/simulate", json=sim_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["simulated_score"] > data["baseline_score"]
    assert data["score_delta"] > 0
    assert "delta_summary" in data

def test_cohort_endpoint():
    response = client.get("/api/v1/cohort")
    assert response.status_code == 200
    data = response.json()
    assert data["sample_count"] == 5000
    assert "target_stats" in data
    assert "feature_averages" in data

def test_legacy_predict_endpoint():
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_mental_health_score" in data
    assert 1.0 <= data["predicted_mental_health_score"] <= 10.0

def test_validation_error_handling():
    invalid_payload = SAMPLE_PAYLOAD.copy()
    invalid_payload["avg_daily_usage_hours"] = 35.0  # > 24 hours impossible
    response = client.post("/api/v1/predict", json=invalid_payload)
    assert response.status_code == 422
