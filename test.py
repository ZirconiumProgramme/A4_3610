from fastapi.testclient import TestClient
from app import app
import pytest


# Initialize the test client
client = TestClient(app)


# 1. Test Health Check [cite: 88]
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


# 2. Successful Single Prediction [cite: 85]
def test_predict_single_success():
    payload = {
        "passenger_count": 1,
        "trip_distance": 2.5,
        "fare_amount": 15.0,
        "pickup_hour": 14,
        "trip_duration_minutes": 10.0,
        "VendorID": 1,
        "RatecodeID": 1,
        "PULocationID": 132,
        "DOLocationID": 230,
        "payment_type": 1,
        "extra": 0.5,
        "mta_tax": 0.5,
        "tolls_amount": 0.0,
        "improvement_surcharge": 0.3,
        "congestion_surcharge": 2.5,
        "Airport_fee": 0.0,
        "trip_speed_mph": 15.0,
        "pickup_day_of_week": 4,
        "PU_Borough": 1.0,
        "DO_Borough": 1.0,
        "is_weekend": 0,
        "log_trip_distance": 0.9,
        "fare_per_mile": 6.0,
        "fare_per_minute": 1.5,
        "total_amount": 18.8
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "predicted_tip_amount" in response.json()
    assert "prediction_id" in response.json()


# 3. Successful Batch Prediction [cite: 86]
def test_predict_batch_success():
    # Sending a list containing 2 identical records
    record = {
        "passenger_count": 1, "trip_distance": 1.0, "fare_amount": 10.0,
        "pickup_hour": 10, "trip_duration_minutes": 5.0, "VendorID": 1,
        "RatecodeID": 1, "PULocationID": 1, "DOLocationID": 1, "payment_type": 1,
        "extra": 0.0, "mta_tax": 0.0, "tolls_amount": 0.0, "improvement_surcharge": 0.0,
        "congestion_surcharge": 0.0, "Airport_fee": 0.0, "trip_speed_mph": 12.0,
        "pickup_day_of_week": 1, "PU_Borough": 0.0, "DO_Borough": 0.0,
        "is_weekend": 0, "log_trip_distance": 0.0, "fare_per_mile": 0.0,
        "fare_per_minute": 0.0, "total_amount": 10.0
    }
    response = client.post("/predict/batch", json=[record, record])
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2


# 4. Invalid Input Rejection (Out-of-range value) [cite: 87]
def test_invalid_input_rejection():
    # pickup_hour is 25 (valid range is 0-23)
    payload = {"passenger_count": 1, "trip_distance": 1.0, "fare_amount": 10.0, "pickup_hour": 25}
    response = client.post("/predict", json=payload)
    # Pydantic should return 422 for validation errors
    assert response.status_code == 422 


# 5. Edge Case: Zero distance trip [cite: 89]
def test_edge_case_zero_distance():
    # trip_distance is 0 (Pydantic constraint is gt=0)
    payload = {
        "passenger_count": 1,
        "trip_distance": 0, 
        "fare_amount": 1000.0, # Extreme fare value
        "pickup_hour": 12,
        "trip_duration_minutes": 1.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422