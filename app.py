import os  
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import mlflow.sklearn
import uuid
from typing import List


TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
mlflow.set_tracking_uri(TRACKING_URI)


app = FastAPI()


# --- Model Configuration ---
MODEL_NAME = "taxi-tip-regressor"
MODEL_VERSION = 1
FEATURE_NAMES = [
    'VendorID', 'passenger_count', 'trip_distance', 'RatecodeID', 'PULocationID', 
    'DOLocationID', 'payment_type', 'fare_amount', 'extra', 'mta_tax', 
    'tolls_amount', 'improvement_surcharge', 'congestion_surcharge', 'Airport_fee', 
    'trip_duration_minutes', 'trip_speed_mph', 'pickup_hour', 'pickup_day_of_week', 
    'PU_Borough', 'DO_Borough', 'is_weekend', 'log_trip_distance', 
    'fare_per_mile', 'fare_per_minute', 'total_amount'
]


print(f"Loading registered model: {MODEL_NAME} (v{MODEL_VERSION})")
model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{MODEL_VERSION}")


# --- Schema Definition ---
class TaxiTrip(BaseModel):
    passenger_count: int = Field(..., ge=1)
    trip_distance: float = Field(..., gt=0)
    fare_amount: float = Field(..., ge=0)
    pickup_hour: int = Field(..., ge=0, le=23)
    trip_duration_minutes: float = Field(..., gt=0)
    VendorID: int = 1
    RatecodeID: int = 1
    PULocationID: int = 1
    DOLocationID: int = 1
    payment_type: int = 1
    extra: float = 0.0
    mta_tax: float = 0.5
    tolls_amount: float = 0.0
    improvement_surcharge: float = 0.3
    congestion_surcharge: float = 0.0
    Airport_fee: float = 0.0
    trip_speed_mph: float = 15.0
    pickup_day_of_week: int = 0
    PU_Borough: float = 0.0
    DO_Borough: float = 0.0
    is_weekend: int = 0
    log_trip_distance: float = 0.0
    fare_per_mile: float = 0.0
    fare_per_minute: float = 0.0
    total_amount: float = 0.0


# --- Global Exception Handler (Task 2.2) ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"message": "An unexpected error occurred. Please try again later."},
    )


# --- Endpoints ---

@app.get("/health") #
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_version": MODEL_VERSION
    }


@app.get("/model/info") #
async def model_info():
    return {
        "model_name": MODEL_NAME,
        "version": MODEL_VERSION,
        "feature_names": FEATURE_NAMES,
        "training_metrics": {
            "MAE": 1.21,
            "RMSE": 2.36,
            "R2": 0.62
        }
    }


@app.post("/predict")
async def predict(trip: TaxiTrip):
    # Extract values in correct order
    input_data = [[getattr(trip, name) for name in FEATURE_NAMES]]
    prediction = model.predict(input_data)[0]
    
    return {
        "prediction_id": str(uuid.uuid4()),
        "predicted_tip_amount": round(float(prediction), 2),
        "model_version": MODEL_VERSION
    }


@app.post("/predict/batch") #
async def predict_batch(trips: List[TaxiTrip]):
    if len(trips) > 100:
        raise HTTPException(status_code=400, detail="Batch size cannot exceed 100 records")
    
    # Prepare all records for model
    all_inputs = [[getattr(trip, name) for name in FEATURE_NAMES] for trip in trips]
    predictions = model.predict(all_inputs)
    
    results = []
    for pred in predictions:
        results.append({
            "prediction_id": str(uuid.uuid4()),
            "predicted_tip_amount": round(float(pred), 2)
        })
        
    return {
        "model_version": MODEL_VERSION,
        "predictions": results
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)