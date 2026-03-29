import os
import pandas as pd
import io
from fastapi import FastAPI, HTTPException, Security, Header, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from api.schemas import (
    CarPredictionRequest, 
    CarPredictionResponse, 
    BatchPredictionResponse, 
    BatchPredictionItem
)
from api.redis_cache import RedisCache
from src.pipelines.predict_pipeline import PredictPipeline
from src.logger import logging

# Load environment variables
load_dotenv()

# SECURITY SETUP: API KEY VALIDATION
API_SECRET_KEY = os.getenv("AEROPRICE_API_KEY")

def verify_api_key(x_api_key: str = Header(None)):
    if not x_api_key or x_api_key != API_SECRET_KEY:
        logging.warning("Unauthorized API access attempt blocked!")
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid or Missing API Key")
    return x_api_key

# INITIALIZE APP
app = FastAPI(
    title="AeroPrice Analytics API",
    description="Enterprise-grade ML API for B2B Car Dealerships.",
    version="1.1.1"
)

# CORS SETUP
allowed_origin = os.getenv("ALLOWED_ORIGIN", "*") 

app.add_middleware(
    CORSMiddleware,
    allow_origins=[allowed_origin],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*", "x-api-key"],
)

# Initialize singletons at startup
logging.info("Booting up the AeroPrice Enterprise Engine...")
predict_pipeline = PredictPipeline()
cache = RedisCache()

# ROUTES
@app.get("/")
def read_root():
    return {"status": "Enterprise Active", "model": "XGBoost Tuned", "features": ["B2C Prediction", "B2B Batch Inference"]}

@app.post("/predict", response_model=CarPredictionResponse)
def predict_car_price(
    request_data: CarPredictionRequest,
    api_key: str = Security(verify_api_key)
):
    try:
        input_dict = request_data.dict()
        
        # Check Redis Cache
        cache_key = cache.generate_cache_key(input_dict)
        cached_price = cache.get_cached_prediction(cache_key)
        
        if cached_price is not None:
            return CarPredictionResponse(predicted_price=cached_price, source="cache")
        
        # ML Pipeline
        input_df = pd.DataFrame([input_dict])
        predicted_price = predict_pipeline.predict(input_df)
        final_price = round(predicted_price, 2)
        
        cache.set_cached_prediction(cache_key, final_price)
        return CarPredictionResponse(predicted_price=final_price, source="model")

    except Exception as e:
        logging.error(f"Prediction Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict_cars(
    file: UploadFile = File(...),
    api_key: str = Security(verify_api_key)
):
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        if df.empty:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Basic verification: Check if required columns exist
        required_cols = ['year', 'odometer', 'manufacturer', 'model']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing required columns: {missing}")

        # Run Batch Prediction
        preds_output = predict_pipeline.predict(df)
        
        # Ensure predictions are always in list format for pandas series assignment
        if isinstance(preds_output, (float, int)):
            predictions = [preds_output]
        else:
            predictions = preds_output
            
        df['predicted_price'] = [round(p, 2) for p in predictions]
        
        # Calculate Analytics
        total_value = float(df['predicted_price'].sum())
        
        # Mocking Profitability Logic
        df = df.sort_values(by='predicted_price', ascending=False)
        results = []
        for i, row in df.iterrows():
            status = "Good"
            if i < len(df) * 0.1: status = "High Profit"
            elif i > len(df) * 0.9: status = "Loss Leader"
            
            results.append(BatchPredictionItem(
                id=int(i),
                manufacturer=str(row['manufacturer']),
                model=str(row['model']),
                year=int(row['year']),
                odometer=float(row['odometer']),
                predicted_price=float(row['predicted_price']),
                status=status
            ))

        return BatchPredictionResponse(
            total_inventory_value=total_value,
            top_profitable=[r for r in results if r.status == "High Profit"][:3],
            loss_leaders=[r for r in results if r.status == "Loss Leader"][:5],
            predictions=results,
            processed_count=len(results)
        )

    except Exception as e:
        logging.error(f"Batch Prediction Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"CSV Processing Error: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "awake", "engine": "Enterprise AI"}
