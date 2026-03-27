import os
import pandas as pd
from fastapi import FastAPI, HTTPException, Security, Header
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from api.schemas import CarPredictionRequest, CarPredictionResponse
from api.redis_cache import RedisCache
from src.pipelines.predict_pipeline import PredictPipeline
from src.logger import logging

# Load environment variables
load_dotenv()

# SECURITY SETUP: API KEY VALIDATION
# Fetch secret from .env (Hugging Face Secrets later). Fallback is for local dev only.
API_SECRET_KEY = os.getenv("AEROPRICE_API_KEY")

def verify_api_key(x_api_key: str = Header(None)):
    if not x_api_key or x_api_key != API_SECRET_KEY:
        logging.warning("Unauthorized API access attempt blocked!")
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid or Missing API Key")
    return x_api_key

# INITIALIZE APP
app = FastAPI(
    title="AeroPrice Analytics API",
    description="ML-powered API with Redis caching and strict API Key security.",
    version="1.0.0"
)

# CORS SETUP (Production Ready)
# In production, set ALLOWED_ORIGIN in your .env to your Vercel URL
allowed_origin = os.getenv("ALLOWED_ORIGIN", "*") 

app.add_middleware(
    CORSMiddleware,
    allow_origins=[allowed_origin],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*", "x-api-key"],
)

# Initialize singletons at startup
logging.info("Booting up the Secure Pricing Engine API...")
predict_pipeline = PredictPipeline()
cache = RedisCache()

# ROUTES
@app.get("/")
def read_root():
    return {"status": "Active", "model": "XGBoost Tuned", "security": "API Key Required"}

@app.post("/predict", response_model=CarPredictionResponse)
def predict_car_price(
    request_data: CarPredictionRequest,
    api_key: str = Security(verify_api_key)
):
    try:
        input_dict = request_data.dict()
        
        # Check Redis Cache First
        cache_key = cache.generate_cache_key(input_dict)
        cached_price = cache.get_cached_prediction(cache_key)
        
        if cached_price is not None:
            return CarPredictionResponse(predicted_price=cached_price, source="cache")
        
        # Cache Miss: Run ML Pipeline
        input_df = pd.DataFrame([input_dict])
        predicted_price = predict_pipeline.predict(input_df)
        final_price = round(predicted_price, 2)
        
        # Save to Cache for future requests
        cache.set_cached_prediction(cache_key, final_price)
        
        return CarPredictionResponse(predicted_price=final_price, source="model")

    except Exception as e:
        logging.error(f"API Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error during prediction.")

# Health Check Endpoint (To keep awake api on Hugging Face)
@app.get("/health")
def health_check():
    return {"status": "awake", "message": "AeroPrice API is running!"}
