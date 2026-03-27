import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import CarPredictionRequest, CarPredictionResponse
from api.redis_cache import RedisCache
from src.pipelines.predict_pipeline import PredictPipeline
from src.logger import logging

# Initialize FastAPI App
app = FastAPI(
    title="Used Car Dynamic Pricing Engine",
    description="ML-powered API with Redis caching for real-time car valuations.",
    version="1.0.0"
)

# CORS Setup (Crucial for Vercel Frontend to communicate with this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace "*" with your Vercel React/Next.js domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize singletons at startup
logging.info("Booting up the Pricing Engine API...")
predict_pipeline = PredictPipeline()
cache = RedisCache()

@app.get("/")
def read_root():
    return {"status": "Active", "model": "XGBoost Tuned", "caching": "Enabled (Fallback Mode)"}

@app.post("/predict", response_model=CarPredictionResponse)
def predict_car_price(request_data: CarPredictionRequest):
    try:
        input_dict = request_data.dict()
        
        # Check Redis Cache First
        cache_key = cache.generate_cache_key(input_dict)
        cached_price = cache.get_cached_prediction(cache_key)
        
        if cached_price is not None:
            return CarPredictionResponse(predicted_price=cached_price, source="cache")
        
        # Cache Miss: Run ML Pipeline
        # Convert dict to Pandas DataFrame (as expected by PredictPipeline)
        input_df = pd.DataFrame([input_dict])
        
        # Get prediction from our trained XGBoost model
        predicted_price = predict_pipeline.predict(input_df)
        final_price = round(predicted_price, 2)
        
        # Save to Cache for future requests
        cache.set_cached_prediction(cache_key, final_price)
        
        return CarPredictionResponse(predicted_price=final_price, source="model")

    except Exception as e:
        logging.error(f"API Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error during prediction.")