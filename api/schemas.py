from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class CarPredictionRequest(BaseModel):
    # Core numerical features
    year: int = Field(..., ge=1990, le=datetime.now().year + 1, description="Manufacturing year of the car")
    odometer: float = Field(..., ge=0, description="Miles driven")
    
    # Categorical features (with explicit examples for Swagger UI)
    manufacturer: str = Field(..., example="honda")
    model: str = Field(..., example="civic")
    condition: Optional[str] = Field('unknown', example="excellent")
    cylinders: Optional[str] = Field('unknown', example="4 cylinders")
    fuel: Optional[str] = Field('gas', example="gas")
    title_status: Optional[str] = Field('clean', example="clean")
    transmission: Optional[str] = Field('automatic', example="automatic")
    drive: Optional[str] = Field('fwd', example="fwd")
    size: Optional[str] = Field('compact', example="compact")
    type: Optional[str] = Field('sedan', example="sedan")
    paint_color: Optional[str] = Field('unknown', example="black")
    state: str = Field(..., example="ca")
    region: str = Field(..., example="los angeles")

class CarPredictionResponse(BaseModel):
    predicted_price: float
    currency: str = "USD"
    source: str = "model" # Will return "cache" if data comes from Redis