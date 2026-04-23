"""Pydantic models for inference API input validation."""
from typing import Any, Dict

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Input validation for prediction requests matching training schema."""
    
    features: Dict[str, Any] = Field(
        ...,
        description="Feature dictionary with column names as keys and values matching training schema"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": {
                    "longitude": -122.23,
                    "latitude": 37.88,
                    "housing_median_age": 41.0,
                    "total_rooms": 880.0,
                    "total_bedrooms": 129.0,
                    "population": 322.0,
                    "households": 126.0,
                    "median_income": 8.3252,
                    "ocean_proximity": "NEAR BAY"
                }
            }
        }


class PredictionResponse(BaseModel):
    """Response model for prediction requests."""
    
    prediction: float | int = Field(..., description="Model prediction value")
    confidence_score: float = Field(..., description="Model confidence score (0-1 for classification, or relative)")
    model_version: str = Field(..., description="Version/ID of the model used for prediction")
    model_name: str = Field(..., description="Name of the model")
    task_type: str = Field(..., description="Type of task (classification or regression)")
    timestamp: str = Field(..., description="ISO format timestamp of prediction")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 206000.0,
                "confidence_score": 0.89,
                "model_version": "prod-2024-04-20-001",
                "model_name": "random_forest_regressor",
                "task_type": "regression",
                "timestamp": "2024-04-20T10:30:45.123456Z"
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str = Field(..., description="Health status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_version: str | None = Field(None, description="Current model version")
    mlflow_tracking_uri: str | None = Field(None, description="MLflow tracking URI")
    timestamp: str = Field(..., description="ISO format timestamp")
