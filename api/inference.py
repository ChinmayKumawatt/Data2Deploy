"""FastAPI inference endpoints for model predictions."""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError

from api.database import get_prediction_db
from api.models import HealthResponse, PredictionRequest, PredictionResponse
from src.utils.logger import logger


class ModelLoader:
    """Dependency for loading and caching MLflow models."""
    
    def __init__(
        self,
        mlflow_tracking_uri: str | None = None,
        production_stage: str = "production",
    ):
        """Initialize model loader.
        
        Args:
            mlflow_tracking_uri: MLflow tracking URI (uses local mlruns if None)
            production_stage: MLflow model stage to load (default: production)
        """
        self.mlflow_tracking_uri = mlflow_tracking_uri or "file:///mlruns"
        self.production_stage = production_stage
        self._model = None
        self._model_metadata = None
    
    def load_model(self) -> tuple[Any, Dict[str, Any]]:
        """Load the latest production model from MLflow.
        
        Returns:
            Tuple of (model, metadata)
            
        Raises:
            HTTPException: If model cannot be loaded
        """
        if self._model is not None:
            return self._model, self._model_metadata
        
        try:
            # Set tracking URI
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            
            # List available models
            registered_models = mlflow.search_registered_models()
            
            if not registered_models:
                logger.warning("No registered models found in MLflow")
                raise HTTPException(
                    status_code=503,
                    detail="No registered models found. Please train a model first."
                )
            
            # Find production model
            production_model = None
            for model in registered_models:
                for version in model.latest_versions:
                    if version.current_stage == self.production_stage:
                        production_model = model
                        break
                if production_model:
                    break
            
            if not production_model:
                logger.warning(
                    "No model found in %s stage. Available stages: %s",
                    self.production_stage,
                    [v.current_stage for m in registered_models for v in m.latest_versions]
                )
                raise HTTPException(
                    status_code=503,
                    detail=f"No model found in '{self.production_stage}' stage"
                )
            
            # Get the latest version in production stage
            latest_prod_version = None
            for version in production_model.latest_versions:
                if version.current_stage == self.production_stage:
                    latest_prod_version = version
                    break
            
            if not latest_prod_version:
                raise HTTPException(
                    status_code=503,
                    detail=f"Could not find latest {self.production_stage} version"
                )
            
            # Load model using MLflow
            model_uri = f"models:/{production_model.name}/{self.production_stage}"
            self._model = mlflow.sklearn.load_model(model_uri)
            
            # Capture metadata
            self._model_metadata = {
                "model_name": production_model.name,
                "model_version": latest_prod_version.version,
                "stage": latest_prod_version.current_stage,
                "uri": model_uri,
                "loaded_at": datetime.now(timezone.utc).isoformat(),
            }
            
            logger.info(
                "Loaded model: %s (version %s) from stage %s",
                production_model.name,
                latest_prod_version.version,
                self.production_stage
            )
            
            return self._model, self._model_metadata
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Error loading model from MLflow: %s", str(e))
            raise HTTPException(
                status_code=503,
                detail=f"Could not load model from MLflow: {str(e)}"
            ) from e
    
    def is_loaded(self) -> bool:
        """Check if model is currently loaded."""
        return self._model is not None
    
    def get_metadata(self) -> Optional[Dict[str, Any]]:
        """Get loaded model metadata."""
        return self._model_metadata


class InferenceService:
    """Service for making predictions with loaded models."""
    
    def __init__(self, model_loader: ModelLoader):
        """Initialize inference service.
        
        Args:
            model_loader: ModelLoader instance
        """
        self.model_loader = model_loader
        self.db = get_prediction_db()
    
    async def predict(
        self,
        request: PredictionRequest,
        task_type: str = "regression",
    ) -> PredictionResponse:
        """Make a prediction with the loaded model.
        
        Args:
            request: PredictionRequest with features
            task_type: Type of task (regression/classification)
            
        Returns:
            PredictionResponse with prediction and metadata
            
        Raises:
            HTTPException: If prediction fails
        """
        try:
            # Load model
            model, metadata = self.model_loader.load_model()
            
            # Prepare features
            features = request.features
            features_array = self._prepare_features(features)
            
            # Make prediction
            if task_type == "classification":
                prediction = model.predict(features_array)[0]
                # Get probability for confidence score
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(features_array)[0]
                    confidence_score = float(max(probabilities))
                else:
                    confidence_score = 1.0
            else:  # regression
                prediction = float(model.predict(features_array)[0])
                confidence_score = 1.0  # Regression models don't have confidence
            
            # Create timestamp
            timestamp = datetime.now(timezone.utc).isoformat()
            
            # Create response
            response = PredictionResponse(
                prediction=prediction,
                confidence_score=confidence_score,
                model_version=metadata["model_version"],
                model_name=metadata["model_name"],
                task_type=task_type,
                timestamp=timestamp,
            )
            
            # Log prediction asynchronously
            try:
                self.db.log_prediction(
                    model_version=metadata["model_version"],
                    model_name=metadata["model_name"],
                    task_type=task_type,
                    input_features=features,
                    prediction=prediction,
                    confidence_score=confidence_score,
                    timestamp=timestamp,
                )
            except Exception as e:
                logger.warning("Failed to log prediction to database: %s", str(e))
            
            return response
        
        except HTTPException:
            raise
        except ValidationError as e:
            logger.error("Feature validation error: %s", str(e))
            raise HTTPException(
                status_code=422,
                detail=f"Feature validation error: {str(e)}"
            ) from e
        except Exception as e:
            logger.error("Error during prediction: %s", str(e))
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {str(e)}"
            ) from e
    
    @staticmethod
    def _prepare_features(features: Dict[str, Any]) -> list[list[Any]]:
        """Prepare feature dictionary into array format for model.
        
        Args:
            features: Feature dictionary
            
        Returns:
            Feature array in shape (1, n_features)
        """
        # Convert features dict to ordered array
        feature_values = list(features.values())
        return [feature_values]


# Global model loader instance (singleton)
_model_loader: Optional[ModelLoader] = None


def get_model_loader(
    mlflow_tracking_uri: str | None = None,
) -> ModelLoader:
    """Get or create global model loader instance.
    
    Args:
        mlflow_tracking_uri: MLflow tracking URI
        
    Returns:
        ModelLoader instance
    """
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader(mlflow_tracking_uri=mlflow_tracking_uri)
    return _model_loader


def get_inference_service(
    model_loader: ModelLoader = Depends(get_model_loader),
) -> InferenceService:
    """Dependency to get inference service.
    
    Args:
        model_loader: Model loader instance
        
    Returns:
        InferenceService instance
    """
    return InferenceService(model_loader)


# Create router for inference endpoints
router = APIRouter(prefix="/api", tags=["inference"])


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    service: InferenceService = Depends(get_inference_service),
) -> PredictionResponse:
    """Make a prediction with the production model.
    
    Args:
        request: PredictionRequest with input features
        service: InferenceService instance
        
    Returns:
        PredictionResponse with prediction and metadata
    """
    return await service.predict(request, task_type="regression")


@router.get("/health/predict", response_model=HealthResponse)
async def health_check(
    model_loader: ModelLoader = Depends(get_model_loader),
) -> HealthResponse:
    """Health check endpoint for prediction service.
    
    Returns:
        HealthResponse with service status
    """
    try:
        is_loaded = model_loader.is_loaded()
        metadata = model_loader.get_metadata()
        
        if not is_loaded:
            # Try to load model to check availability
            try:
                _, metadata = model_loader.load_model()
                is_loaded = True
            except HTTPException as e:
                logger.warning("Model health check failed: %s", e.detail)
                return HealthResponse(
                    status="unhealthy",
                    model_loaded=False,
                    model_version=None,
                    mlflow_tracking_uri=model_loader.mlflow_tracking_uri,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
        
        return HealthResponse(
            status="healthy",
            model_loaded=is_loaded,
            model_version=metadata["model_version"] if metadata else None,
            mlflow_tracking_uri=model_loader.mlflow_tracking_uri,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    
    except Exception as e:
        logger.error("Error in health check: %s", str(e))
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            model_version=None,
            mlflow_tracking_uri=model_loader.mlflow_tracking_uri,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
