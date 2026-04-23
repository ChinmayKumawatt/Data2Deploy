# Data2Deploy Inference API Documentation

## Overview

The inference API provides fast, production-ready endpoints for making predictions using models trained and registered in MLflow. It includes:

- **POST /api/predict** - Make predictions with the production model
- **GET /api/health/predict** - Health check for model availability
- **SQLite Logging** - Automatic prediction logging for audit and analytics
- **Dependency Injection** - Efficient model loading with caching
- **Error Handling** - Comprehensive error messages and logging

## Quick Start

### Prerequisites

1. Train a model using the Data2Deploy training endpoint (`POST /train`)
2. Ensure MLflow is tracking the training run
3. Register the model in MLflow and tag it with "production" stage

### Making Your First Prediction

```bash
# Check model health
curl http://localhost:8000/api/health/predict

# Make a prediction
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

## API Endpoints

### 1. POST /api/predict

Make predictions with the production model.

**Request:**
```json
{
  "features": {
    "column_1": value1,
    "column_2": value2,
    "...": "..."
  }
}
```

**Response (200):**
```json
{
  "prediction": 206000.0,
  "confidence_score": 0.89,
  "model_version": "2",
  "model_name": "random_forest_regressor",
  "task_type": "regression",
  "timestamp": "2024-04-20T10:30:45.123456Z"
}
```

**Error Responses:**
- `400` - Invalid input features
- `422` - Feature validation error (missing required fields)
- `503` - Model not found or MLflow unavailable

**Features:**
- Input validation using Pydantic models matching training schema
- Automatic logging of all predictions to SQLite
- Support for both classification and regression models
- Confidence scores for classification (probability) and regression (1.0)

### 2. GET /api/health/predict

Health check endpoint to verify model availability.

**Response (200):**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "2",
  "mlflow_tracking_uri": "file:///mlruns",
  "timestamp": "2024-04-20T10:30:45.123456Z"
}
```

**Response (503 - Unhealthy):**
```json
{
  "status": "unhealthy",
  "model_loaded": false,
  "model_version": null,
  "mlflow_tracking_uri": "file:///mlruns",
  "timestamp": "2024-04-20T10:30:45.123456Z"
}
```

## Architecture

### Dependency Injection Pattern

The inference system uses FastAPI dependency injection for efficient model loading:

```python
# Model is loaded once and cached for all requests
model_loader = get_model_loader(mlflow_tracking_uri="file:///mlruns")

# Subsequent requests reuse the cached model
@app.post("/api/predict")
async def predict(
    request: PredictionRequest,
    service: InferenceService = Depends(get_inference_service),
):
    return await service.predict(request)
```

**Benefits:**
- Model loaded once at first request, reused for all subsequent requests
- No performance penalty for repeated predictions
- Clean separation of concerns

### Model Loading Flow

1. **Request arrives** → Dependency injection triggered
2. **Model loader checks cache** → Returns cached model if available
3. **First request only**: MLflow API queried for production model
4. **Model loaded** → Uses `mlflow.sklearn.load_model()`
5. **Metadata captured** → Version, name, stage stored
6. **Prediction made** → Using cached model
7. **Logged to SQLite** → Async logging (non-blocking)

### SQLite Database Schema

Predictions are automatically logged to `predictions.db`:

```sql
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,              -- ISO format timestamp of prediction
    model_version TEXT NOT NULL,          -- MLflow model version
    model_name TEXT NOT NULL,             -- Registered model name
    task_type TEXT NOT NULL,              -- 'classification' or 'regression'
    input_features TEXT NOT NULL,         -- JSON-encoded feature dict
    prediction REAL NOT NULL,             -- Prediction value
    confidence_score REAL NOT NULL,       -- Confidence/probability score
    created_at TEXT DEFAULT CURRENT_TIMESTAMP  -- Database record timestamp
);
```

**Query examples:**

```python
# Get recent predictions
SELECT * FROM predictions ORDER BY created_at DESC LIMIT 10;

# Get statistics
SELECT 
    COUNT(*) as total_predictions,
    COUNT(DISTINCT model_version) as unique_models,
    AVG(confidence_score) as avg_confidence
FROM predictions;

# Get predictions for a specific model
SELECT * FROM predictions WHERE model_version = '2' ORDER BY created_at DESC;
```

## Configuration

### MLflow Tracking URI

By default, the inference service uses local MLflow tracking:

```python
mlflow_tracking_uri = "file:///mlruns"  # Local file-based tracking
```

To use remote MLflow server:

```python
# In app.py
from api.inference import get_model_loader

# Configure remote MLflow
model_loader = get_model_loader(mlflow_tracking_uri="http://mlflow-server:5000")
```

### Database Path

By default, predictions are logged to `predictions.db` in the current directory:

```python
# In api/inference.py
db = get_prediction_db("predictions.db")  # Default

# Custom path
db = get_prediction_db("/var/log/ml_predictions.db")
```

## Error Handling

### Common Errors

**503 Service Unavailable - Model Not Found:**
```
Cause: No model registered in MLflow with "production" stage
Solution: Train and register a model using the /train endpoint
```

**422 Unprocessable Entity - Validation Error:**
```
Cause: Missing or invalid features in request
Solution: Check feature names match training schema
```

**400 Bad Request - Invalid Input:**
```
Cause: Malformed JSON or invalid data types
Solution: Ensure request format matches PredictionRequest schema
```

### Logging

All errors and predictions are logged to the application logs:

```
INFO: Loaded model: random_forest_regressor (version 2) from stage production
INFO: Prediction logged with ID: 42
WARNING: Failed to log prediction to database: ...
ERROR: Error loading model from MLflow: ...
```

## Performance Considerations

### Model Caching

Models are loaded once and cached in memory. For applications with memory constraints:

```python
# Clear cache between requests (not recommended)
model_loader._model = None
model_loader._model_metadata = None
```

### Batch Processing

For high-volume predictions, consider batch requests:

```python
# Efficient batch processing
for features in features_list:
    response = requests.post(endpoint, json={"features": features})
    predictions.append(response.json())
```

### Async Support

The prediction endpoint is async-ready:

```python
@app.post("/api/predict")
async def predict(
    request: PredictionRequest,
    service: InferenceService = Depends(get_inference_service),
) -> PredictionResponse:
    return await service.predict(request)
```

## Integration Examples

### Python Client

```python
import requests

endpoint = "http://localhost:8000/api/predict"

features = {
    "longitude": -122.23,
    "latitude": 37.88,
    # ... other features
}

response = requests.post(endpoint, json={"features": features})
prediction = response.json()

print(f"Prediction: {prediction['prediction']}")
print(f"Confidence: {prediction['confidence_score']}")
```

### JavaScript Client

```javascript
const endpoint = "http://localhost:8000/api/predict";

const features = {
  longitude: -122.23,
  latitude: 37.88,
  // ... other features
};

const response = await fetch(endpoint, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ features }),
});

const prediction = await response.json();
console.log(`Prediction: ${prediction.prediction}`);
```

### cURL

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d @features.json
```

## Backward Compatibility

The inference API is completely independent from existing endpoints:

- **Existing /train endpoint** - Unchanged ✅
- **Existing /preview endpoint** - Unchanged ✅
- **Existing /runs endpoints** - Unchanged ✅
- **New /api/predict endpoint** - Added ✅
- **New /api/health/predict endpoint** - Added ✅

All existing functionality remains fully operational.

## Troubleshooting

### Model not loading

**Check 1:** Verify model exists in MLflow
```python
import mlflow
mlflow.set_tracking_uri("file:///mlruns")
mlflow.search_registered_models()
```

**Check 2:** Verify model stage is "production"
```python
models = mlflow.search_registered_models()
for model in models:
    print(f"{model.name}: {[v.current_stage for v in model.latest_versions]}")
```

**Check 3:** Check health endpoint
```bash
curl http://localhost:8000/api/health/predict
```

### Feature validation errors

**Check 1:** Verify feature names match training schema
```bash
# Get schema from params.yaml or training output
cat params.yaml | grep -A 20 "schema:"
```

**Check 2:** Verify feature data types
```python
# Test with correct types
features = {
    "numeric_feature": 1.5,      # float
    "int_feature": 42,           # int
    "string_feature": "value"    # str
}
```

### Database issues

**Clear prediction log:**
```bash
rm predictions.db
```

The database will be automatically recreated on next prediction.

## Security Considerations

### Production Deployment

1. **Validate all inputs** - Pydantic handles this automatically
2. **Rate limiting** - Add middleware for rate limiting:
   ```python
   from slowapi import Limiter
   limiter = Limiter(key_func=get_remote_address)
   app.state.limiter = limiter
   ```

3. **Authentication** - Add to inference endpoints:
   ```python
   from fastapi.security import HTTPBearer
   security = HTTPBearer()
   
   @router.post("/predict")
   async def predict(..., credentials: HTTPAuthCredentials = Depends(security)):
       # Validate token
   ```

4. **HTTPS only** - Use in production environment
5. **Logging** - Monitor `predictions.db` for anomalies

## Advanced Usage

### Custom Model Loading

To load from a specific model stage:

```python
# In app.py
from api.inference import ModelLoader

loader = ModelLoader(production_stage="staging")  # Instead of "production"
```

### Adding Custom Preprocessing

```python
# Extend InferenceService
class CustomInferenceService(InferenceService):
    def _prepare_features(self, features):
        # Custom preprocessing logic
        return super()._prepare_features(features)
```

### Monitoring Predictions

Query prediction statistics:

```python
from api.database import get_prediction_db

db = get_prediction_db()
stats = db.get_statistics()
print(f"Total predictions: {stats['total_predictions']}")
print(f"Avg confidence: {stats['avg_confidence']}")
```

## Testing

Run the example script:

```bash
python examples/inference_api_example.py
```

This will:
1. Check model health
2. Make sample predictions
3. Demonstrate error handling
4. Show database statistics

## Support

For issues or questions:

1. Check the [troubleshooting section](#troubleshooting)
2. Review logs at `logs/` directory
3. Check predictions database at `predictions.db`
4. Verify MLflow setup at `mlruns/` directory
