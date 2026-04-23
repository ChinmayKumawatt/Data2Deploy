"""
Example usage of the Data2Deploy Inference API.

This demonstrates how to use the new POST /api/predict and GET /api/health/predict endpoints.
"""

import requests
import json
from typing import Dict, Any


# Configuration
API_BASE_URL = "http://localhost:8000"
PREDICT_ENDPOINT = f"{API_BASE_URL}/api/predict"
HEALTH_ENDPOINT = f"{API_BASE_URL}/api/health/predict"


def check_model_health() -> Dict[str, Any]:
    """Check if the inference service and model are healthy.
    
    Returns:
        Health status response
    """
    try:
        response = requests.get(HEALTH_ENDPOINT)
        response.raise_for_status()
        health = response.json()
        
        print("🏥 Model Health Check:")
        print(f"  Status: {health['status']}")
        print(f"  Model Loaded: {health['model_loaded']}")
        print(f"  Model Version: {health['model_version']}")
        print(f"  Tracking URI: {health['mlflow_tracking_uri']}")
        print(f"  Timestamp: {health['timestamp']}")
        
        return health
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check failed: {e}")
        return None


def make_prediction(features: Dict[str, Any]) -> Dict[str, Any]:
    """Make a prediction using the production model.
    
    Args:
        features: Dictionary of features matching the training schema
        
    Returns:
        Prediction response with confidence and metadata
    """
    try:
        payload = {"features": features}
        
        response = requests.post(PREDICT_ENDPOINT, json=payload)
        response.raise_for_status()
        prediction = response.json()
        
        print("🎯 Prediction Result:")
        print(f"  Prediction: {prediction['prediction']}")
        print(f"  Confidence: {prediction['confidence_score']:.2%}")
        print(f"  Model: {prediction['model_name']} (v{prediction['model_version']})")
        print(f"  Task Type: {prediction['task_type']}")
        print(f"  Timestamp: {prediction['timestamp']}")
        
        return prediction
    except requests.exceptions.RequestException as e:
        print(f"❌ Prediction failed: {e}")
        return None


def batch_predict(features_list: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    """Make multiple predictions in batch.
    
    Args:
        features_list: List of feature dictionaries
        
    Returns:
        List of prediction responses
    """
    predictions = []
    for i, features in enumerate(features_list, 1):
        print(f"\n📊 Sample {i}:")
        pred = make_prediction(features)
        if pred:
            predictions.append(pred)
    
    return predictions


# Example 1: Housing Dataset (Regression)
# ========================================
def example_housing_dataset():
    """Example using housing dataset features."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Housing Price Prediction (Regression)")
    print("="*60)
    
    # Check model health
    health = check_model_health()
    if not health or not health.get("model_loaded"):
        print("⚠️  Model not available. Please train a model first.")
        return
    
    # Single prediction
    housing_features = {
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
    
    print("\n📍 Making single prediction...")
    make_prediction(housing_features)
    
    # Batch predictions
    print("\n📍 Making batch predictions...")
    batch_features = [
        {
            "longitude": -118.24,
            "latitude": 34.05,
            "housing_median_age": 15.0,
            "total_rooms": 3200.0,
            "total_bedrooms": 640.0,
            "population": 1200.0,
            "households": 450.0,
            "median_income": 3.5,
            "ocean_proximity": "NEAR OCEAN"
        },
        {
            "longitude": -117.23,
            "latitude": 32.72,
            "housing_median_age": 25.0,
            "total_rooms": 2500.0,
            "total_bedrooms": 500.0,
            "population": 950.0,
            "households": 380.0,
            "median_income": 4.2,
            "ocean_proximity": "1H OCEAN"
        },
    ]
    
    batch_predict(batch_features)


# Example 2: Classification Dataset
# ==================================
def example_classification():
    """Example using classification dataset (modify features as needed)."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Classification Task")
    print("="*60)
    
    # Check model health
    health = check_model_health()
    if not health or not health.get("model_loaded"):
        print("⚠️  Model not available. Please train a model first.")
        return
    
    # Classification features (customize based on your dataset)
    classification_features = {
        "feature_1": 1.5,
        "feature_2": 2.3,
        "feature_3": 0.8,
        # ... add other features matching your schema
    }
    
    print("\n🔍 Making classification prediction...")
    make_prediction(classification_features)


# Example 3: Error Handling
# =========================
def example_error_handling():
    """Example demonstrating error handling."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Error Handling")
    print("="*60)
    
    # Invalid features (missing required fields)
    print("\n❌ Attempt 1: Missing required features...")
    invalid_features = {
        "longitude": -122.23,
        # Missing other required fields
    }
    
    try:
        response = requests.post(PREDICT_ENDPOINT, json={"features": invalid_features})
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"   Expected error: {e.response.status_code} - {e.response.text}")
    
    # No model loaded
    print("\n❌ Attempt 2: Model not available...")
    try:
        response = requests.post(PREDICT_ENDPOINT, json={"features": {}})
        if response.status_code == 503:
            print(f"   Expected error: {response.status_code} - Model not found")
    except requests.exceptions.RequestException as e:
        print(f"   Error: {e}")


# Example 4: Integration with Monitoring
# =======================================
def get_prediction_statistics() -> Dict[str, Any]:
    """Get prediction statistics from the SQLite database.
    
    This would require adding a statistics endpoint to the API.
    For now, you can query the predictions.db file directly.
    """
    import sqlite3
    
    db_path = "predictions.db"
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute(
                """
                SELECT 
                    COUNT(*) as total_predictions,
                    COUNT(DISTINCT model_version) as unique_models,
                    AVG(confidence_score) as avg_confidence,
                    MIN(confidence_score) as min_confidence,
                    MAX(confidence_score) as max_confidence
                FROM predictions
                """
            )
            stats = dict(cursor.fetchone())
        
        print("\n📈 Prediction Statistics:")
        print(f"  Total Predictions: {stats['total_predictions']}")
        print(f"  Unique Models: {stats['unique_models']}")
        print(f"  Avg Confidence: {stats['avg_confidence']:.2%}")
        print(f"  Min Confidence: {stats['min_confidence']:.2%}")
        print(f"  Max Confidence: {stats['max_confidence']:.2%}")
        
        return stats
    except Exception as e:
        print(f"  Could not retrieve statistics: {e}")
        return {}


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Data2Deploy Inference API - Usage Examples")
    print("="*60)
    
    # Run examples
    example_housing_dataset()
    # example_classification()
    example_error_handling()
    
    # Get statistics
    get_prediction_statistics()
    
    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60)
