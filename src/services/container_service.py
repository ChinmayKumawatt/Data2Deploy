"""
Service for creating containerized ML applications with trained models.
Generates a Docker container with a simple prediction API and UI.
"""

import io
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import yaml


def _create_dockerfile(task_type: str) -> str:
    """Generate a Dockerfile for the containerized app."""
    return """FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
"""


def _create_container_app(model_path: str, preprocessor_path: str, target_column: str, task_type: str, feature_names: list[str]) -> str:
    """Generate the FastAPI application code for the container."""
    
    feature_list_str = ', '.join([f'"{f}"' for f in feature_names])
    
    app_code = f'''"""
Auto-generated Prediction API
This API was automatically generated from a trained ML model.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import joblib
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict

# Load model and preprocessor
MODEL_PATH = "models/model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

# Configuration
FEATURE_NAMES = [{feature_list_str}]
TARGET_COLUMN = "{target_column}"
TASK_TYPE = "{task_type}"

app = FastAPI(
    title="ML Prediction API",
    description="Auto-generated API for making predictions with your trained model",
    version="1.0.0"
)

# Serve static files
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except:
    pass


@app.get("/", response_class=HTMLResponse)
async def serve_home():
    """Serve the prediction interface."""
    return """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Prediction API</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }}
        .container {{
            max-width: 600px;
            width: 100%;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 40px;
        }}
        h1 {{
            color: #333;
            margin-bottom: 10px;
            font-size: 28px;
        }}
        .subtitle {{
            color: #666;
            margin-bottom: 30px;
            font-size: 14px;
        }}
        .form-group {{
            margin-bottom: 20px;
        }}
        label {{
            display: block;
            margin-bottom: 8px;
            color: #333;
            font-weight: 600;
            font-size: 14px;
        }}
        input {{
            width: 100%;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 8px;
            font-size: 14px;
            transition: border-color 0.3s;
        }}
        input:focus {{
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }}
        button {{
            width: 100%;
            padding: 12px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }}
        button:disabled {{
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }}
        .result {{
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            display: none;
        }}
        .result.show {{
            display: block;
        }}
        .result-label {{
            color: #666;
            font-size: 14px;
            margin-bottom: 10px;
        }}
        .result-value {{
            font-size: 24px;
            font-weight: 700;
            color: #667eea;
        }}
        .error {{
            color: #e74c3c;
            padding: 12px;
            background: #fadbd8;
            border-radius: 8px;
            margin-top: 10px;
            display: none;
            font-size: 14px;
        }}
        .error.show {{
            display: block;
        }}
        .loading {{
            display: none;
            text-align: center;
            color: #667eea;
        }}
        .loading.show {{
            display: block;
        }}
        .spinner {{
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }}
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>ML Prediction API</h1>
        <p class="subtitle">Enter values for all features to get predictions</p>
        
        <form id="predictionForm">
            <div id="inputFields"></div>
            <button type="submit">Get Prediction</button>
        </form>
        
        <div class="error" id="error"></div>
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Processing...</p>
        </div>
        <div class="result" id="result">
            <div class="result-label">Prediction Result</div>
            <div class="result-value" id="resultValue"></div>
        </div>
    </div>

    <script>
        const FEATURES = [{feature_list_str}];
        const TASK_TYPE = "{task_type}";
        
        // Generate input fields
        function initializeForm() {{
            const container = document.getElementById('inputFields');
            FEATURES.forEach(feature => {{
                const group = document.createElement('div');
                group.className = 'form-group';
                group.innerHTML = `
                    <label for="${{feature}}">${{feature}}</label>
                    <input type="number" id="${{feature}}" name="${{feature}}" step="any" required placeholder="Enter value">
                `;
                container.appendChild(group);
            }});
        }}
        
        // Handle form submission
        document.getElementById('predictionForm').addEventListener('submit', async (e) => {{
            e.preventDefault();
            
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');
            const error = document.getElementById('error');
            
            loading.classList.add('show');
            result.classList.remove('show');
            error.classList.remove('show');
            
            const data = {{}};
            FEATURES.forEach(feature => {{
                data[feature] = parseFloat(document.getElementById(feature).value);
            }});
            
            try {{
                const response = await fetch('/api/predict', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify(data)
                }});
                
                if (!response.ok) {{
                    throw new Error('Prediction failed');
                }}
                
                const prediction = await response.json();
                document.getElementById('resultValue').textContent = 
                    typeof prediction.prediction === 'number' 
                        ? prediction.prediction.toFixed(4)
                        : prediction.prediction;
                result.classList.add('show');
            }} catch (err) {{
                error.textContent = 'Error: ' + err.message;
                error.classList.add('show');
            }} finally {{
                loading.classList.remove('show');
            }}
        }});
        
        // Initialize on load
        window.addEventListener('load', initializeForm);
    </script>
</body>
</html>"""'''
    
    return app_code


def _create_predict_handler(task_type: str) -> str:
    """Generate the prediction handler route."""
    return '''

@app.post("/api/predict")
async def predict(features: Dict):
    """
    Make a prediction based on input features.
    
    Expected input: JSON object with feature names as keys and numeric values.
    """
    try:
        # Validate input
        for feature in FEATURE_NAMES:
            if feature not in features:
                raise ValueError(f"Missing required feature: {feature}")
        
        # Create DataFrame with same structure as training data
        df = pd.DataFrame([features])
        
        # Apply preprocessing (scaling, encoding, etc.)
        try:
            df_processed = preprocessor.transform(df[FEATURE_NAMES])
        except Exception as e:
            # If preprocessing fails, use raw features
            df_processed = df[FEATURE_NAMES].values
        
        # Make prediction
        prediction = model.predict(df_processed)[0]
        
        return {
            "prediction": float(prediction),
            "task_type": TASK_TYPE
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}
'''


def _create_requirements() -> str:
    """Generate requirements.txt for the container."""
    return """fastapi==0.104.1
uvicorn[standard]==0.24.0
joblib==1.3.2
pandas==2.1.3
numpy==1.24.3
scikit-learn==1.3.2
xgboost==2.0.2
lightgbm==4.1.0
catboost==1.2.2
python-multipart==0.0.6
"""


def _create_docker_compose() -> str:
    """Generate docker-compose.yml for easy deployment."""
    return """version: '3.8'

services:
  ml-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
"""


def _create_readme(model_name: str, task_type: str, target_column: str) -> str:
    """Generate README.md with instructions."""
    return f"""# ML Prediction API

Auto-generated containerized ML application with your trained model.

## Model Information
- **Model Type**: {model_name}
- **Task Type**: {task_type.title()}
- **Target Column**: {target_column}

## Quick Start

### Using Docker Compose (Recommended)
```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000`

### Using Docker
```bash
docker build -t ml-api .
docker run -p 8000:8000 ml-api
```

### Local Development
```bash
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### GET /
Web interface for making predictions interactively.

### POST /api/predict
Make predictions programmatically.

**Request:**
```json
{{
  "feature_1": 1.5,
  "feature_2": 2.3,
  ...
}}
```

**Response:**
```json
{{
  "prediction": 42.5,
  "task_type": "{task_type}"
}}
```

### GET /api/health
Health check endpoint.

## Making Predictions

### From Browser
Navigate to `http://localhost:8000` and fill in the feature values.

### From Command Line
```bash
curl -X POST "http://localhost:8000/api/predict" \\
  -H "Content-Type: application/json" \\
  -d '{{"feature_1": 1.5, "feature_2": 2.3}}'
```

### From Python
```python
import requests
import json

features = {{
    "feature_1": 1.5,
    "feature_2": 2.3,
}}

response = requests.post(
    "http://localhost:8000/api/predict",
    json=features
)

prediction = response.json()
print(f"Prediction: {{prediction['prediction']}}")
```

## File Structure
```
.
+-- app.py                 # Main FastAPI application
+-- requirements.txt       # Python dependencies
+-- Dockerfile             # Docker container definition
+-- docker-compose.yml     # Docker Compose configuration
+-- models/
    +-- model.pkl         # Trained ML model
    +-- preprocessor.pkl  # Data preprocessor (scaler, encoder, etc.)
+-- README.md             # This file
```

## Notes
- The model and preprocessor are loaded at startup
- All requests are processed synchronously
- For production use, consider using Gunicorn or similar ASGI server

## Support
For issues or improvements, refer to the original model training logs.
"""


def create_container_package(
    run_id: str,
    model_path: str,
    preprocessor_path: str,
    target_column: str,
    task_type: str,
    feature_names: list[str],
    model_name: str = "trained_model"
) -> bytes:
    """
    Create a containerized ML application package.
    
    Args:
        run_id: The training run ID
        model_path: Path to the trained model file
        preprocessor_path: Path to the preprocessor file
        target_column: Name of the target column
        task_type: 'regression' or 'classification'
        feature_names: List of feature names used in the model
        model_name: User-friendly model name
        
    Returns:
        ZIP file bytes containing the containerized app
    """
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        
        # Create directory structure
        models_dir = tmppath / "models"
        models_dir.mkdir()
        
        # Copy model files
        shutil.copy(model_path, models_dir / "model.pkl")
        shutil.copy(preprocessor_path, models_dir / "preprocessor.pkl")
        
        # Generate application files
        app_code = _create_container_app(
            model_path=str(models_dir / "model.pkl"),
            preprocessor_path=str(models_dir / "preprocessor.pkl"),
            target_column=target_column,
            task_type=task_type,
            feature_names=feature_names
        )
        app_code += _create_predict_handler(task_type)
        
        # Write files with explicit UTF-8 encoding
        (tmppath / "app.py").write_text(app_code, encoding='utf-8')
        (tmppath / "Dockerfile").write_text(_create_dockerfile(task_type), encoding='utf-8')
        (tmppath / "docker-compose.yml").write_text(_create_docker_compose(), encoding='utf-8')
        (tmppath / "requirements.txt").write_text(_create_requirements(), encoding='utf-8')
        (tmppath / "README.md").write_text(_create_readme(model_name, task_type, target_column), encoding='utf-8')
        
        # Create .dockerignore
        (tmppath / ".dockerignore").write_text("""__pycache__
*.pyc
*.pyo
.git
.gitignore
*.md
.DS_Store
""", encoding='utf-8')
        
        # Create .gitignore
        (tmppath / ".gitignore").write_text("""__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
.venv
*.pkl
*.joblib
""", encoding='utf-8')
        
        # Create ZIP file
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in tmppath.rglob("*"):
                if file_path.is_file():
                    arcname = str(file_path.relative_to(tmppath))
                    # Read file as binary to avoid encoding issues
                    with open(file_path, 'rb') as f:
                        zipf.writestr(arcname, f.read())
        
        return zip_buffer.getvalue()
