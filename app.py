import os
import shutil
from fastapi import FastAPI, BackgroundTasks, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse
import pprint

# Import the core logic from our dynamically created module
from src.automl_runner import setup_automl_run

app = FastAPI(title="AutoML Pipeline API", description="REST endpoints for triggering the Data2Deploy ML pipeline.", version="1.0.0")

# Create data dir if it doesn't exist
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


def background_automl_task(dataset_path: str, target_column: str, feature_mode: str, n_features: int):
    """
    Executes the DVC pipeline in the background so the HTTP response is not blocked.
    """
    try:
        print(f"[API] Starting background AutoML task for dataset: {dataset_path}")
        setup_automl_run(
            csv_path=dataset_path,
            target_column=target_column,
            feature_mode=feature_mode,
            n_features=n_features
        )
        print("[API] Background AutoML task completed successfully.")
    except Exception as e:
        print(f"[API] Background AutoML task failed: {str(e)}")


@app.post("/train")
async def trigger_training(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_column: str = Form(..., description="The name of the target column to predict"),
    feature_mode: str = Form("auto", description="Either 'auto' or 'manual'"),
    n_features: int = Form(5, description="Number of features to auto-select")
):
    """
    Endpoint that accepts a CSV dataset, saves it, and fires off the DVC pipeline in the background.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only .csv files are supported.")

    file_path = os.path.join(DATA_DIR, file.filename)

    # Save the file to disk
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Trigger background execution
    background_tasks.add_task(
        background_automl_task,
        dataset_path=file_path,
        target_column=target_column,
        feature_mode=feature_mode,
        n_features=n_features
    )

    return {
        "status": "success",
        "message": "Dataset uploaded successfully. AutoML pipeline has been triggered in the background.",
        "details": {
            "dataset_path": file_path,
            "target_column": target_column,
            "feature_mode": feature_mode,
            "n_features": n_features
        }
    }


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """
    A simple web UI to upload the CSV manually for testing.
    """
    html_content = """
    <html>
        <head>
            <title>AutoML Predictor Portal</title>
            <style>
                body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f7f6; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
                .container { background-color: #fff; padding: 40px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); width: 100%; max-width: 500px; }
                h1 { color: #333; margin-bottom: 5px; }
                p { color: #666; font-size: 14px; margin-bottom: 25px; }
                label { display: block; margin-top: 15px; font-weight: 600; color: #444; }
                input[type="text"], input[type="number"], select { width: 100%; padding: 10px; margin-top: 5px; border-radius: 6px; border: 1px solid #ccc; font-size: 14px; }
                input[type="file"] { margin-top: 10px; }
                input[type="submit"] { margin-top: 30px; background-color: #007bff; color: white; border: none; padding: 12px; width: 100%; border-radius: 6px; font-size: 16px; cursor: pointer; transition: 0.3s; }
                input[type="submit"]:hover { background-color: #0056b3; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>AutoML Studio</h1>
                <p>Upload a dataset and let the pipeline figure it out.</p>
                <form action="/train" enctype="multipart/form-data" method="post" target="_blank">
                    <label>CSV Dataset File:</label>
                    <input name="file" type="file" accept=".csv" required>

                    <label>Target Column Name:</label>
                    <input name="target_column" type="text" placeholder="e.g. target, Price, etc" required>

                    <label>Feature Mode:</label>
                    <select name="feature_mode">
                        <option value="auto">Auto</option>
                        <option value="manual">Manual (All but target)</option>
                    </select>

                    <label>Number of Features (Auto mode only):</label>
                    <input name="n_features" type="number" value="5">

                    <input type="submit" value="Start Pipeline Training">
                </form>
            </div>
        </body>
    </html>
    """
    return html_content
