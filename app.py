from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.services.automl_service import (
    get_experiment_compare_payload,
    list_run_summaries,
    load_run_summary,
    preview_dataset,
    rerun_existing_experiment,
    run_training_pipeline,
)
from src.services.container_service import create_container_package


BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(
    title="Data2Deploy AutoML Studio",
    description="Upload a dataset, train top models, compare predictions, and track experiments with MLflow.",
    version="3.0.0",
)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


def _render(request: Request, name: str, context: dict, status_code: int = 200):
    return templates.TemplateResponse(
        request=request,
        name=name,
        context={"request": request, **context},
        status_code=status_code,
    )


@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request, error: str | None = None):
    return _render(
        request,
        "index.html",
        {
            "title": "Data2Deploy AutoML Studio",
            "error": error,
            "experiments_count": len(list_run_summaries()),
        },
    )


@app.post("/preview")
async def dataset_preview(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported.")

    try:
        file_bytes = await file.read()
        return JSONResponse(preview_dataset(file_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/train", response_class=HTMLResponse)
async def trigger_training(
    request: Request,
    file: UploadFile = File(...),
    target_column: str = Form(...),
    feature_mode: str = Form("auto"),
    n_features: int = Form(5),
    selected_features: list[str] = Form(default=[]),
    test_size: float = Form(0.2),
    random_state: int = Form(42),
    top_k: int = Form(3),
    n_iter: int = Form(5),
    cv: int = Form(3),
):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported.")

    try:
        result = run_training_pipeline(
            file_bytes=await file.read(),
            filename=file.filename,
            target_column=target_column,
            feature_mode=feature_mode,
            n_features=n_features,
            selected_features=selected_features,
            training_options={
                "test_size": test_size,
                "random_state": random_state,
                "top_k": top_k,
                "n_iter": n_iter,
                "cv": cv,
            },
        )
        return _render(
            request,
            "results.html",
            {
                "title": f"Run {result['run_id']} Dashboard",
                "result": result,
            },
        )
    except Exception as exc:
        return _render(
            request,
            "index.html",
            {
                "title": "Data2Deploy AutoML Studio",
                "error": str(exc),
                "experiments_count": len(list_run_summaries()),
            },
            status_code=400,
        )


@app.get("/runs/{run_id}", response_class=HTMLResponse)
async def view_run(request: Request, run_id: str):
    try:
        result = load_run_summary(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return _render(
        request,
        "results.html",
        {
            "title": f"Run {run_id} Dashboard",
            "result": result,
        },
    )


@app.post("/runs/{run_id}/rerun", response_class=HTMLResponse)
async def rerun_experiment(
    request: Request,
    run_id: str,
    target_column: str = Form(...),
    feature_mode: str = Form(...),
    n_features: int = Form(5),
    selected_features: list[str] = Form(default=[]),
    test_size: float = Form(0.2),
    random_state: int = Form(42),
    top_k: int = Form(3),
    n_iter: int = Form(5),
    cv: int = Form(3),
):
    try:
        result = rerun_existing_experiment(
            source_run_id=run_id,
            target_column=target_column,
            feature_mode=feature_mode,
            n_features=n_features,
            selected_features=selected_features,
            training_options={
                "test_size": test_size,
                "random_state": random_state,
                "top_k": top_k,
                "n_iter": n_iter,
                "cv": cv,
            },
        )
        return _render(
            request,
            "results.html",
            {
                "title": f"Run {result['run_id']} Dashboard",
                "result": result,
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/experiments", response_class=HTMLResponse)
async def experiments_page(request: Request):
    experiments = list_run_summaries()
    return _render(
        request,
        "experiments.html",
        {
            "title": "Experiments",
            "experiments": experiments,
        },
    )


@app.get("/experiments/compare", response_class=HTMLResponse)
async def compare_experiments(request: Request, run_ids: list[str] = Query(default=[])):
    if not run_ids:
        return RedirectResponse(url="/experiments", status_code=302)

    payload = get_experiment_compare_payload(run_ids)
    return _render(
        request,
        "compare.html",
        {
            "title": "Compare Experiments",
            "comparison": payload,
        },
    )




@app.get("/runs/{run_id}/download-app")
async def download_containerized_app(run_id: str, model_index: int = 0):
    """
    Download a containerized ML application with the selected trained model.
    
    Args:
        run_id: The training run ID
        model_index: Index of the model to download (0 = best model)
    """
    import tempfile
    import io
    
    try:
        # Load run summary
        summary = load_run_summary(run_id)
        
        # Get model information
        evaluation_models = summary.get("evaluation_models", [])
        if not evaluation_models or model_index >= len(evaluation_models):
            raise HTTPException(status_code=400, detail="Invalid model index")
        
        selected_model = evaluation_models[model_index]
        model_path = selected_model["model_path"]
        model_name = selected_model.get("model_name", "trained_model")
        
        # Get preprocessor path
        run_dir = Path(summary.get("dataset_path", "")).parent.parent
        preprocessor_path = run_dir / "artifacts" / "data_transformation" / "preprocessor.pkl"
        
        if not Path(model_path).exists():
            raise HTTPException(status_code=400, detail="Model file not found")
        if not preprocessor_path.exists():
            raise HTTPException(status_code=400, detail="Preprocessor file not found")
        
        # Get training data structure
        run_summary_path = Path(f"runs/{run_id}/results/summary.json")
        import json
        with open(run_summary_path) as f:
            run_data = json.load(f)
        
        # Get feature names from comparison data
        comparison = run_data.get("comparison", {})
        top_models = comparison.get("top_models", [])
        
        if not top_models:
            raise HTTPException(status_code=400, detail="Could not determine features")
        
        # Get feature names from the first prediction row
        preview_rows = comparison.get("preview_rows", [])
        if preview_rows:
            feature_names = [key for key in preview_rows[0].keys() if key not in ["row_id", "actual"]]
            # Filter out model prediction columns
            feature_names = [f for f in feature_names if not f.endswith("_prediction")]
        else:
            feature_names = []
        
        # Create container package
        package_bytes = create_container_package(
            run_id=run_id,
            model_path=model_path,
            preprocessor_path=str(preprocessor_path),
            target_column=summary["target_column"],
            task_type=summary["task_type"],
            feature_names=feature_names,
            model_name=model_name
        )
        
        # Save to temporary file and return
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
            tmp.write(package_bytes)
            tmp_path = tmp.name
        
        filename = f"ml-model-{run_id[:8]}.zip"
        return FileResponse(
            path=tmp_path,
            media_type="application/zip",
            filename=filename,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/runs/{run_id}/delete")
async def delete_run(run_id: str):
    """Delete a run and all its associated files."""
    try:
        import shutil
        run_dir = Path(f"runs/{run_id}")
        
        if not run_dir.exists():
            raise HTTPException(status_code=404, detail="Run not found")
        
        # Delete the run directory and all its contents
        shutil.rmtree(run_dir)
        
        return JSONResponse(
            status_code=200,
            content={"message": f"Run {run_id} deleted successfully"}
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/health")
async def health_check():
    return {"status": "ok"}
