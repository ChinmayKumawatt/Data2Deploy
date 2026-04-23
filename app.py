from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from api.inference import router as inference_router
from src.services.automl_service import (
    get_experiment_compare_payload,
    list_run_summaries,
    load_run_summary,
    preview_dataset,
    rerun_existing_experiment,
    run_training_pipeline,
)
from src.services.container_service import create_container_package
from src.services.eda_service import EDAService, create_eda_from_dataframe


BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(
    title="Data2Deploy AutoML Studio",
    description="Upload a dataset, train top models, compare predictions, and track experiments with MLflow.",
    version="3.0.0",
)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Include inference router
app.include_router(inference_router)


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


@app.get("/predict", response_class=HTMLResponse)
async def predict_page(request: Request):
    return _render(
        request,
        "predict.html",
        {
            "title": "Make Predictions",
        },
    )


@app.get("/api/predict-schema")
async def get_predict_schema():
    """Get the feature schema for predictions."""
    try:
        from src.utils.config import load_stage_config
        
        config = load_stage_config("validation")
        schema = getattr(config, "schema", {})
        
        # Also get target column and task type
        ingestion_config = load_stage_config("ingestion")
        target_column = getattr(ingestion_config, "target_column", None)
        task_type = getattr(ingestion_config, "task_type", "regression")
        
        return {
            "columns": schema.get("columns", {}),
            "target_column": target_column,
            "task_type": task_type,
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Could not load schema: {str(e)}"
        ) from e


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


# ==================== EDA / Insight Studio Routes ====================

# Store current dataset in session for EDA (in production, use proper session management)
_eda_session = {}


@app.get("/insight-studio", response_class=HTMLResponse)
async def insight_studio(request: Request, run_id: str = None):
    """Insight Studio page for EDA."""
    try:
        if run_id:
            # Load dataset from run
            summary = load_run_summary(run_id)
            dataset_path = summary.get("dataset_path")
            
            if not dataset_path or not Path(dataset_path).exists():
                raise HTTPException(status_code=400, detail="Dataset not found")
            
            # Store in session
            _eda_session['current_dataset'] = dataset_path
            _eda_session['run_id'] = run_id
        elif 'current_dataset' not in _eda_session:
            raise HTTPException(status_code=400, detail="No dataset loaded")
        
        return _render(
            request,
            "insight_studio.html",
            {
                "title": "Insight Studio",
                "run_id": run_id or _eda_session.get("run_id", "uploaded"),
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/eda/columns")
async def eda_get_columns():
    """Get column names from current dataset."""
    try:
        if 'current_dataset' not in _eda_session:
            raise HTTPException(status_code=400, detail="No dataset loaded")
        
        import pandas as pd
        df = pd.read_csv(_eda_session['current_dataset'])
        
        return {
            "columns": df.columns.tolist(),
            "numeric": df.select_dtypes(include=['number']).columns.tolist(),
            "categorical": df.select_dtypes(include=['object', 'category']).columns.tolist()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/eda/overview")
async def eda_overview():
    """Get data overview."""
    try:
        if 'current_dataset' not in _eda_session:
            raise HTTPException(status_code=400, detail="No dataset loaded")
        
        import pandas as pd
        df = pd.read_csv(_eda_session['current_dataset'])
        eda = EDAService(df)
        
        return {
            "overview": eda.get_data_overview(),
            "preview": eda.get_data_preview()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/eda/statistics")
async def eda_statistics():
    """Get summary statistics."""
    try:
        if 'current_dataset' not in _eda_session:
            raise HTTPException(status_code=400, detail="No dataset loaded")
        
        import pandas as pd
        df = pd.read_csv(_eda_session['current_dataset'])
        eda = EDAService(df)
        
        return {
            "summary_stats": eda.get_summary_statistics(),
            "categorical_summary": eda.get_categorical_summary(),
            "correlation_matrix": eda.get_correlation_matrix()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/eda/insights")
async def eda_insights():
    """Get all insights."""
    try:
        if 'current_dataset' not in _eda_session:
            raise HTTPException(status_code=400, detail="No dataset loaded")
        
        import pandas as pd
        df = pd.read_csv(_eda_session['current_dataset'])
        eda = EDAService(df)
        
        return {"insights": eda.get_all_insights()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/eda/recommendations")
async def eda_recommendations():
    """Get feature engineering recommendations."""
    try:
        if 'current_dataset' not in _eda_session:
            raise HTTPException(status_code=400, detail="No dataset loaded")
        
        import pandas as pd
        df = pd.read_csv(_eda_session['current_dataset'])
        eda = EDAService(df)
        
        return {"recommendations": eda.get_feature_engineering_recommendations()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/eda/plot-suggestions")
async def eda_plot_suggestions():
    """Get plot suggestions."""
    try:
        if 'current_dataset' not in _eda_session:
            raise HTTPException(status_code=400, detail="No dataset loaded")
        
        import pandas as pd
        df = pd.read_csv(_eda_session['current_dataset'])
        eda = EDAService(df)
        
        return {"suggestions": eda.suggest_plots()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/eda/plot")
async def eda_generate_plot(
    plot_type: str,
    x: str = None,
    y: str = None,
    group: str = None,
    features: list[str] = Query(default=None)
):
    """Generate a plot visualization."""
    try:
        if 'current_dataset' not in _eda_session:
            raise HTTPException(status_code=400, detail="No dataset loaded")
        
        # Validate input parameters
        if not plot_type or plot_type.strip() == '':
            raise HTTPException(status_code=400, detail="Plot type is required")
        
        # Remove whitespace from parameters
        plot_type = plot_type.strip()
        
        # Only validate x for non-heatmap plots
        if plot_type != "heatmap":
            if not x or x.strip() == '':
                raise HTTPException(status_code=400, detail="X-axis feature is required")
            x = x.strip()
            y = y.strip() if y else None
            group = group.strip() if group else None
        
        import pandas as pd
        import numpy as np
        import plotly.graph_objects as go
        import json
        
        df = pd.read_csv(_eda_session['current_dataset'])
        
        # Validate column names exist (for non-heatmap plots)
        if plot_type != "heatmap":
            if x not in df.columns:
                raise HTTPException(status_code=400, detail=f"Column '{x}' not found in dataset. Available columns: {list(df.columns)}")
            if y and y not in df.columns:
                raise HTTPException(status_code=400, detail=f"Column '{y}' not found in dataset. Available columns: {list(df.columns)}")
            if group and group not in df.columns:
                raise HTTPException(status_code=400, detail=f"Column '{group}' not found in dataset. Available columns: {list(df.columns)}")
        
        # Helper function to convert pandas objects to JSON-serializable Python types
        def to_json_serializable(val):
            """Convert various types to JSON-serializable format"""
            # Handle None first
            if val is None:
                return None
            
            # Handle numpy/pandas scalar types
            if isinstance(val, (np.integer, np.floating)):
                return float(val) if isinstance(val, np.floating) else int(val)
            
            # Handle numpy arrays (BEFORE list check)
            if isinstance(val, np.ndarray):
                return val.tolist()
            
            # Handle pandas Series (BEFORE list check)
            if isinstance(val, pd.Series):
                return val.tolist()
            
            # Handle lists and tuples
            if isinstance(val, (list, tuple)):
                return [to_json_serializable(v) for v in val]
            
            # Handle dicts
            if isinstance(val, dict):
                return {k: to_json_serializable(v) for k, v in val.items()}
            
            # Handle NaN for scalar values (only after we've ruled out arrays)
            try:
                if pd.isna(val):
                    return None
            except (TypeError, ValueError):
                # pd.isna() might fail on some types, just pass through
                pass
            
            return val
        
        # Generate plot based on type
        if plot_type == "histogram":
            x_data = df[x].dropna().tolist()
            fig = go.Figure(data=[
                go.Histogram(
                    x=x_data,
                    nbinsx=30,
                    name=x,
                    marker=dict(color='rgba(99, 110, 250, 0.7)')
                )
            ])
            fig.update_layout(
                title=f"Distribution of {x}",
                xaxis_title=x,
                yaxis_title="Count",
                showlegend=False
            )
        
        elif plot_type == "scatter":
            if not y:
                raise HTTPException(status_code=400, detail="Y-axis feature required for scatter plot")
            
            # Remove NaN values for scatter plot
            valid_idx = df[[x, y]].notna().all(axis=1)
            x_data = df.loc[valid_idx, x].tolist()
            y_data = df.loc[valid_idx, y].tolist()
            
            trace_data = []
            if group and group in df.columns:
                # Group by color
                group_df = df.loc[valid_idx]
                for group_val in group_df[group].unique():
                    if pd.isna(group_val):
                        continue
                    mask = group_df[group] == group_val
                    trace_data.append(
                        go.Scatter(
                            x=group_df.loc[mask, x].tolist(),
                            y=group_df.loc[mask, y].tolist(),
                            mode='markers',
                            name=str(group_val),
                            marker=dict(size=8, opacity=0.7)
                        )
                    )
            else:
                # Single trace
                trace_data.append(
                    go.Scatter(
                        x=x_data,
                        y=y_data,
                        mode='markers',
                        name='data',
                        marker=dict(
                            size=8,
                            color='rgba(99, 110, 250, 0.7)',
                            opacity=0.7
                        )
                    )
                )
            
            fig = go.Figure(data=trace_data)
            fig.update_layout(
                title=f"{x} vs {y}",
                xaxis_title=x,
                yaxis_title=y,
                hovermode='closest'
            )
        
        elif plot_type == "boxplot":
            trace_data = []
            
            if group and group in df.columns:
                # Box plot grouped by category
                for group_val in df[group].dropna().unique():
                    mask = df[group] == group_val
                    y_values = df.loc[mask, x].dropna().tolist()
                    if y_values:  # Only add if there's data
                        trace_data.append(
                            go.Box(
                                y=y_values,
                                name=str(group_val),
                                boxmean='sd'
                            )
                        )
                title_str = f"Box Plot of {x} by {group}"
            else:
                # Single box plot
                y_values = df[x].dropna().tolist()
                trace_data.append(
                    go.Box(
                        y=y_values,
                        name=x,
                        boxmean='sd',
                        marker=dict(color='rgba(99, 110, 250, 0.7)')
                    )
                )
                title_str = f"Box Plot of {x}"
            
            if trace_data:
                fig = go.Figure(data=trace_data)
                fig.update_layout(
                    title=title_str,
                    yaxis_title=x
                )
            else:
                raise HTTPException(status_code=400, detail="No valid data for box plot")
        
        elif plot_type == "bar":
            value_counts = df[x].value_counts().head(20)
            # Ensure index is converted to list of strings
            x_labels = [str(idx) for idx in value_counts.index]
            y_values = value_counts.values.tolist()
            
            fig = go.Figure(data=[
                go.Bar(
                    x=x_labels,
                    y=y_values,
                    marker=dict(
                        color='rgba(99, 110, 250, 0.7)',
                        line=dict(color='rgba(99, 110, 250, 1.0)', width=1)
                    )
                )
            ])
            fig.update_layout(
                title=f"Top Values in {x}",
                xaxis_title=x,
                yaxis_title="Count",
                showlegend=False
            )
        
        elif plot_type == "heatmap":
            # Use selected features or all numeric columns
            if features and len(features) > 0:
                # Validate selected features are numeric
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                selected_features = [f for f in features if f in numeric_cols]
                if len(selected_features) < 2:
                    raise HTTPException(status_code=400, detail=f"Need at least 2 numeric features. Available numeric columns: {numeric_cols}")
                cols_to_use = selected_features
            else:
                # Default to all numeric columns
                cols_to_use = df.select_dtypes(include=['number']).columns.tolist()
                if len(cols_to_use) < 2:
                    raise HTTPException(status_code=400, detail="Need at least 2 numeric columns for heatmap")
            
            corr = df[cols_to_use].corr()
            
            # Convert to lists for proper JSON serialization
            z_values = corr.values.tolist()
            x_labels = cols_to_use
            y_labels = cols_to_use
            
            fig = go.Figure(data=[
                go.Heatmap(
                    z=z_values,
                    x=x_labels,
                    y=y_labels,
                    colorscale='RdBu',
                    zmid=0,
                    zmin=-1,
                    zmax=1,
                    colorbar=dict(title="Correlation")
                )
            ])
            fig.update_layout(
                title=f"Correlation Heatmap ({len(cols_to_use)} features)",
                xaxis_title="Features",
                yaxis_title="Features"
            )
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown plot type: {plot_type}. Valid types: histogram, scatter, boxplot, bar, heatmap")
        
        # Update layout for better appearance
        fig.update_layout(
            height=600,
            hovermode='closest',
            margin=dict(l=50, r=50, t=80, b=50),
            template='plotly_white'
        )
        
        # Convert to JSON serializable format
        plot_json = fig.to_json()
        data = json.loads(plot_json)
        
        # Ensure all data is properly serialized
        plot_data = {
            "title": data.get("layout", {}).get("title", {}).get("text", "Plot") if isinstance(data.get("layout", {}).get("title"), dict) else data.get("layout", {}).get("title", "Plot"),
            "data": to_json_serializable(data.get("data", [])),
            "layout": to_json_serializable(data.get("layout", {})),
            "plot_type": plot_type
        }
        
        return plot_data
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in plot generation: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Plot generation failed: {str(e)}") from e


@app.post("/api/eda/upload-dataset")
async def eda_upload_dataset(file: UploadFile = File(...)):
    """Upload a dataset for EDA."""
    try:
        import pandas as pd
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Store temporarily
        from src.services.automl_service import ROOT_DIR
        temp_path = ROOT_DIR / "temp_eda_dataset.csv"
        df.to_csv(temp_path, index=False)
        
        _eda_session['current_dataset'] = str(temp_path)
        _eda_session['run_id'] = None
        
        return {
            "status": "success",
            "message": f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/health")
async def health_check():
    return {"status": "ok"}
