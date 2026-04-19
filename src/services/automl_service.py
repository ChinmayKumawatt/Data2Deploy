import io
import json
import uuid
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import yaml

from src.services.mlflow_service import MLflowService
from src.training.components.data_ingestion import DataIngestion
from src.training.components.data_transformation import DataTransformation
from src.training.components.data_validation import DataValidation
from src.training.components.model_evaluation import ModelEvaluator
from src.training.components.model_trainer import ModelTrainer
from src.training.components.model_tuner import ModelTuner
from src.utils.common import load_object


ROOT_DIR = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
MLFLOW_DIR = ROOT_DIR / "mlruns"
MLFLOW_SERVICE = MLflowService(tracking_dir=MLFLOW_DIR)


def _read_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _sanitize_filename(filename: str) -> str:
    candidate = Path(filename or "dataset.csv").name
    if not candidate.lower().endswith(".csv"):
        candidate = f"{candidate}.csv"
    return candidate


def _infer_task_type(df: pd.DataFrame, target_column: str) -> str:
    return "classification" if df[target_column].nunique(dropna=True) <= 20 else "regression"


def _friendly_metric_name(task_type: str) -> str:
    return "R2" if task_type == "regression" else "F1"


def _load_json(path_value: Path) -> dict:
    with Path(path_value).open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _build_run_paths(run_id: str) -> dict:
    run_dir = RUNS_DIR / run_id
    paths = {
        "run_id": run_id,
        "run_dir": run_dir,
        "input_dir": run_dir / "input",
        "config_dir": run_dir / "config",
        "artifacts_dir": run_dir / "artifacts",
        "models_dir": run_dir / "models",
        "results_dir": run_dir / "results",
    }
    for path in paths.values():
        if isinstance(path, Path):
            path.mkdir(parents=True, exist_ok=True)
    return paths


def _default_training_options() -> dict:
    return {
        "test_size": 0.2,
        "random_state": 42,
        "min_rows": 30,
        "feature_selection_method": "mutual_info",
        "numerical_imputation_strategy": "mean",
        "categorical_imputation_strategy": "most_frequent",
        "scaling_method": "standard",
        "training_mode": "auto",
        "model_list": [],
        "top_k": 3,
        "n_iter": 5,
        "cv": 3,
        "n_jobs": 1,
        "minimum_performance_threshold": None,
        "overfitting_threshold": 0.1,
        "missing_value_threshold": 0.35,
        "drift_threshold": 0.6,
        "imbalance_threshold": 0.9,
    }


def preview_dataset(file_bytes: bytes) -> dict:
    df = _read_csv_bytes(file_bytes)
    if df.empty:
        raise ValueError("Uploaded CSV is empty.")

    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [column for column in df.columns if column not in set(numeric_columns)]

    return {
        "rows": int(df.shape[0]),
        "column_names": [str(column) for column in df.columns],
        "numeric_columns": [str(column) for column in numeric_columns],
        "categorical_columns": [str(column) for column in categorical_columns],
        "sample_rows": df.head(8).fillna("").to_dict(orient="records"),
    }


def _resolve_training_options(task_type: str, overrides: dict | None) -> dict:
    options = _default_training_options()
    options.update({key: value for key, value in (overrides or {}).items() if value not in (None, "")})

    options["test_size"] = float(options["test_size"])
    options["random_state"] = int(options["random_state"])
    options["min_rows"] = int(options["min_rows"])
    options["top_k"] = max(1, int(options["top_k"]))
    options["n_iter"] = max(1, int(options["n_iter"]))
    options["cv"] = max(2, int(options["cv"]))
    options["n_jobs"] = int(options["n_jobs"])
    options["overfitting_threshold"] = float(options["overfitting_threshold"])
    options["missing_value_threshold"] = float(options["missing_value_threshold"])
    options["drift_threshold"] = float(options["drift_threshold"])
    options["imbalance_threshold"] = float(options["imbalance_threshold"])
    options["training_mode"] = str(options["training_mode"]).lower()
    options["model_list"] = [item for item in options.get("model_list", []) if item]

    if options["minimum_performance_threshold"] in (None, ""):
        options["minimum_performance_threshold"] = 0.1 if task_type == "regression" else 0.3
    else:
        options["minimum_performance_threshold"] = float(options["minimum_performance_threshold"])

    return options


def _build_stage_configs(
    run_paths: dict,
    dataset_path: Path,
    target_column: str,
    task_type: str,
    feature_mode: str,
    n_features: int,
    selected_features: list[str],
    df: pd.DataFrame,
    training_options: dict,
) -> dict:
    raw_path = run_paths["artifacts_dir"] / "data_ingestion" / "raw.csv"
    train_path = run_paths["artifacts_dir"] / "data_ingestion" / "train.csv"
    test_path = run_paths["artifacts_dir"] / "data_ingestion" / "test.csv"
    transformed_dir = run_paths["artifacts_dir"] / "data_transformation"
    transformed_train_path = transformed_dir / "transformed_train.csv"
    transformed_test_path = transformed_dir / "transformed_test.csv"
    preprocessor_object_path = transformed_dir / "preprocessor.pkl"
    target_encoder_path = transformed_dir / "target_encoder.json"
    training_report_path = run_paths["artifacts_dir"] / "model_trainer" / "model_training_report.json"
    tuning_report_path = run_paths["artifacts_dir"] / "model_tuner" / "model_tuning_report.json"
    evaluation_report_path = run_paths["artifacts_dir"] / "model_evaluator" / "model_evaluation_report.json"
    schema = {str(column): str(dtype) for column, dtype in df.dtypes.items()}

    return {
        "ingestion": {
            "dataset_path": str(dataset_path),
            "target_column": target_column,
            "test_size": training_options["test_size"],
            "random_state": training_options["random_state"],
            "min_rows": training_options["min_rows"],
            "task_type": task_type,
            "stratify": True,
            "raw_path": str(raw_path),
            "train_path": str(train_path),
            "test_path": str(test_path),
        },
        "validation": {
            "train_path": str(train_path),
            "test_path": str(test_path),
            "target_column": target_column,
            "schema": {"columns": schema},
            "missing_value_threshold": training_options["missing_value_threshold"],
            "drift_threshold": training_options["drift_threshold"],
            "imbalance_threshold": training_options["imbalance_threshold"],
            "report_path": str(run_paths["artifacts_dir"] / "data_validation" / "validation_report.json"),
        },
        "transformation": {
            "train_path": str(train_path),
            "test_path": str(test_path),
            "target_column": target_column,
            "feature_selection_mode": feature_mode,
            "selected_features": list(selected_features),
            "n_features": int(n_features),
            "feature_selection_method": training_options["feature_selection_method"],
            "task_type": task_type,
            "numerical_columns": None,
            "categorical_columns": None,
            "numerical_imputation_strategy": training_options["numerical_imputation_strategy"],
            "categorical_imputation_strategy": training_options["categorical_imputation_strategy"],
            "scaling_method": training_options["scaling_method"],
            "preprocessor_object_path": str(preprocessor_object_path),
            "target_encoder_path": str(target_encoder_path),
            "transformed_train_path": str(transformed_train_path),
            "transformed_test_path": str(transformed_test_path),
        },
        "training": {
            "transformed_train_path": str(transformed_train_path),
            "transformed_test_path": str(transformed_test_path),
            "target_column": target_column,
            "training_mode": training_options["training_mode"],
            "selected_model": "random_forest_regressor" if task_type == "regression" else "random_forest_classifier",
            "model_list": list(training_options["model_list"]),
            "evaluation_metric": "r2" if task_type == "regression" else "f1",
            "model_output_dir": str(run_paths["models_dir"] / "training"),
            "report_path": str(training_report_path),
            "random_state": training_options["random_state"],
            "task_type": task_type,
        },
        "tuning": {
            "transformed_train_path": str(transformed_train_path),
            "transformed_test_path": str(transformed_test_path),
            "target_column": target_column,
            "models_to_tune": "top_k",
            "top_k": training_options["top_k"],
            "model_trainer_report_path": str(training_report_path),
            "n_iter": training_options["n_iter"],
            "cv": training_options["cv"],
            "n_jobs": training_options["n_jobs"],
            "scoring_metric": "r2" if task_type == "regression" else "f1_weighted",
            "tuner_output_dir": str(run_paths["models_dir"] / "tuning"),
            "report_path": str(tuning_report_path),
            "random_state": training_options["random_state"],
            "task_type": task_type,
        },
        "evaluation": {
            "transformed_test_path": str(transformed_test_path),
            "target_column": target_column,
            "evaluation_metric": "r2" if task_type == "regression" else "f1",
            "minimum_performance_threshold": training_options["minimum_performance_threshold"],
            "evaluator_output_path": str(run_paths["models_dir"] / "final" / "final_model.pkl"),
            "report_path": str(evaluation_report_path),
            "task_type": task_type,
            "baseline_model_paths": [],
            "baseline_model_dir": str(run_paths["models_dir"] / "training"),
            "tuned_model_paths": [],
            "tuned_model_dir": str(run_paths["models_dir"] / "tuning"),
            "trainer_report_path": str(training_report_path),
            "tuner_report_path": str(tuning_report_path),
            "overfitting_threshold": training_options["overfitting_threshold"],
        },
    }


def _persist_run_config(run_paths: dict, params: dict, metadata: dict) -> None:
    with (run_paths["config_dir"] / "params.yaml").open("w", encoding="utf-8") as params_file:
        yaml.safe_dump(params, params_file, sort_keys=False)

    with (run_paths["config_dir"] / "run_metadata.json").open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=4)


def _load_target_decoder(target_encoder_path: Path):
    if not target_encoder_path.exists():
        return None

    classes = _load_json(target_encoder_path).get("classes", [])
    if not classes:
        return None

    def decode(values):
        decoded = []
        for value in values:
            try:
                index = int(value)
            except (TypeError, ValueError):
                decoded.append(value)
                continue
            decoded.append(classes[index] if 0 <= index < len(classes) else value)
        return decoded

    return decode


def _build_model_alias_map(training_report: dict, tuning_report: dict) -> dict:
    alias_map = {}
    for item in training_report.get("ranking", []):
        if item.get("model_path"):
            alias_map[str(Path(item["model_path"]))] = item.get("model_name", Path(item["model_path"]).stem)
    for item in tuning_report.get("models", []):
        if item.get("saved_model_path"):
            alias_map[str(Path(item["saved_model_path"]))] = item.get("model_name", Path(item["saved_model_path"]).stem)
    return alias_map


def _build_prediction_artifacts(run_paths: dict, params: dict, training_report: dict, tuning_report: dict, evaluation_report: dict) -> dict:
    transformed_test = pd.read_csv(params["transformation"]["transformed_test_path"])
    raw_test = pd.read_csv(params["ingestion"]["test_path"])
    target_column = params["evaluation"]["target_column"]
    task_type = params["evaluation"]["task_type"]
    X_test = transformed_test.drop(columns=[target_column])
    actual_values = raw_test[target_column].tolist()
    decoder = _load_target_decoder(Path(params["transformation"]["target_encoder_path"]))
    model_aliases = _build_model_alias_map(training_report, tuning_report)

    prediction_frame = pd.DataFrame({"row_id": list(range(1, len(raw_test) + 1)), "actual": actual_values})
    top_models = []

    for entry in evaluation_report.get("models", [])[:3]:
        model_path = Path(entry["model_path"])
        model = load_object(str(model_path))
        predictions = list(model.predict(X_test))
        if decoder is not None:
            predictions = decoder(predictions)

        display_name = model_aliases.get(str(model_path), entry["model_name"])
        prediction_column = f"{display_name}_prediction"
        prediction_frame[prediction_column] = [round(value, 4) if isinstance(value, float) else value for value in predictions]
        top_models.append(
            {
                "display_name": display_name,
                "model_type": entry["model_type"],
                "score": _safe_float(entry["score"]),
                "metrics": {key: _safe_float(value) for key, value in entry["metrics"].items()},
                "model_path": str(model_path),
            }
        )

    prediction_csv_path = run_paths["results_dir"] / "prediction_comparison.csv"
    prediction_json_path = run_paths["results_dir"] / "prediction_comparison.json"
    prediction_frame.to_csv(prediction_csv_path, index=False)

    payload = {
        "task_type": task_type,
        "metric_label": _friendly_metric_name(task_type),
        "top_models": top_models,
        "preview_rows": prediction_frame.head(25).to_dict(orient="records"),
        "row_count": int(len(prediction_frame)),
        "prediction_csv_path": str(prediction_csv_path),
        "score_plot": [
            {
                "label": model["display_name"],
                "score": model["score"],
                "width_pct": max(4.0, min(100.0, float(model["score"]) * 100 if float(model["score"]) > 0 else 4.0)),
            }
            for model in top_models
        ],
    }

    with prediction_json_path.open("w", encoding="utf-8") as comparison_file:
        json.dump(payload, comparison_file, indent=4)

    return payload


def _flatten_metrics(summary: dict) -> dict:
    metrics = {}
    for item in summary.get("comparison", {}).get("top_models", []):
        label = item["display_name"]
        metrics[f"{label}_score"] = _safe_float(item["score"])
    metrics["best_score"] = _safe_float(summary.get("best_score"))
    return metrics


def _collect_artifacts(params: dict, run_paths: dict) -> list[str]:
    artifacts = [
        params["validation"]["report_path"],
        params["training"]["report_path"],
        params["tuning"]["report_path"],
        params["evaluation"]["report_path"],
        str(run_paths["config_dir"] / "params.yaml"),
        str(run_paths["results_dir"] / "prediction_comparison.csv"),
        str(run_paths["results_dir"] / "summary.json"),
    ]
    return artifacts


def _log_to_mlflow(summary: dict, params: dict, run_paths: dict) -> dict:
    mlflow_params = {
        "run_id": summary["run_id"],
        "target_column": summary["target_column"],
        "task_type": summary["task_type"],
        "feature_mode": summary["feature_mode"],
        "n_features": summary["n_features"],
        "selected_features": summary["selected_features"],
        "test_size": params["ingestion"]["test_size"],
        "random_state": params["ingestion"]["random_state"],
        "training_mode": params["training"]["training_mode"],
        "top_k": params["tuning"]["top_k"],
        "n_iter": params["tuning"]["n_iter"],
        "cv": params["tuning"]["cv"],
        "minimum_performance_threshold": params["evaluation"]["minimum_performance_threshold"],
    }
    mlflow_tags = {
        "run_id": summary["run_id"],
        "task_type": summary["task_type"],
        "target_column": summary["target_column"],
    }
    return MLFLOW_SERVICE.log_run(
        run_name=f"run-{summary['run_id']}",
        tags=mlflow_tags,
        params=mlflow_params,
        metrics=_flatten_metrics(summary),
        artifacts=_collect_artifacts(params, run_paths),
    )


def _build_run_summary(
    run_id: str,
    params: dict,
    preview: dict,
    training_report: dict,
    tuning_report: dict,
    evaluation_report: dict,
    comparison_payload: dict,
    metadata: dict,
    mlflow_info: dict,
) -> dict:
    training_options = metadata["training_options"]
    return {
        "run_id": run_id,
        "task_type": params["ingestion"]["task_type"],
        "target_column": params["ingestion"]["target_column"],
        "feature_mode": params["transformation"]["feature_selection_mode"],
        "selected_features": params["transformation"]["selected_features"],
        "n_features": params["transformation"]["n_features"],
        "dataset_rows": preview["rows"],
        "dataset_columns": len(preview["column_names"]),
        "metric_label": _friendly_metric_name(params["ingestion"]["task_type"]),
        "best_model": evaluation_report.get("best_model"),
        "best_score": _safe_float(evaluation_report.get("best_score")),
        "threshold_passed": evaluation_report.get("threshold_check", {}).get("passed", False),
        "training_ranking": training_report.get("ranking", []),
        "tuning_models": tuning_report.get("models", []),
        "evaluation_models": evaluation_report.get("models", []),
        "comparison": comparison_payload,
        "dataset_name": metadata["original_filename"],
        "dataset_path": str(metadata["dataset_path"]),
        "created_at": metadata["created_at"],
        "training_options": training_options,
        "parameter_cards": [
            {"label": "Test Size", "value": training_options["test_size"]},
            {"label": "Feature Mode", "value": metadata["feature_mode"]},
            {"label": "Top K Tuned Models", "value": training_options["top_k"]},
            {"label": "Random Search Iterations", "value": training_options["n_iter"]},
            {"label": "CV Folds", "value": training_options["cv"]},
            {"label": "Random State", "value": training_options["random_state"]},
        ],
        "mlflow": mlflow_info,
    }


def _normalize_summary(summary: dict) -> dict:
    summary.setdefault("dataset_name", "dataset.csv")
    summary.setdefault("dataset_path", "")
    summary.setdefault("created_at", "")
    summary.setdefault("training_options", _default_training_options())
    summary.setdefault(
        "mlflow",
        {
            "enabled": False,
            "tracking_uri": MLFLOW_DIR.as_uri(),
            "experiment_name": MLFLOW_SERVICE.experiment_name,
            "run_id": None,
        },
    )
    summary.setdefault(
        "parameter_cards",
        [
            {"label": "Test Size", "value": summary["training_options"].get("test_size")},
            {"label": "Feature Mode", "value": summary.get("feature_mode")},
            {"label": "Top K Tuned Models", "value": summary["training_options"].get("top_k")},
            {"label": "Random Search Iterations", "value": summary["training_options"].get("n_iter")},
            {"label": "CV Folds", "value": summary["training_options"].get("cv")},
            {"label": "Random State", "value": summary["training_options"].get("random_state")},
        ],
    )
    comparison = summary.setdefault("comparison", {})
    comparison.setdefault("top_models", [])
    comparison.setdefault("preview_rows", [])
    comparison.setdefault("row_count", 0)
    comparison.setdefault("score_plot", [])
    if not comparison["score_plot"]:
        comparison["score_plot"] = [
            {
                "label": model["display_name"],
                "score": model["score"],
                "width_pct": max(4.0, min(100.0, float(model["score"]) * 100 if float(model["score"]) > 0 else 4.0)),
            }
            for model in comparison.get("top_models", [])
        ]
    return summary


def list_run_summaries() -> list[dict]:
    summaries = []
    for summary_path in RUNS_DIR.glob("*/results/summary.json"):
        try:
            summary = _normalize_summary(_load_json(summary_path))
            summary["compare_score_width"] = max(
                4.0,
                min(100.0, float(summary.get("best_score", 0.0)) * 100 if float(summary.get("best_score", 0.0)) > 0 else 4.0),
            )
            summaries.append(summary)
        except Exception:
            continue

    summaries.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return summaries


def load_run_summary(run_id: str) -> dict:
    summary_path = RUNS_DIR / run_id / "results" / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Run '{run_id}' was not found.")
    return _normalize_summary(_load_json(summary_path))


def load_run_metadata(run_id: str) -> dict:
    metadata_path = RUNS_DIR / run_id / "config" / "run_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Run '{run_id}' metadata was not found.")
    return _load_json(metadata_path)


def get_experiment_compare_payload(run_ids: list[str]) -> dict:
    runs = [load_run_summary(run_id) for run_id in run_ids]
    if not runs:
        raise ValueError("Select at least one experiment to compare.")

    metric_label = runs[0]["metric_label"]
    parameter_rows = []
    parameter_keys = [
        "test_size",
        "feature_mode",
        "n_features",
        "top_k",
        "n_iter",
        "cv",
        "random_state",
    ]
    for key in parameter_keys:
        values = []
        for run in runs:
            if key == "feature_mode":
                values.append(run["feature_mode"])
            elif key == "n_features":
                values.append(run["n_features"])
            else:
                values.append(run["training_options"].get(key))
        parameter_rows.append({"label": key.replace("_", " ").title(), "values": values})

    chart_rows = []
    for run in runs:
        chart_rows.append(
            {
                "run_id": run["run_id"],
                "dataset_name": run["dataset_name"],
                "best_score": run["best_score"],
                "width_pct": max(4.0, min(100.0, float(run["best_score"]) * 100 if float(run["best_score"]) > 0 else 4.0)),
                "top_models": run["comparison"]["top_models"],
            }
        )

    return {
        "runs": runs,
        "metric_label": metric_label,
        "parameter_rows": parameter_rows,
        "chart_rows": chart_rows,
    }


def rerun_existing_experiment(
    source_run_id: str,
    target_column: str | None = None,
    feature_mode: str | None = None,
    n_features: int | None = None,
    selected_features: list[str] | None = None,
    training_options: dict | None = None,
) -> dict:
    metadata = load_run_metadata(source_run_id)
    dataset_path = Path(metadata["dataset_path"])
    if not dataset_path.exists():
        raise FileNotFoundError("Original dataset file for this experiment is missing.")

    original_options = metadata.get("training_options", _default_training_options())
    merged_options = dict(original_options)
    merged_options.update({key: value for key, value in (training_options or {}).items() if value not in (None, "")})

    return run_training_pipeline(
        file_bytes=dataset_path.read_bytes(),
        filename=metadata["original_filename"],
        target_column=target_column or metadata["target_column"],
        feature_mode=feature_mode or metadata["feature_mode"],
        n_features=n_features if n_features is not None else int(metadata["n_features"]),
        selected_features=selected_features if selected_features is not None else list(metadata.get("selected_features", [])),
        training_options=merged_options,
    )


def run_training_pipeline(
    file_bytes: bytes,
    filename: str,
    target_column: str,
    feature_mode: str = "auto",
    n_features: int = 5,
    selected_features: list[str] | None = None,
    training_options: dict | None = None,
) -> dict:
    selected_features = [feature for feature in (selected_features or []) if feature]
    feature_mode = (feature_mode or "auto").lower()
    if feature_mode not in {"auto", "manual"}:
        raise ValueError("feature_mode must be either 'auto' or 'manual'.")

    preview = preview_dataset(file_bytes)
    if target_column not in preview["column_names"]:
        raise ValueError(f"Target column '{target_column}' was not found in the uploaded dataset.")

    if feature_mode == "manual" and not selected_features:
        raise ValueError("Select at least one feature when using manual feature mode.")
    if target_column in selected_features:
        raise ValueError("The target column cannot also be selected as a feature.")
    if feature_mode == "auto" and int(n_features) <= 0:
        raise ValueError("Number of features must be greater than 0 in auto mode.")

    run_id = uuid.uuid4().hex[:8]
    run_paths = _build_run_paths(run_id)
    dataset_path = run_paths["input_dir"] / _sanitize_filename(filename)
    dataset_path.write_bytes(file_bytes)

    df = pd.read_csv(dataset_path)
    task_type = _infer_task_type(df, target_column)
    resolved_options = _resolve_training_options(task_type=task_type, overrides=training_options)

    if feature_mode == "manual":
        missing_features = [feature for feature in selected_features if feature not in df.columns]
        if missing_features:
            raise ValueError(f"Selected features were not found in the dataset: {missing_features}")
    else:
        n_features = min(int(n_features), max(len(df.columns) - 1, 1))

    params = _build_stage_configs(
        run_paths=run_paths,
        dataset_path=dataset_path,
        target_column=target_column,
        task_type=task_type,
        feature_mode=feature_mode,
        n_features=int(n_features),
        selected_features=selected_features,
        df=df,
        training_options=resolved_options,
    )
    metadata = {
        "run_id": run_id,
        "original_filename": _sanitize_filename(filename),
        "dataset_path": str(dataset_path),
        "target_column": target_column,
        "feature_mode": feature_mode,
        "selected_features": selected_features,
        "n_features": int(n_features),
        "task_type": task_type,
        "training_options": resolved_options,
        "created_at": pd.Timestamp.utcnow().isoformat(),
    }
    _persist_run_config(run_paths, params, metadata)

    try:
        DataIngestion(SimpleNamespace(**params["ingestion"])).initiate_data_ingestion()
        validation_artifact = DataValidation(SimpleNamespace(**params["validation"])).initiate_data_validation()
        if not validation_artifact.validation_status:
            raise ValueError("Validation checks failed for the uploaded dataset.")

        DataTransformation(SimpleNamespace(**params["transformation"])).initiate_data_transformation()
        ModelTrainer(SimpleNamespace(**params["training"])).initiate_model_training()
        ModelTuner(SimpleNamespace(**params["tuning"])).initiate_model_tuning()
        ModelEvaluator(SimpleNamespace(**params["evaluation"])).initiate_model_evaluation()

        training_report = _load_json(Path(params["training"]["report_path"]))
        tuning_report = _load_json(Path(params["tuning"]["report_path"]))
        evaluation_report = _load_json(Path(params["evaluation"]["report_path"]))
        comparison_payload = _build_prediction_artifacts(run_paths, params, training_report, tuning_report, evaluation_report)

        summary = _build_run_summary(
            run_id=run_id,
            params=params,
            preview=preview,
            training_report=training_report,
            tuning_report=tuning_report,
            evaluation_report=evaluation_report,
            comparison_payload=comparison_payload,
            metadata=metadata,
            mlflow_info={
                "enabled": False,
                "tracking_uri": MLFLOW_DIR.as_uri(),
                "experiment_name": MLFLOW_SERVICE.experiment_name,
                "run_id": None,
            },
        )

        with (run_paths["results_dir"] / "summary.json").open("w", encoding="utf-8") as summary_file:
            json.dump(summary, summary_file, indent=4)

        mlflow_info = _log_to_mlflow(summary, params, run_paths)
        summary["mlflow"] = mlflow_info

        with (run_paths["results_dir"] / "summary.json").open("w", encoding="utf-8") as summary_file:
            json.dump(summary, summary_file, indent=4)

        return summary
    except Exception:
        (run_paths["results_dir"] / "failed.txt").write_text("Run failed. Check logs for details.", encoding="utf-8")
        raise
