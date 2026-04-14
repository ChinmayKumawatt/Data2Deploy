import json
import math
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from src.utils.config import load_stage_config
from src.utils.common import save_object
from src.utils.exception import CustomException
from src.utils.logger import logger

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier = None
    XGBRegressor = None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    LGBMClassifier = None
    LGBMRegressor = None

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except ImportError:
    CatBoostClassifier = None
    CatBoostRegressor = None


class ModelTrainerArtifact:
    def __init__(self, best_model_path, top_models_paths, report_path, best_score):
        self.best_model_path = best_model_path
        self.top_models_paths = top_models_paths
        self.report_path = report_path
        self.best_score = best_score


class ModelTrainer:
    def __init__(self, config):
        self.config = config

    def _resolve_path(self, path_value):
        if not path_value:
            raise ValueError("Path is empty")
        return Path(path_value)

    def _ensure_directory(self, directory_path):
        Path(directory_path).mkdir(parents=True, exist_ok=True)

    def _load_dataset(self, file_path, dataset_name):
        path = self._resolve_path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"{dataset_name} file does not exist: {path}")

        if not path.is_file():
            raise ValueError(f"{dataset_name} path is not a file: {path}")

        logger.info("Loading %s dataset from %s", dataset_name, path)
        df = pd.read_csv(path)

        if df.empty:
            raise ValueError(f"{dataset_name} dataset is empty")

        logger.info("%s dataset loaded with shape %s", dataset_name, df.shape)
        return df

    def _load_data(self):
        train_df = self._load_dataset(self.config.transformed_train_path, "transformed train")
        test_df = self._load_dataset(self.config.transformed_test_path, "transformed test")

        target_column = getattr(self.config, "target_column", None)
        if not target_column:
            raise ValueError("Target column is not configured")

        if target_column not in train_df.columns:
            raise ValueError(f"Target column '{target_column}' missing in transformed train dataset")

        if target_column not in test_df.columns:
            raise ValueError(f"Target column '{target_column}' missing in transformed test dataset")

        X_train = train_df.drop(columns=[target_column])
        y_train = train_df[target_column]
        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column]

        if X_train.empty:
            raise ValueError("Training features are empty")

        if X_test.empty:
            raise ValueError("Test features are empty")

        return X_train, y_train, X_test, y_test

    def _infer_task_type(self, y_train):
        configured_task_type = getattr(self.config, "task_type", None)
        if configured_task_type:
            return configured_task_type.lower()

        return "classification" if y_train.nunique(dropna=True) <= 20 else "regression"

    def _safe_random_state(self):
        return getattr(self.config, "random_state", 42)

    def _build_classification_models(self):
        random_state = self._safe_random_state()
        models = {
            "logistic_regression": LogisticRegression(
                random_state=random_state,
                max_iter=1000,
            ),
            "random_forest_classifier": RandomForestClassifier(
                random_state=random_state,
            ),
            "gradient_boosting_classifier": GradientBoostingClassifier(
                random_state=random_state,
            ),
        }

        if XGBClassifier is not None:
            models["xgb_classifier"] = XGBClassifier(
                random_state=random_state,
                eval_metric="logloss",
                use_label_encoder=False,
            )
        else:
            logger.warning("xgboost is not available. Skipping XGBClassifier")

        if LGBMClassifier is not None:
            models["lgbm_classifier"] = LGBMClassifier(
                random_state=random_state,
            )
        else:
            logger.warning("lightgbm is not available. Skipping LGBMClassifier")

        if CatBoostClassifier is not None:
            models["catboost_classifier"] = CatBoostClassifier(
                random_state=random_state,
                verbose=0,
            )
        else:
            logger.warning("catboost is not available. Skipping CatBoostClassifier")

        return models

    def _build_regression_models(self):
        random_state = self._safe_random_state()
        models = {
            "linear_regression": LinearRegression(),
            "random_forest_regressor": RandomForestRegressor(
                random_state=random_state,
            ),
            "gradient_boosting_regressor": GradientBoostingRegressor(
                random_state=random_state,
            ),
        }

        if XGBRegressor is not None:
            models["xgb_regressor"] = XGBRegressor(
                random_state=random_state,
                objective="reg:squarederror",
            )
        else:
            logger.warning("xgboost is not available. Skipping XGBRegressor")

        if LGBMRegressor is not None:
            models["lgbm_regressor"] = LGBMRegressor(
                random_state=random_state,
            )
        else:
            logger.warning("lightgbm is not available. Skipping LGBMRegressor")

        if CatBoostRegressor is not None:
            models["catboost_regressor"] = CatBoostRegressor(
                random_state=random_state,
                verbose=0,
            )
        else:
            logger.warning("catboost is not available. Skipping CatBoostRegressor")

        return models

    def _get_models(self, task_type):
        logger.info("Building model registry for %s task", task_type)

        if task_type == "classification":
            models = self._build_classification_models()
        elif task_type == "regression":
            models = self._build_regression_models()
        else:
            raise ValueError("task_type must be either 'classification' or 'regression'")

        if not models:
            raise ValueError("No models are available for training")

        return models

    def _filter_models_by_mode(self, models):
        training_mode = getattr(self.config, "training_mode", "manual").lower()
        logger.info("Training mode: %s", training_mode)

        if training_mode == "manual":
            selected_model = getattr(self.config, "selected_model", None)
            if not selected_model:
                raise ValueError("selected_model must be provided when training_mode is 'manual'")

            if selected_model not in models:
                raise ValueError(
                    f"Selected model '{selected_model}' is invalid. Available models: {sorted(models.keys())}"
                )

            return {selected_model: models[selected_model]}

        if training_mode == "auto":
            model_list = getattr(self.config, "model_list", None)
            if not model_list:
                return models

            invalid_models = [model_name for model_name in model_list if model_name not in models]
            if invalid_models:
                raise ValueError(
                    f"Invalid model names in model_list: {invalid_models}. Available models: {sorted(models.keys())}"
                )

            return {model_name: models[model_name] for model_name in model_list}

        raise ValueError("training_mode must be either 'manual' or 'auto'")

    def _train_model(self, model_name, model, X_train, y_train):
        logger.info("Training model: %s", model_name)
        model.fit(X_train, y_train)
        logger.info("Completed training for model: %s", model_name)
        return model

    def _classification_metrics(self, y_true, y_pred):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }

    def _regression_metrics(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        return {
            "r2": r2_score(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": math.sqrt(mse),
        }

    def _evaluate_model(self, task_type, y_true, y_pred):
        if task_type == "classification":
            return self._classification_metrics(y_true, y_pred)

        return self._regression_metrics(y_true, y_pred)

    def _is_higher_score_better(self, metric_name):
        return metric_name.lower() in {"accuracy", "precision", "recall", "f1", "r2"}

    def _rank_models(self, model_results, evaluation_metric):
        metric_name = evaluation_metric.lower()

        for result in model_results:
            if metric_name not in result["metrics"]:
                raise ValueError(
                    f"Evaluation metric '{evaluation_metric}' is not available for model ranking"
                )
            result["score"] = float(result["metrics"][metric_name])

        reverse = self._is_higher_score_better(metric_name)
        ranked_results = sorted(model_results, key=lambda item: item["score"], reverse=reverse)
        logger.info("Model ranking completed using metric: %s", evaluation_metric)
        return ranked_results

    def _save_models(self, ranked_results):
        model_output_dir = self._resolve_path(getattr(self.config, "model_output_dir", None))
        self._ensure_directory(model_output_dir)

        top_results = ranked_results[:3]
        saved_model_paths = []

        for index, result in enumerate(top_results, start=1):
            model_path = model_output_dir / f"model_{index}.pkl"
            save_object(str(model_path), result["model"])
            saved_model_paths.append(str(model_path))
            result["saved_model_path"] = str(model_path)
            logger.info("Saved ranked model %s at %s", result["model_name"], model_path)

        return saved_model_paths

    def _save_report(self, ranked_results):
        report_path_value = getattr(self.config, "report_path", None)
        if report_path_value:
            report_path = self._resolve_path(report_path_value)
            self._ensure_directory(report_path.parent)
        else:
            model_output_dir = self._resolve_path(getattr(self.config, "model_output_dir", None))
            self._ensure_directory(model_output_dir)
            report_path = model_output_dir / "model_training_report.json"

        report_payload = {
            "evaluation_metric": getattr(self.config, "evaluation_metric", None),
            "best_model": ranked_results[0]["model_name"],
            "best_score": ranked_results[0]["score"],
            "ranking": [
                {
                    "rank": index,
                    "model_name": result["model_name"],
                    "score": result["score"],
                    "metrics": {metric: float(value) for metric, value in result["metrics"].items()},
                    "model_path": result.get("saved_model_path"),
                }
                for index, result in enumerate(ranked_results[:3], start=1)
            ],
            "all_model_scores": {
                result["model_name"]: {
                    metric: float(value) for metric, value in result["metrics"].items()
                }
                for result in ranked_results
            },
        }

        with report_path.open("w", encoding="utf-8") as report_file:
            json.dump(report_payload, report_file, indent=4)

        logger.info("Model training report saved at %s", report_path)
        return str(report_path)

    def initiate_model_training(self):
        try:
            logger.info("Model training initiated")

            X_train, y_train, X_test, y_test = self._load_data()
            task_type = self._infer_task_type(y_train)
            evaluation_metric = getattr(self.config, "evaluation_metric", None)

            if not evaluation_metric:
                raise ValueError("evaluation_metric must be configured")

            models = self._get_models(task_type=task_type)
            selected_models = self._filter_models_by_mode(models)

            model_results = []

            for model_name, model in selected_models.items():
                trained_model = self._train_model(
                    model_name=model_name,
                    model=model,
                    X_train=X_train,
                    y_train=y_train,
                )
                predictions = trained_model.predict(X_test)
                metrics = self._evaluate_model(
                    task_type=task_type,
                    y_true=y_test,
                    y_pred=predictions,
                )

                logger.info("Evaluation metrics for %s: %s", model_name, metrics)
                model_results.append(
                    {
                        "model_name": model_name,
                        "model": trained_model,
                        "metrics": metrics,
                    }
                )

            if not model_results:
                raise ValueError("No models were trained")

            ranked_results = self._rank_models(
                model_results=model_results,
                evaluation_metric=evaluation_metric,
            )
            top_model_paths = self._save_models(ranked_results)
            report_path = self._save_report(ranked_results)

            best_result = ranked_results[0]
            best_model_path = top_model_paths[0]
            best_score = float(best_result["score"])

            logger.info(
                "Best model: %s with %s score of %.6f",
                best_result["model_name"],
                evaluation_metric,
                best_score,
            )

            return ModelTrainerArtifact(
                best_model_path=best_model_path,
                top_models_paths=top_model_paths,
                report_path=report_path,
                best_score=best_score,
            )

        except Exception as e:
            logger.exception("Model training failed")
            raise CustomException(e, sys)


def main():
    config = load_stage_config("training")
    trainer = ModelTrainer(config=config)
    trainer.initiate_model_training()


if __name__ == "__main__":
    main()
