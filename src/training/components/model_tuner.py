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
from sklearn.model_selection import RandomizedSearchCV

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


class ModelTunerArtifact:
    def __init__(self, best_tuned_model_path, tuned_model_paths, report_path, best_score):
        self.best_tuned_model_path = best_tuned_model_path
        self.tuned_model_paths = tuned_model_paths
        self.report_path = report_path
        self.best_score = best_score


class ModelTuner:
    def __init__(self, config):
        self.config = config

    def _resolve_path(self, path_value):
        if not path_value:
            raise ValueError("Path is empty")
        return Path(path_value)

    def _ensure_directory(self, directory_path):
        Path(directory_path).mkdir(parents=True, exist_ok=True)

    def _safe_random_state(self):
        return getattr(self.config, "random_state", 42)

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

    def _build_classification_models(self):
        random_state = self._safe_random_state()
        models = {
            "logistic_regression": LogisticRegression(
                random_state=random_state,
                max_iter=1000,
                solver="liblinear",
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
        logger.info("Building tuner model registry for %s task", task_type)

        if task_type == "classification":
            models = self._build_classification_models()
        elif task_type == "regression":
            models = self._build_regression_models()
        else:
            raise ValueError("task_type must be either 'classification' or 'regression'")

        if not models:
            raise ValueError("No models are available for tuning")

        return models

    def _load_trainer_report(self):
        report_path_value = getattr(self.config, "model_trainer_report_path", None)
        if not report_path_value:
            raise ValueError(
                "model_trainer_report_path must be provided to tune the top models from the trainer"
            )

        report_path = self._resolve_path(report_path_value)
        if not report_path.exists():
            raise FileNotFoundError(f"Model trainer report does not exist: {report_path}")

        with report_path.open("r", encoding="utf-8") as report_file:
            report = json.load(report_file)

        return report

    def _get_models_to_tune(self, available_models):
        models_to_tune = getattr(self.config, "models_to_tune", "top_k")

        if models_to_tune == "top_k":
            trainer_report = self._load_trainer_report()
            top_k = int(getattr(self.config, "top_k", 3))
            ranked_models = trainer_report.get("ranking", [])
            selected_model_names = [item["model_name"] for item in ranked_models[:top_k]]
            baseline_scores = trainer_report.get("all_model_scores", {})
        elif isinstance(models_to_tune, list):
            selected_model_names = list(models_to_tune)
            baseline_scores = {}
        else:
            raise ValueError("models_to_tune must be a list of model names or 'top_k'")

        if not selected_model_names:
            raise ValueError("No models were selected for tuning")

        invalid_models = [model_name for model_name in selected_model_names if model_name not in available_models]
        if invalid_models:
            raise ValueError(
                f"Selected models are not available for tuning: {invalid_models}. "
                f"Available models: {sorted(available_models.keys())}"
            )

        logger.info("Models selected for tuning: %s", selected_model_names)
        return {name: available_models[name] for name in selected_model_names}, baseline_scores

    def _get_search_space(self, model_name):
        random_state = self._safe_random_state()

        search_spaces = {
            "logistic_regression": {
                "C": [0.01, 0.1, 1.0, 10.0, 100.0],
                "penalty": ["l1", "l2"],
                "solver": ["liblinear"],
                "max_iter": [500, 1000, 1500],
            },
            "random_forest_classifier": {
                "n_estimators": [100, 200, 300, 500],
                "max_depth": [None, 5, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
            "gradient_boosting_classifier": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_depth": [3, 5, 7],
                "subsample": [0.8, 1.0],
            },
            "xgb_classifier": {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_depth": [3, 5, 7, 10],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            },
            "lgbm_classifier": {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "num_leaves": [31, 63, 127],
                "max_depth": [-1, 5, 10, 20],
            },
            "catboost_classifier": {
                "iterations": [100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "depth": [4, 6, 8, 10],
                "l2_leaf_reg": [1, 3, 5, 7, 9],
            },
            "linear_regression": {},
            "random_forest_regressor": {
                "n_estimators": [100, 200, 300, 500],
                "max_depth": [None, 5, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
            "gradient_boosting_regressor": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_depth": [3, 5, 7],
                "subsample": [0.8, 1.0],
            },
            "xgb_regressor": {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_depth": [3, 5, 7, 10],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            },
            "lgbm_regressor": {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "num_leaves": [31, 63, 127],
                "max_depth": [-1, 5, 10, 20],
            },
            "catboost_regressor": {
                "iterations": [100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "depth": [4, 6, 8, 10],
                "l2_leaf_reg": [1, 3, 5, 7, 9],
            },
        }

        if model_name not in search_spaces:
            raise ValueError(f"No search space configured for model '{model_name}'")

        return search_spaces[model_name]

    def _tune_model(self, model_name, model, X_train, y_train):
        logger.info("Starting hyperparameter tuning for model: %s", model_name)

        search_space = self._get_search_space(model_name)
        if not search_space:
            logger.info("No tunable parameters configured for %s. Fitting baseline model", model_name)
            model.fit(X_train, y_train)
            return model, {}, None

        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=search_space,
            n_iter=int(getattr(self.config, "n_iter", 10)),
            cv=int(getattr(self.config, "cv", 3)),
            scoring=getattr(self.config, "scoring_metric", None),
            random_state=self._safe_random_state(),
            n_jobs=-1,
            verbose=0,
        )

        random_search.fit(X_train, y_train)
        logger.info(
            "Completed tuning for %s. Best CV score: %.6f",
            model_name,
            float(random_search.best_score_),
        )
        return random_search.best_estimator_, random_search.best_params_, float(random_search.best_score_)

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

    def _rank_tuned_models(self, tuned_results, ranking_metric):
        metric_name = ranking_metric.lower()

        for result in tuned_results:
            if metric_name not in result["test_metrics"]:
                raise ValueError(
                    f"Ranking metric '{ranking_metric}' is not available in tuned test metrics"
                )
            result["ranking_score"] = float(result["test_metrics"][metric_name])

        ranked_results = sorted(
            tuned_results,
            key=lambda item: item["ranking_score"],
            reverse=self._is_higher_score_better(metric_name),
        )
        logger.info("Ranked tuned models using test metric: %s", ranking_metric)
        return ranked_results

    def _save_tuned_models(self, tuned_results):
        tuner_output_dir = self._resolve_path(getattr(self.config, "tuner_output_dir", None))
        self._ensure_directory(tuner_output_dir)

        saved_paths = []
        for index, result in enumerate(tuned_results, start=1):
            model_path = tuner_output_dir / f"tuned_model_{index}.pkl"
            save_object(str(model_path), result["model"])
            saved_paths.append(str(model_path))
            result["saved_model_path"] = str(model_path)
            logger.info("Saved tuned model %s at %s", result["model_name"], model_path)

        return saved_paths

    def _save_report(self, tuned_results):
        tuner_output_dir = self._resolve_path(getattr(self.config, "tuner_output_dir", None))
        self._ensure_directory(tuner_output_dir)
        report_path = tuner_output_dir / "model_tuning_report.json"

        report_payload = {
            "scoring_metric": getattr(self.config, "scoring_metric", None),
            "best_model": tuned_results[0]["model_name"],
            "best_score": tuned_results[0]["ranking_score"],
            "models": [
                {
                    "rank": index,
                    "model_name": result["model_name"],
                    "best_params": result["best_params"],
                    "cv_best_score": result["cv_best_score"],
                    "ranking_score": result["ranking_score"],
                    "test_metrics": {
                        metric: float(value) for metric, value in result["test_metrics"].items()
                    },
                    "baseline_test_score": result.get("baseline_test_score"),
                    "saved_model_path": result.get("saved_model_path"),
                }
                for index, result in enumerate(tuned_results, start=1)
            ],
        }

        with report_path.open("w", encoding="utf-8") as report_file:
            json.dump(report_payload, report_file, indent=4)

        logger.info("Model tuning report saved at %s", report_path)
        return str(report_path)

    def initiate_model_tuning(self):
        try:
            logger.info("Model tuning initiated")

            X_train, y_train, X_test, y_test = self._load_data()
            task_type = self._infer_task_type(y_train)
            available_models = self._get_models(task_type)
            selected_models, baseline_scores = self._get_models_to_tune(available_models)

            tuned_results = []
            ranking_metric = getattr(self.config, "scoring_metric", None)
            if not ranking_metric:
                raise ValueError("scoring_metric must be configured")

            for model_name, model in selected_models.items():
                tuned_model, best_params, cv_best_score = self._tune_model(
                    model_name=model_name,
                    model=model,
                    X_train=X_train,
                    y_train=y_train,
                )
                predictions = tuned_model.predict(X_test)
                test_metrics = self._evaluate_model(
                    task_type=task_type,
                    y_true=y_test,
                    y_pred=predictions,
                )

                baseline_test_score = None
                if baseline_scores and model_name in baseline_scores:
                    baseline_test_score = baseline_scores[model_name].get(ranking_metric)

                logger.info(
                    "Tuned model %s test metrics: %s",
                    model_name,
                    test_metrics,
                )

                tuned_results.append(
                    {
                        "model_name": model_name,
                        "model": tuned_model,
                        "best_params": best_params,
                        "cv_best_score": cv_best_score,
                        "test_metrics": test_metrics,
                        "baseline_test_score": baseline_test_score,
                    }
                )

            if not tuned_results:
                raise ValueError("No models were tuned")

            ranked_tuned_results = self._rank_tuned_models(
                tuned_results=tuned_results,
                ranking_metric=ranking_metric,
            )
            tuned_model_paths = self._save_tuned_models(ranked_tuned_results)
            report_path = self._save_report(ranked_tuned_results)
            best_tuned_model_path = tuned_model_paths[0]
            best_score = float(ranked_tuned_results[0]["ranking_score"])

            return ModelTunerArtifact(
                best_tuned_model_path=best_tuned_model_path,
                tuned_model_paths=tuned_model_paths,
                report_path=report_path,
                best_score=best_score,
            )

        except Exception as e:
            logger.exception("Model tuning failed")
            raise CustomException(e, sys)
