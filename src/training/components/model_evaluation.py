import json
import math
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from src.utils.common import load_object, save_object
from src.utils.exception import CustomException
from src.utils.logger import logger


class ModelEvaluatorArtifact:
    def __init__(self, final_model_path, evaluation_report_path, best_score):
        self.final_model_path = final_model_path
        self.evaluation_report_path = evaluation_report_path
        self.best_score = best_score


class ModelEvaluator:
    def __init__(self, config, model_trainer_artifact=None, model_tuner_artifact=None):
        self.config = config
        self.model_trainer_artifact = model_trainer_artifact
        self.model_tuner_artifact = model_tuner_artifact

    def _resolve_path(self, path_value):
        if not path_value:
            raise ValueError("Path is empty")
        return Path(path_value)

    def _ensure_parent_directory(self, file_path):
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    def _load_dataset(self, file_path):
        path = self._resolve_path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Test dataset does not exist: {path}")

        if not path.is_file():
            raise ValueError(f"Test dataset path is not a file: {path}")

        logger.info("Loading transformed test dataset from %s", path)
        df = pd.read_csv(path)

        if df.empty:
            raise ValueError("Transformed test dataset is empty")

        logger.info("Transformed test dataset loaded with shape %s", df.shape)
        return df

    def _load_test_data(self):
        test_df = self._load_dataset(self.config.transformed_test_path)
        target_column = getattr(self.config, "target_column", None)

        if not target_column:
            raise ValueError("Target column is not configured")

        if target_column not in test_df.columns:
            raise ValueError(f"Target column '{target_column}' missing in transformed test dataset")

        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column]

        if X_test.empty:
            raise ValueError("Test features are empty")

        return X_test, y_test

    def _infer_task_type(self, y_test):
        configured_task_type = getattr(self.config, "task_type", None)
        if configured_task_type:
            return configured_task_type.lower()

        return "classification" if y_test.nunique(dropna=True) <= 20 else "regression"

    def _extract_model_paths(self, source_name, configured_paths, artifact, artifact_attribute):
        paths = []

        if configured_paths:
            paths.extend(configured_paths)

        if artifact is not None and hasattr(artifact, artifact_attribute):
            artifact_value = getattr(artifact, artifact_attribute)
            if isinstance(artifact_value, list):
                paths.extend(artifact_value)
            elif artifact_value:
                paths.append(artifact_value)

        normalized_paths = []
        seen = set()
        for path_value in paths:
            resolved = str(self._resolve_path(path_value))
            if resolved not in seen:
                seen.add(resolved)
                normalized_paths.append(resolved)

        if not normalized_paths:
            logger.warning("No %s model paths were provided", source_name)

        return normalized_paths

    def _load_models(self):
        baseline_paths = self._extract_model_paths(
            source_name="baseline",
            configured_paths=getattr(self.config, "baseline_model_paths", None),
            artifact=self.model_trainer_artifact,
            artifact_attribute="top_models_paths",
        )
        tuned_paths = self._extract_model_paths(
            source_name="tuned",
            configured_paths=getattr(self.config, "tuned_model_paths", None),
            artifact=self.model_tuner_artifact,
            artifact_attribute="tuned_model_paths",
        )

        model_entries = []

        for path_value in baseline_paths:
            model_entries.append(
                {
                    "model_type": "baseline",
                    "model_path": path_value,
                    "model_name": Path(path_value).stem,
                    "model": self._load_single_model(path_value),
                }
            )

        for path_value in tuned_paths:
            model_entries.append(
                {
                    "model_type": "tuned",
                    "model_path": path_value,
                    "model_name": Path(path_value).stem,
                    "model": self._load_single_model(path_value),
                }
            )

        if not model_entries:
            raise ValueError("No baseline or tuned models were provided for evaluation")

        logger.info("Loaded %s models for evaluation", len(model_entries))
        return model_entries

    def _load_single_model(self, model_path):
        path = self._resolve_path(model_path)

        if not path.exists():
            raise FileNotFoundError(f"Model file does not exist: {path}")

        if not path.is_file():
            raise ValueError(f"Model path is not a file: {path}")

        logger.info("Loading model from %s", path)
        return load_object(str(path))

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

    def _evaluate_model(self, model, task_type, X_test, y_test):
        predictions = model.predict(X_test)

        if task_type == "classification":
            return self._classification_metrics(y_test, predictions)

        return self._regression_metrics(y_test, predictions)

    def _is_higher_score_better(self, metric_name):
        return metric_name.lower() in {"accuracy", "precision", "recall", "f1", "r2"}

    def _extract_train_scores(self):
        train_scores = {}

        trainer_report_path = getattr(self.config, "trainer_report_path", None)
        tuner_report_path = getattr(self.config, "tuner_report_path", None)
        ranking_metric = getattr(self.config, "evaluation_metric", None)

        if trainer_report_path:
            trainer_path = self._resolve_path(trainer_report_path)
            if trainer_path.exists():
                with trainer_path.open("r", encoding="utf-8") as report_file:
                    trainer_report = json.load(report_file)
                for item in trainer_report.get("ranking", []):
                    model_key = Path(item.get("model_path", item["model_name"])).stem
                    score = item.get("score")
                    if score is not None:
                        train_scores[model_key] = float(score)

        if tuner_report_path:
            tuner_path = self._resolve_path(tuner_report_path)
            if tuner_path.exists():
                with tuner_path.open("r", encoding="utf-8") as report_file:
                    tuner_report = json.load(report_file)
                for item in tuner_report.get("models", []):
                    model_key = Path(item.get("saved_model_path", item["model_name"])).stem
                    score = item.get("cv_best_score")
                    if score is not None:
                        train_scores[model_key] = float(score)

        return train_scores

    def _compare_models(self, model_results, evaluation_metric):
        metric_name = evaluation_metric.lower()

        for result in model_results:
            if metric_name not in result["metrics"]:
                raise ValueError(
                    f"Evaluation metric '{evaluation_metric}' is not available for model comparison"
                )
            result["score"] = float(result["metrics"][metric_name])

        ranked_results = sorted(
            model_results,
            key=lambda item: item["score"],
            reverse=self._is_higher_score_better(metric_name),
        )

        logger.info("Model comparison completed using metric: %s", evaluation_metric)
        return ranked_results

    def _check_threshold(self, best_score):
        threshold = float(getattr(self.config, "minimum_performance_threshold", 0.0))
        metric_name = getattr(self.config, "evaluation_metric", "")

        if self._is_higher_score_better(metric_name):
            passed = best_score >= threshold
        else:
            passed = best_score <= threshold

        return passed, threshold

    def _annotate_overfitting(self, ranked_results):
        overfitting_threshold = float(getattr(self.config, "overfitting_threshold", 0.1))
        train_scores = self._extract_train_scores()
        evaluation_metric = getattr(self.config, "evaluation_metric", None)

        for result in ranked_results:
            train_score = train_scores.get(result["model_name"])
            if train_score is None:
                result["train_reference_score"] = None
                result["overfitting_warning"] = False
                continue

            score_gap = abs(float(train_score) - float(result["score"]))
            result["train_reference_score"] = float(train_score)
            result["score_gap"] = float(score_gap)
            result["overfitting_warning"] = score_gap > overfitting_threshold

            if result["overfitting_warning"]:
                logger.warning(
                    "Possible overfitting detected for %s. Train reference score: %.6f, test score: %.6f",
                    result["model_name"],
                    train_score,
                    result["score"],
                )

    def _save_final_model(self, best_model):
        output_path = self._resolve_path(getattr(self.config, "evaluator_output_path", None))
        self._ensure_parent_directory(output_path)
        save_object(str(output_path), best_model)
        logger.info("Final production model saved at %s", output_path)
        return str(output_path)

    def _save_report(self, ranked_results, threshold_passed, threshold_value):
        output_path = self._resolve_path(getattr(self.config, "evaluator_output_path", None))
        report_path = output_path.parent / "model_evaluation_report.json"

        report_payload = {
            "evaluation_metric": getattr(self.config, "evaluation_metric", None),
            "best_model": ranked_results[0]["model_name"],
            "best_model_type": ranked_results[0]["model_type"],
            "best_score": ranked_results[0]["score"],
            "threshold_check": {
                "threshold": threshold_value,
                "passed": threshold_passed,
            },
            "models": [
                {
                    "rank": index,
                    "model_name": result["model_name"],
                    "model_type": result["model_type"],
                    "model_path": result["model_path"],
                    "score": result["score"],
                    "metrics": {metric: float(value) for metric, value in result["metrics"].items()},
                    "train_reference_score": result.get("train_reference_score"),
                    "score_gap": result.get("score_gap"),
                    "overfitting_warning": result.get("overfitting_warning", False),
                }
                for index, result in enumerate(ranked_results, start=1)
            ],
        }

        with report_path.open("w", encoding="utf-8") as report_file:
            json.dump(report_payload, report_file, indent=4)

        logger.info("Model evaluation report saved at %s", report_path)
        return str(report_path)

    def initiate_model_evaluation(self):
        try:
            logger.info("Model evaluation initiated")

            X_test, y_test = self._load_test_data()
            task_type = self._infer_task_type(y_test)
            evaluation_metric = getattr(self.config, "evaluation_metric", None)

            if not evaluation_metric:
                raise ValueError("evaluation_metric must be configured")

            model_entries = self._load_models()
            model_results = []

            for entry in model_entries:
                metrics = self._evaluate_model(
                    model=entry["model"],
                    task_type=task_type,
                    X_test=X_test,
                    y_test=y_test,
                )
                logger.info("Evaluation metrics for %s: %s", entry["model_name"], metrics)
                model_results.append(
                    {
                        "model_name": entry["model_name"],
                        "model_type": entry["model_type"],
                        "model_path": entry["model_path"],
                        "model": entry["model"],
                        "metrics": metrics,
                    }
                )

            ranked_results = self._compare_models(model_results, evaluation_metric)
            self._annotate_overfitting(ranked_results)

            best_result = ranked_results[0]
            threshold_passed, threshold_value = self._check_threshold(best_result["score"])

            if not threshold_passed:
                raise ValueError(
                    f"Best model score {best_result['score']:.6f} does not meet the minimum "
                    f"performance threshold of {threshold_value}"
                )

            final_model_path = self._save_final_model(best_result["model"])
            evaluation_report_path = self._save_report(
                ranked_results=ranked_results,
                threshold_passed=threshold_passed,
                threshold_value=threshold_value,
            )

            logger.info(
                "Final selected model: %s (%s) with score %.6f",
                best_result["model_name"],
                best_result["model_type"],
                float(best_result["score"]),
            )

            return ModelEvaluatorArtifact(
                final_model_path=final_model_path,
                evaluation_report_path=evaluation_report_path,
                best_score=float(best_result["score"]),
            )

        except Exception as e:
            logger.exception("Model evaluation failed")
            raise CustomException(e, sys)
