import json
import sys
from pathlib import Path

import pandas as pd
from pandas.api.types import is_dtype_equal

from src.utils.exception import CustomException
from src.utils.logger import logger


class DataValidationArtifact:
    def __init__(self, train_file_path, test_file_path, report_file_path, validation_status):
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.report_file_path = report_file_path
        self.validation_status = validation_status


class DataValidation:
    def __init__(self, config):
        self.config = config
        self.report = {
            "schema_validation": {},
            "missing_values_summary": {},
            "target_column_validation": {},
            "data_integrity": {},
            "data_drift": {},
            "warnings": [],
            "overall_status": "FAIL",
        }

    def _resolve_path(self, file_path):
        if not file_path:
            raise ValueError("File path is empty")
        return Path(file_path)

    def _ensure_parent_directory(self, file_path):
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    def _read_data(self, file_path, dataset_name):
        path = self._resolve_path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"{dataset_name} file does not exist: {path}")

        if not path.is_file():
            raise ValueError(f"{dataset_name} path is not a file: {path}")

        logger.info("Reading %s dataset from %s", dataset_name, path)
        df = pd.read_csv(path)

        logger.info("%s dataset loaded with shape %s", dataset_name, df.shape)
        return df

    def _get_expected_schema(self):
        schema = getattr(self.config, "schema", {})

        if not schema:
            return {}

        if isinstance(schema, dict) and "columns" in schema and isinstance(schema["columns"], dict):
            return schema["columns"]

        if isinstance(schema, dict):
            return schema

        raise ValueError("Schema must be a dictionary or contain a 'columns' dictionary")

    def _get_missing_threshold(self):
        return float(getattr(self.config, "missing_value_threshold", 0.2))

    def _get_drift_threshold(self):
        return float(getattr(self.config, "drift_threshold", 0.2))

    def _get_imbalance_threshold(self):
        return float(getattr(self.config, "imbalance_threshold", 0.9))

    def _validate_data_integrity(self, train_df, test_df):
        logger.info("Running data integrity checks")

        integrity_status = True
        issues = []

        if train_df.empty:
            raise ValueError("Train dataset is empty")

        if test_df.empty:
            raise ValueError("Test dataset is empty")

        train_columns = set(train_df.columns)
        test_columns = set(test_df.columns)

        if train_columns != test_columns:
            integrity_status = False
            issues.append("Train and test columns do not match")

        self.report["data_integrity"] = {
            "status": integrity_status,
            "train_shape": list(train_df.shape),
            "test_shape": list(test_df.shape),
            "column_mismatch": sorted(list(train_columns.symmetric_difference(test_columns))),
            "issues": issues,
        }

        return integrity_status

    def _validate_schema(self, train_df, test_df):
        logger.info("Running schema validation")

        expected_schema = self._get_expected_schema()
        expected_columns = set(expected_schema.keys())

        train_columns = set(train_df.columns)
        test_columns = set(test_df.columns)
        combined_columns = train_columns.union(test_columns)

        missing_columns = sorted(list(expected_columns - combined_columns))
        unexpected_columns = sorted(list(combined_columns - expected_columns)) if expected_columns else []

        dtype_mismatches = {}
        schema_status = not missing_columns and not unexpected_columns

        for column_name, expected_dtype in expected_schema.items():
            if column_name not in train_df.columns or column_name not in test_df.columns:
                continue

            train_dtype = train_df[column_name].dtype
            test_dtype = test_df[column_name].dtype
            expected_pd_dtype = pd.api.types.pandas_dtype(expected_dtype)

            train_matches = is_dtype_equal(train_dtype, expected_pd_dtype)
            test_matches = is_dtype_equal(test_dtype, expected_pd_dtype)

            if not train_matches or not test_matches:
                schema_status = False
                dtype_mismatches[column_name] = {
                    "expected": str(expected_pd_dtype),
                    "train_actual": str(train_dtype),
                    "test_actual": str(test_dtype),
                }

        self.report["schema_validation"] = {
            "status": schema_status,
            "expected_columns_count": len(expected_columns),
            "train_columns_count": len(train_columns),
            "test_columns_count": len(test_columns),
            "missing_columns": missing_columns,
            "unexpected_columns": unexpected_columns,
            "dtype_mismatches": dtype_mismatches,
        }

        return schema_status

    def _check_missing_values(self, train_df, test_df):
        logger.info("Running missing values analysis")

        threshold = self._get_missing_threshold()
        missing_summary = {}
        flagged_columns = []

        for column in train_df.columns:
            train_missing_pct = round(float(train_df[column].isnull().mean()), 4)
            test_missing_pct = round(float(test_df[column].isnull().mean()), 4) if column in test_df.columns else None

            missing_summary[column] = {
                "train_missing_percentage": train_missing_pct,
                "test_missing_percentage": test_missing_pct,
            }

            if train_missing_pct > threshold or (test_missing_pct is not None and test_missing_pct > threshold):
                flagged_columns.append(column)

        status = len(flagged_columns) == 0

        self.report["missing_values_summary"] = {
            "status": status,
            "threshold": threshold,
            "flagged_columns": flagged_columns,
            "columns": missing_summary,
        }

        return status

    def _validate_target_column(self, train_df, test_df):
        logger.info("Running target column validation")

        target_column = getattr(self.config, "target_column", None)
        if not target_column:
            raise ValueError("Target column is not configured")

        if target_column not in train_df.columns:
            raise ValueError(f"Target column '{target_column}' missing in train dataset")

        if target_column not in test_df.columns:
            raise ValueError(f"Target column '{target_column}' missing in test dataset")

        train_unique = int(train_df[target_column].nunique(dropna=True))
        test_unique = int(test_df[target_column].nunique(dropna=True))

        if train_unique < 2:
            raise ValueError(f"Train target column '{target_column}' must contain at least 2 unique values")

        if test_unique < 2:
            raise ValueError(f"Test target column '{target_column}' must contain at least 2 unique values")

        train_distribution = train_df[target_column].value_counts(normalize=True, dropna=False).to_dict()
        test_distribution = test_df[target_column].value_counts(normalize=True, dropna=False).to_dict()

        train_majority_ratio = max(train_distribution.values()) if train_distribution else 0.0
        test_majority_ratio = max(test_distribution.values()) if test_distribution else 0.0
        imbalance_threshold = self._get_imbalance_threshold()

        severe_imbalance = (
            train_majority_ratio > imbalance_threshold or test_majority_ratio > imbalance_threshold
        )

        if severe_imbalance:
            self.report["warnings"].append(
                f"Severe class imbalance detected in target column '{target_column}'"
            )

        self.report["target_column_validation"] = {
            "status": True,
            "target_column": target_column,
            "train_unique_values": train_unique,
            "test_unique_values": test_unique,
            "train_class_distribution": {str(k): round(float(v), 4) for k, v in train_distribution.items()},
            "test_class_distribution": {str(k): round(float(v), 4) for k, v in test_distribution.items()},
            "class_imbalance_warning": severe_imbalance,
            "imbalance_threshold": imbalance_threshold,
        }

        return True

    def _detect_data_drift(self, train_df, test_df):
        logger.info("Running basic data drift detection")

        drift_threshold = self._get_drift_threshold()
        target_column = getattr(self.config, "target_column", None)
        drift_results = {}
        drifted_columns = []

        common_columns = [column for column in train_df.columns if column in test_df.columns and column != target_column]

        for column in common_columns:
            if not pd.api.types.is_numeric_dtype(train_df[column]) or not pd.api.types.is_numeric_dtype(test_df[column]):
                continue

            train_mean = float(train_df[column].mean()) if not train_df[column].dropna().empty else 0.0
            test_mean = float(test_df[column].mean()) if not test_df[column].dropna().empty else 0.0
            train_std = float(train_df[column].std()) if not train_df[column].dropna().empty else 0.0
            test_std = float(test_df[column].std()) if not test_df[column].dropna().empty else 0.0

            mean_diff = abs(train_mean - test_mean)
            std_diff = abs(train_std - test_std)

            train_scale = max(abs(train_mean), abs(train_std), 1.0)
            test_scale = max(abs(test_mean), abs(test_std), 1.0)
            normalized_diff = max(mean_diff / train_scale, std_diff / test_scale)

            has_drift = normalized_diff > drift_threshold

            drift_results[column] = {
                "train_mean": round(train_mean, 4),
                "test_mean": round(test_mean, 4),
                "train_std": round(train_std, 4),
                "test_std": round(test_std, 4),
                "drift_score": round(float(normalized_diff), 4),
                "drift_detected": has_drift,
            }

            if has_drift:
                drifted_columns.append(column)

        drift_status = len(drifted_columns) == 0

        self.report["data_drift"] = {
            "status": drift_status,
            "threshold": drift_threshold,
            "drifted_columns": drifted_columns,
            "columns": drift_results,
        }

        return drift_status

    def _build_report_payload(self, validation_status):
        self.report["overall_status"] = "PASS" if validation_status else "FAIL"
        return self.report

    def _save_report(self, report_payload):
        report_path = self._resolve_path(getattr(self.config, "report_path", None))
        self._ensure_parent_directory(report_path)

        with report_path.open("w", encoding="utf-8") as report_file:
            json.dump(report_payload, report_file, indent=4)

        logger.info("Validation report saved at %s", report_path)
        return str(report_path)

    def initiate_data_validation(self):
        try:
            logger.info("Data validation initiated")

            train_df = self._read_data(self.config.train_path, "train")
            test_df = self._read_data(self.config.test_path, "test")

            integrity_status = self._validate_data_integrity(train_df, test_df)
            schema_status = self._validate_schema(train_df, test_df)
            missing_status = self._check_missing_values(train_df, test_df)
            target_status = self._validate_target_column(train_df, test_df)
            drift_status = self._detect_data_drift(train_df, test_df)

            validation_status = all(
                [integrity_status, schema_status, missing_status, target_status, drift_status]
            )

            report_payload = self._build_report_payload(validation_status)
            report_file_path = self._save_report(report_payload)

            logger.info("Data validation completed with status: %s", validation_status)

            return DataValidationArtifact(
                train_file_path=str(self._resolve_path(self.config.train_path)),
                test_file_path=str(self._resolve_path(self.config.test_path)),
                report_file_path=report_file_path,
                validation_status=validation_status,
            )

        except Exception as e:
            logger.exception("Data validation failed")
            raise CustomException(e, sys)
