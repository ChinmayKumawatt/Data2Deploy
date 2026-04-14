import sys
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler

from src.utils.config import load_stage_config
from src.utils.common import save_object
from src.utils.exception import CustomException
from src.utils.logger import logger


class DataTransformationArtifact:
    def __init__(self, transformed_train_path, transformed_test_path, preprocessor_object_path):
        self.transformed_train_path = transformed_train_path
        self.transformed_test_path = transformed_test_path
        self.preprocessor_object_path = preprocessor_object_path


class DataTransformation:
    def __init__(self, config):
        self.config = config

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

        if df.empty:
            raise ValueError(f"{dataset_name} dataset is empty")

        return df

    def _validate_target_column(self, train_df, test_df):
        target_column = getattr(self.config, "target_column", None)

        if not target_column:
            raise ValueError("Target column is not configured")

        if target_column not in train_df.columns:
            raise ValueError(f"Target column '{target_column}' missing in train dataset")

        if target_column not in test_df.columns:
            raise ValueError(f"Target column '{target_column}' missing in test dataset")

        return target_column

    def _separate_features_and_target(self, train_df, test_df):
        logger.info("Separating features and target column")
        target_column = self._validate_target_column(train_df, test_df)

        X_train = train_df.drop(columns=[target_column])
        y_train = train_df[target_column].copy()
        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column].copy()

        # If classification and targets are categorical strings, encode them
        task_type = self._get_task_type(y_train)
        if task_type == "classification" and not pd.api.types.is_numeric_dtype(y_train):
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            # Fit on both to safely capture all target categories existing in the dataset
            le.fit(pd.concat([y_train, y_test]).astype(str))
            
            y_train = pd.Series(le.transform(y_train.astype(str)), index=y_train.index, name=y_train.name)
            y_test = pd.Series(le.transform(y_test.astype(str)), index=y_test.index, name=y_test.name)

        if X_train.empty:
            raise ValueError("Train feature set is empty after dropping target column")

        if X_test.empty:
            raise ValueError("Test feature set is empty after dropping target column")

        return X_train, X_test, y_train, y_test

    def _infer_feature_types(self, X_train):
        logger.info("Determining numerical and categorical columns")

        numerical_columns = getattr(self.config, "numerical_columns", None)
        categorical_columns = getattr(self.config, "categorical_columns", None)

        if numerical_columns is None:
            numerical_columns = X_train.select_dtypes(include=["number"]).columns.tolist()

        if categorical_columns is None:
            categorical_columns = [
                column for column in X_train.columns if column not in set(numerical_columns)
            ]

        missing_columns = [
            column
            for column in numerical_columns + categorical_columns
            if column not in X_train.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Configured feature columns missing from dataset: {sorted(set(missing_columns))}"
            )

        duplicated_columns = set(numerical_columns).intersection(set(categorical_columns))
        if duplicated_columns:
            raise ValueError(
                f"Columns cannot be both numerical and categorical: {sorted(duplicated_columns)}"
            )

        uncovered_columns = [
            column
            for column in X_train.columns
            if column not in set(numerical_columns).union(set(categorical_columns))
        ]
        if uncovered_columns:
            logger.warning(
                "Columns not assigned explicitly to numerical/categorical groups will be treated as categorical: %s",
                uncovered_columns,
            )
            categorical_columns.extend(uncovered_columns)

        logger.info(
            "Detected %s numerical and %s categorical columns",
            len(numerical_columns),
            len(categorical_columns),
        )
        return list(numerical_columns), list(categorical_columns)

    def _validate_feature_presence(self, X_train, X_test, selected_features):
        missing_in_train = [feature for feature in selected_features if feature not in X_train.columns]
        missing_in_test = [feature for feature in selected_features if feature not in X_test.columns]

        if missing_in_train or missing_in_test:
            raise ValueError(
                "Selected features are not present in both datasets. "
                f"Missing in train: {missing_in_train}, missing in test: {missing_in_test}"
            )

    def _get_numeric_imputer_strategy(self):
        return getattr(self.config, "numerical_imputation_strategy", "mean")

    def _get_categorical_imputer_strategy(self):
        return getattr(self.config, "categorical_imputation_strategy", "most_frequent")

    def _get_scaling_strategy(self):
        return getattr(self.config, "scaling_method", "standard")

    def _get_task_type(self, y_train):
        configured_task_type = getattr(self.config, "task_type", None)
        if configured_task_type:
            return configured_task_type.lower()

        return "classification" if y_train.nunique(dropna=True) <= 20 else "regression"

    def _build_selector_preprocessor(self, numerical_columns, categorical_columns):
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy=self._get_numeric_imputer_strategy())),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy=self._get_categorical_imputer_strategy())),
                (
                    "encoder",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numerical_columns),
                ("cat", categorical_pipeline, categorical_columns),
            ],
            remainder="drop",
        )

    def _score_features(self, X_train, y_train, numerical_columns, categorical_columns):
        logger.info("Scoring features for automatic feature selection")

        selector_preprocessor = self._build_selector_preprocessor(
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
        )
        transformed_train = selector_preprocessor.fit_transform(X_train)

        ordered_columns = list(numerical_columns) + list(categorical_columns)
        if len(ordered_columns) == 0:
            raise ValueError("No features are available for automatic selection")

        task_type = self._get_task_type(y_train)
        selection_method = getattr(self.config, "feature_selection_method", None)

        if task_type == "regression":
            selection_method = selection_method or "correlation"
            if selection_method == "correlation":
                scores = []
                y_series = pd.Series(y_train).reset_index(drop=True)

                for index, column_name in enumerate(ordered_columns):
                    feature_series = pd.Series(transformed_train[:, index]).reset_index(drop=True)

                    if feature_series.nunique(dropna=True) <= 1 or y_series.nunique(dropna=True) <= 1:
                        score = 0.0
                    else:
                        score = abs(feature_series.corr(y_series))
                        if pd.isna(score):
                            score = 0.0
                    scores.append(score)
            elif selection_method == "mutual_info":
                scores = mutual_info_regression(transformed_train, y_train)
            else:
                raise ValueError(
                    f"Unsupported feature selection method '{selection_method}' for regression"
                )

        else:
            selection_method = selection_method or "mutual_info"
            if selection_method != "mutual_info":
                raise ValueError(
                    f"Unsupported feature selection method '{selection_method}' for classification"
                )
            scores = mutual_info_classif(transformed_train, y_train)

        feature_scores = pd.DataFrame(
            {
                "feature": ordered_columns,
                "score": [float(score) for score in scores],
            }
        ).sort_values(by="score", ascending=False)

        logger.info("Feature scores calculated using %s", selection_method)
        return feature_scores

    def _select_features(
        self,
        X_train,
        X_test,
        y_train,
        numerical_columns,
        categorical_columns,
    ):
        selection_mode = getattr(self.config, "feature_selection_mode", "manual").lower()
        logger.info("Applying feature selection in %s mode", selection_mode)

        if selection_mode == "manual":
            selected_features = list(getattr(self.config, "selected_features", []))
            if not selected_features:
                raise ValueError("selected_features must be provided when feature_selection_mode is 'manual'")

            self._validate_feature_presence(X_train, X_test, selected_features)
            return selected_features

        if selection_mode == "auto":
            n_features = int(getattr(self.config, "n_features", 0))
            if n_features <= 0:
                raise ValueError("n_features must be greater than 0 when feature_selection_mode is 'auto'")

            feature_scores = self._score_features(
                X_train=X_train,
                y_train=y_train,
                numerical_columns=numerical_columns,
                categorical_columns=categorical_columns,
            )

            selected_features = feature_scores.head(n_features)["feature"].tolist()
            if not selected_features:
                raise ValueError("Automatic feature selection did not return any features")

            self._validate_feature_presence(X_train, X_test, selected_features)
            logger.info("Selected top %s features: %s", len(selected_features), selected_features)
            return selected_features

        raise ValueError("feature_selection_mode must be either 'manual' or 'auto'")

    def _get_scaler(self):
        scaling_method = self._get_scaling_strategy().lower()

        if scaling_method == "standard":
            return StandardScaler()

        if scaling_method == "minmax":
            return MinMaxScaler()

        if scaling_method == "none":
            return "passthrough"

        raise ValueError("scaling_method must be one of: 'standard', 'minmax', 'none'")

    def _build_preprocessor(self, numerical_columns, categorical_columns):
        logger.info("Building preprocessing pipelines")

        numeric_steps = [
            ("imputer", SimpleImputer(strategy=self._get_numeric_imputer_strategy())),
        ]
        scaler = self._get_scaler()
        if scaler != "passthrough":
            numeric_steps.append(("scaler", scaler))

        numeric_pipeline = Pipeline(steps=numeric_steps)

        try:
            categorical_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            categorical_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy=self._get_categorical_imputer_strategy())),
                ("encoder", categorical_encoder),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numerical_columns),
                ("cat", categorical_pipeline, categorical_columns),
            ],
            remainder="drop",
        )

        return preprocessor

    def _transform_datasets(self, preprocessor, X_train, X_test):
        logger.info("Fitting preprocessor on train data and transforming datasets")
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        logger.info(
            "Transformation complete. Train shape: %s, Test shape: %s",
            X_train_transformed.shape,
            X_test_transformed.shape,
        )
        return X_train_transformed, X_test_transformed

    def _build_transformed_dataframe(self, transformed_array, y, feature_names):
        import re
        sanitized_names = []
        for name in feature_names:
            clean_name = re.sub(r'[\[\]<>]', '_', str(name))
            if clean_name in sanitized_names:
                clean_name = f"{clean_name}_{len(sanitized_names)}"
            sanitized_names.append(clean_name)
            
        transformed_df = pd.DataFrame(transformed_array, columns=sanitized_names, index=y.index)
        transformed_df[self.config.target_column] = y.values
        return transformed_df

    def _resolve_transformed_output_paths(self):
        transformed_train_path = getattr(self.config, "transformed_train_path", None)
        transformed_test_path = getattr(self.config, "transformed_test_path", None)

        if transformed_train_path and transformed_test_path:
            return str(self._resolve_path(transformed_train_path)), str(self._resolve_path(transformed_test_path))

        train_path = self._resolve_path(self.config.train_path)
        test_path = self._resolve_path(self.config.test_path)

        default_train_path = train_path.parent / "train_transformed.csv"
        default_test_path = test_path.parent / "test_transformed.csv"

        return str(default_train_path), str(default_test_path)

    def _save_transformed_datasets(
        self,
        X_train_transformed,
        X_test_transformed,
        y_train,
        y_test,
        preprocessor,
    ):
        feature_names = preprocessor.get_feature_names_out().tolist()
        train_df = self._build_transformed_dataframe(X_train_transformed, y_train, feature_names)
        test_df = self._build_transformed_dataframe(X_test_transformed, y_test, feature_names)

        transformed_train_path, transformed_test_path = self._resolve_transformed_output_paths()
        self._ensure_parent_directory(transformed_train_path)
        self._ensure_parent_directory(transformed_test_path)

        train_df.to_csv(transformed_train_path, index=False)
        test_df.to_csv(transformed_test_path, index=False)

        logger.info("Transformed train dataset saved at %s", transformed_train_path)
        logger.info("Transformed test dataset saved at %s", transformed_test_path)
        return transformed_train_path, transformed_test_path

    def _save_preprocessor_object(self, preprocessor):
        preprocessor_object_path = self._resolve_path(
            getattr(self.config, "preprocessor_object_path", None)
        )
        save_object(str(preprocessor_object_path), preprocessor)
        logger.info("Preprocessor object saved at %s", preprocessor_object_path)
        return str(preprocessor_object_path)

    def initiate_data_transformation(self):
        try:
            logger.info("Data transformation initiated")

            train_df = self._read_data(self.config.train_path, "train")
            test_df = self._read_data(self.config.test_path, "test")

            X_train, X_test, y_train, y_test = self._separate_features_and_target(train_df, test_df)
            numerical_columns, categorical_columns = self._infer_feature_types(X_train)

            selected_features = self._select_features(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                numerical_columns=numerical_columns,
                categorical_columns=categorical_columns,
            )

            X_train = X_train[selected_features].copy()
            X_test = X_test[selected_features].copy()

            numerical_columns = [column for column in numerical_columns if column in selected_features]
            categorical_columns = [column for column in categorical_columns if column in selected_features]

            preprocessor = self._build_preprocessor(
                numerical_columns=numerical_columns,
                categorical_columns=categorical_columns,
            )

            X_train_transformed, X_test_transformed = self._transform_datasets(
                preprocessor=preprocessor,
                X_train=X_train,
                X_test=X_test,
            )

            transformed_train_path, transformed_test_path = self._save_transformed_datasets(
                X_train_transformed=X_train_transformed,
                X_test_transformed=X_test_transformed,
                y_train=y_train,
                y_test=y_test,
                preprocessor=preprocessor,
            )

            preprocessor_object_path = self._save_preprocessor_object(preprocessor)

            logger.info("Data transformation completed successfully")

            return DataTransformationArtifact(
                transformed_train_path=transformed_train_path,
                transformed_test_path=transformed_test_path,
                preprocessor_object_path=preprocessor_object_path,
            )

        except Exception as e:
            logger.exception("Data transformation failed")
            raise CustomException(e, sys)


def main():
    config = load_stage_config("transformation")
    transformation = DataTransformation(config=config)
    transformation.initiate_data_transformation()


if __name__ == "__main__":
    main()
