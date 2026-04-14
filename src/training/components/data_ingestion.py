import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.logger import logger
from src.utils.exception import CustomException

class DataIngestionArtifact:
    def __init__(self, train_file_path, test_file_path, raw_file_path=None):
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.raw_file_path = raw_file_path

class DataIngestion:
    def __init__(self,config):
        self.config = config

    def _validate_dataset_path(self):
        dataset_path = self.config.dataset_path

        if not dataset_path:
            raise ValueError("Dataset path is empty")

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file does not exist: {dataset_path}")

        if not os.path.isfile(dataset_path):
            raise ValueError(f"Dataset path is not a file: {dataset_path}")

    def _validate_dataframe_not_empty(self, df):
        if df.empty:
            raise ValueError("Dataset is empty")

        if df.shape[1] == 0:
            raise ValueError("Dataset has no columns")

    def _validate_target_column(self, df):
        if self.config.target_column not in df.columns:
            raise ValueError(
                f"Target column '{self.config.target_column}' not found in dataset columns"
            )

    def _validate_target_values(self, df):
        target_series = df[self.config.target_column]

        if target_series.empty:
            raise ValueError("Target column is empty")

        if target_series.isnull().all():
            raise ValueError("Target column contains only null values")

        if target_series.isnull().any():
            raise ValueError("Target column contains null values")

    def _validate_minimum_rows(self, df):
        min_rows = getattr(self.config, "min_rows", 200)

        if len(df) < min_rows:
            raise ValueError(
                f"Dataset has {len(df)} rows, which is below the minimum required size of {min_rows}"
            )

    def _validate_split_config(self, df):
        test_size = self.config.test_size

        if isinstance(test_size, float):
            if test_size <= 0 or test_size >= 1:
                raise ValueError("test_size as float must be between 0 and 1")
        elif isinstance(test_size, int):
            if test_size <= 0:
                raise ValueError("test_size as int must be greater than 0")
            if test_size >= len(df):
                raise ValueError("test_size as int must be smaller than the dataset size")
        else:
            raise ValueError("test_size must be either float or int")

    def _resolve_raw_data_path(self):
        raw_data_path = getattr(self.config, "raw_path", None)

        if raw_data_path:
            return raw_data_path

        train_path = Path(self.config.train_path)
        return str(train_path.parent / "raw.csv")

    def _should_use_stratify(self, y):
        task_type = getattr(self.config, "task_type", "classification")
        stratify = getattr(self.config, "stratify", True)

        if not stratify or task_type != "classification":
            return None

        if y.nunique(dropna=False) < 2:
            logger.warning(
                "Stratified split skipped because target column has fewer than 2 unique classes"
            )
            return None

        class_counts = y.value_counts()
        if class_counts.empty or class_counts.min() < 2:
            logger.warning(
                "Stratified split skipped because at least one class has fewer than 2 samples"
            )
            return None

        return y

    def _ensure_parent_directory(self, file_path):
        parent_dir = Path(file_path).parent
        parent_dir.mkdir(parents=True, exist_ok=True)

    def initiate_data_ingestion(self):
        try:
            logger.info("Data ingestion initiated")

            self._validate_dataset_path()
            df = pd.read_csv(self.config.dataset_path)
            logger.info("Dataset read from %s", self.config.dataset_path)

            original_length = len(df)
            logger.info("Length of dataset is %s", original_length)

            self._validate_dataframe_not_empty(df)
            self._validate_target_column(df)

            df = df.drop_duplicates()
            deduplicated_length = len(df)
            logger.info(
                "Dataset deduplicated from %s to %s rows",
                original_length,
                deduplicated_length
            )

            self._validate_dataframe_not_empty(df)
            self._validate_minimum_rows(df)
            self._validate_target_values(df)
            self._validate_split_config(df)

            raw_data_path = self._resolve_raw_data_path()
            self._ensure_parent_directory(raw_data_path)
            df.to_csv(raw_data_path, index=False)
            logger.info("Raw dataset saved successfully at %s", raw_data_path)

            X = df.drop(columns=[self.config.target_column])
            y = df[self.config.target_column]

            stratify_data = self._should_use_stratify(y)
            X_train,X_test,y_train,y_test = train_test_split(
                X,
                y,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=stratify_data
            )

            train_df = pd.concat([X_train,y_train],axis=1)
            test_df = pd.concat([X_test,y_test],axis=1)
            logger.info(f"Shape of train set {train_df.shape}")
            logger.info(f"Shape of test set {test_df.shape}")

            self._ensure_parent_directory(self.config.train_path)
            self._ensure_parent_directory(self.config.test_path)

            train_df.to_csv(self.config.train_path, index = False)
            test_df.to_csv(self.config.test_path, index = False)
            logger.info("Train and test datasets saved successfully")

            return DataIngestionArtifact(
                train_file_path=self.config.train_path,
                test_file_path=self.config.test_path,
                raw_file_path=raw_data_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)
