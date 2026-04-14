import os
import sys
import subprocess
import yaml
import pandas as pd
from pathlib import Path

def setup_automl_run(csv_path: str, target_column: str, feature_mode: str = 'auto', n_features: int = 5):
    """
    Dynamically configures params.yaml based on a new dataset and runs the ml pipeline.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Provided dataset not found: {csv_path}")

    print(f"[*] Analyzing dataset: {csv_path}")
    # Read the dataset to infer schemas
    df = pd.read_csv(csv_path)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")

    # Drop high-cardinality ID columns (e.g., > 95% unique values)
    dropped_cols = []
    for col in list(df.columns):
        if col != target_column and df[col].nunique() / len(df) > 0.95:
            df.drop(columns=[col], inplace=True)
            dropped_cols.append(col)
            
    if dropped_cols:
        print(f"[*] Dropped high-cardinality ID columns: {dropped_cols}")
        df.to_csv(csv_path, index=False)

    # 1. Infer schema
    dynamic_schema = {str(col): str(dtype) for col, dtype in df.dtypes.items()}

    # 2. Infer Task Type
    target_nunique = df[target_column].nunique()
    task_type = "classification" if target_nunique <= 20 else "regression"
    print(f"[*] Deduced Task Type: {task_type} (Target unique values: {target_nunique})")

    # 3. Read params.yaml
    params_file = "params.yaml"
    with open(params_file, "r") as file:
        params = yaml.safe_load(file)

    print("[*] Updating params.yaml...")

    # Update Ingestion
    if "ingestion" in params:
        params["ingestion"]["dataset_path"] = csv_path
        params["ingestion"]["target_column"] = target_column
        params["ingestion"]["task_type"] = task_type

    # Update Validation
    if "validation" in params:
        params["validation"]["target_column"] = target_column
        params["validation"]["schema"]["columns"] = dynamic_schema

    # Update Transformation
    if "transformation" in params:
        params["transformation"]["target_column"] = target_column
        params["transformation"]["task_type"] = task_type
        params["transformation"]["feature_selection_mode"] = feature_mode
        params["transformation"]["n_features"] = n_features
        # Clear out explicit definitions so the pipeline auto-infers them
        params["transformation"]["numerical_columns"] = None
        params["transformation"]["categorical_columns"] = None
        params["transformation"]["selected_features"] = []

        # If manual and user supplied nothing, we shouldn't fail, but let's assume they want auto here 
        # or we could default to taking all columns except target
        if feature_mode.lower() == "manual":
            print("[*] Warning: 'manual' mode selected without providing selected_features. Defaulting to all features.")
            features = [col for col in df.columns if col != target_column]
            params["transformation"]["selected_features"] = list(features)

    # Update Training
    if "training" in params:
        params["training"]["target_column"] = target_column
        params["training"]["task_type"] = task_type
        
        if task_type == "regression":
            if "classifier" in params["training"].get("selected_model", ""):
                params["training"]["selected_model"] = "random_forest_regressor"
            params["training"]["evaluation_metric"] = "r2"
        else:
            if "regressor" in params["training"].get("selected_model", ""):
                params["training"]["selected_model"] = "random_forest_classifier"
            params["training"]["evaluation_metric"] = "f1"

    # Update Tuning
    if "tuning" in params:
        params["tuning"]["target_column"] = target_column
        params["tuning"]["task_type"] = task_type
        if task_type == "regression":
            params["tuning"]["scoring_metric"] = "r2"
        else:
            params["tuning"]["scoring_metric"] = "f1_weighted"
            
    # Update Evaluation
    if "evaluation" in params:
        params["evaluation"]["target_column"] = target_column
        params["evaluation"]["task_type"] = task_type
        if task_type == "regression":
            params["evaluation"]["evaluation_metric"] = "r2"
        else:
            params["evaluation"]["evaluation_metric"] = "f1"

    # Save changes to params.yaml
    with open(params_file, "w") as file:
        yaml.dump(params, file, default_flow_style=False, sort_keys=False)

    print("[*] params.yaml successfully updated!")

    # 4. Trigger DVC Reproduce
    print("[*] Triggering DVC Pipeline execution...")
    try:
        subprocess.run("dvc repro --force", check=True, shell=True)
        print("[*] AutoML Pipeline execution completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"[!] DVC Pipeline failed during execution. Please check the logs.")
        raise Exception(f"DVC pipeline failed with exit code: {e.returncode}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the AutoML Pipeline dynamically.")
    parser.add_argument("--csv", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--target", type=str, required=True, help="Name of the target column")
    parser.add_argument("--feature_mode", type=str, choices=["auto", "manual"], default="auto", help="Feature selection mode (auto or manual)")
    parser.add_argument("--n_features", type=int, default=5, help="Number of features to select if in auto mode")
    
    args = parser.parse_args()
    
    setup_automl_run(
        csv_path=args.csv,
        target_column=args.target,
        feature_mode=args.feature_mode,
        n_features=args.n_features
    )
