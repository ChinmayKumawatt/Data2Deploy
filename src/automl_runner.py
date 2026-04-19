import pandas as pd
from pathlib import Path

from src.services.automl_service import run_training_pipeline

def setup_automl_run(
    csv_path: str,
    target_column: str,
    feature_mode: str = "auto",
    n_features: int = 5,
    selected_features: list[str] | None = None,
):
    """
    Runs the AutoML pipeline against a dataset path and stores outputs in an isolated run folder.
    """
    df = pd.read_csv(csv_path)
    if feature_mode.lower() == "manual" and not selected_features:
        selected_features = [column for column in df.columns if column != target_column]

    return run_training_pipeline(
        file_bytes=Path(csv_path).read_bytes(),
        filename=Path(csv_path).name,
        target_column=target_column,
        feature_mode=feature_mode,
        n_features=n_features,
        selected_features=selected_features,
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the AutoML Pipeline dynamically.")
    parser.add_argument("--csv", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--target", type=str, required=True, help="Name of the target column")
    parser.add_argument("--feature_mode", type=str, choices=["auto", "manual"], default="auto", help="Feature selection mode (auto or manual)")
    parser.add_argument("--n_features", type=int, default=5, help="Number of features to select if in auto mode")
    parser.add_argument("--selected_features", nargs="*", default=None, help="Feature names for manual mode")
    
    args = parser.parse_args()
    
    setup_automl_run(
        csv_path=args.csv,
        target_column=args.target,
        feature_mode=args.feature_mode,
        n_features=args.n_features,
        selected_features=args.selected_features,
    )
