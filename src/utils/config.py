import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

from src.utils.exception import CustomException


def load_stage_config(stage_name, params_file_path="params.yaml"):
    try:
        params_path = Path(params_file_path)
        if not params_path.exists():
            raise FileNotFoundError(f"Params file does not exist: {params_path}")

        with params_path.open("r", encoding="utf-8") as params_file:
            params = yaml.safe_load(params_file) or {}

        if stage_name not in params:
            raise KeyError(f"Stage '{stage_name}' not found in params file")

        stage_params = params[stage_name]
        if not isinstance(stage_params, dict):
            raise ValueError(f"Stage '{stage_name}' configuration must be a dictionary")

        return SimpleNamespace(**stage_params)

    except Exception as e:
        raise CustomException(e, sys)
