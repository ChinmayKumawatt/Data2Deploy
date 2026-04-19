import json
from pathlib import Path


class MLflowService:
    def __init__(self, tracking_dir: Path, experiment_name: str = "Data2Deploy"):
        self.tracking_dir = Path(tracking_dir)
        self.experiment_name = experiment_name
        self.available = False
        self._mlflow = None
        self._setup()

    def _setup(self):
        try:
            import mlflow
        except ImportError:
            return

        self._mlflow = mlflow
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(self.tracking_dir.as_uri())
        mlflow.set_experiment(self.experiment_name)
        self.available = True

    def log_run(self, run_name: str, tags: dict, params: dict, metrics: dict, artifacts: list[str]) -> dict:
        if not self.available:
            return {
                "enabled": False,
                "tracking_uri": self.tracking_dir.as_uri(),
                "experiment_name": self.experiment_name,
                "run_id": None,
            }

        mlflow = self._mlflow
        clean_params = {key: self._stringify(value) for key, value in params.items()}
        clean_metrics = {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and value is not None
        }

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tags({key: self._stringify(value) for key, value in tags.items()})
            if clean_params:
                mlflow.log_params(clean_params)
            if clean_metrics:
                mlflow.log_metrics(clean_metrics)

            for artifact in artifacts:
                artifact_path = Path(artifact)
                if artifact_path.exists():
                    mlflow.log_artifact(str(artifact_path))

            run_id = run.info.run_id

        return {
            "enabled": True,
            "tracking_uri": self.tracking_dir.as_uri(),
            "experiment_name": self.experiment_name,
            "run_id": run_id,
        }

    def _stringify(self, value):
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value)
        return str(value)
