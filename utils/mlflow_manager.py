import os
import pathlib  # Import the whole module
from dotenv import load_dotenv
import mlflow


class MLFlowManager:
    def __init__(self, experiment_name="ChEMBL_Analysis"):
        env_path = pathlib.Path(__file__).resolve().parents[1] / '.env'
        load_dotenv(dotenv_path=env_path)
        self._setup_connection()
        self.set_experiment(experiment_name)

    def _setup_connection(self):
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)
        
        os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "s3admin")
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "s3admin")
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9001")
        os.environ["MLFLOW_S3_IGNORE_TLS"] = os.getenv("MLFLOW_S3_IGNORE_TLS", "true")

    def set_experiment(self, name):
        mlflow.set_experiment(name)

    def start_run(self, run_name=None):
        return mlflow.start_run(run_name=run_name)