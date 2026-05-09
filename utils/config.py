import os
from dotenv import load_dotenv

load_dotenv()

PG_HOST = os.getenv("PG_HOST", "192.168.1.13")
PG_PORT = int(os.getenv("PG_PORT", 5433))
PG_DB   = os.getenv("PG_DB", "chembl_36")
PG_USER = os.getenv("PG_USER", "chembl")
PG_PASS = os.getenv("PG_PASS", "chembl")

SPARK_MASTER = os.getenv("SPARK_MASTER", "spark://192.168.1.13:7077")
SPARK_DRIVER_HOST = os.getenv("SPARK_DRIVER_HOST", "192.168.1.13")

AWS_ACCESS_KEY_ID= os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "s3admin")
AWS_SECRET_ACCESS_KEY= os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "s3admin")
MLFLOW_S3_ENDPOINT_URL = os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9001")
MLFLOW_S3_IGNORE_TLS= os.environ["MLFLOW_S3_IGNORE_TLS"] = os.getenv("MLFLOW_S3_IGNORE_TLS", "true")
