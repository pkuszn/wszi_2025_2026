import os

PG_HOST = os.getenv("PG_HOST", "192.168.1.13")
PG_PORT = int(os.getenv("PG_PORT", 5433))
PG_DB   = os.getenv("PG_DB", "chembl_36")
PG_USER = os.getenv("PG_USER", "chembl")
PG_PASS = os.getenv("PG_PASS", "chembl")

SPARK_MASTER = os.getenv("SPARK_MASTER", "spark://192.168.1.13:7077")
SPARK_DRIVER_HOST = os.getenv("SPARK_DRIVER_HOST", "192.168.1.13")