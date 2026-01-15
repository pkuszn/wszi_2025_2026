from pyspark.sql import SparkSession

def create_spark(app_name: str = "chembl_eda") -> SparkSession:
    spark = SparkSession.builder \
        .master("spark://192.168.1.13:7077") \
        .appName("chembl_eda") \
        .config("spark.driver.host", "192.168.1.13") \
        .config("spark.driver.bindAddress", "0.0.0.0") \
        .config("spark.jars.packages", "org.postgresql:postgresql:42.6.0") \
        .getOrCreate()

    return spark

def read_postgres(
    spark: SparkSession,
    table: str,
    columns: list[str] | None = None,
    predicate: str | None = None
):
    jdbcDF = spark.read \
        .format("jdbc") \
        .option("url", "jdbc:postgresql://192.168.1.13:5433/chembl_36") \
        .option("dbtable", f"{table}") \
        .option("user", "chembl") \
        .option("password", "chembl") \
        .option("driver", "org.postgresql.Driver") \
        .load()
    
    return jdbcDF