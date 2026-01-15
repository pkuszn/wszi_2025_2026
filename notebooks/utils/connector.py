import psycopg2
import pandas as pd
from contextlib import contextmanager
from .config import PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASS

class PostgresConnector:
    def __init__(self):
        self.conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            database=PG_DB,
            user=PG_USER,
            password=PG_PASS
        )
    
    def query(self, sql, params=None) -> pd.DataFrame:
        return pd.read_sql(sql, self.conn, params=params)

    def execute(self, sql, params=None):
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            self.conn.commit()

    def close(self):
        self.conn.close()

@contextmanager
def postgres():
    conn = PostgresConnector()
    try:
        yield conn
    finally:
        conn.close()