import psycopg

from contextlib import contextmanager

from secom_mlops_common.config.database import resolve_monitoring_database_url


def get_database_url() -> str:
    return resolve_monitoring_database_url()


@contextmanager
def connect():
    conn = psycopg.connect(get_database_url())
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
