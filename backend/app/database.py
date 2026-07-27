import sqlite3
from contextlib import contextmanager
from typing import Generator

from .config import DB_PATH

@contextmanager
def db() -> Generator[sqlite3.Connection, None, None]:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
    finally:
        connection.commit()
        connection.close()

def initialize_database() -> None:
    with db() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS app_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS generations (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                video_path TEXT NOT NULL,
                duration_seconds REAL,
                generation_time REAL NOT NULL,
                seed INTEGER,
                created_at REAL NOT NULL
            )
            """
        )
