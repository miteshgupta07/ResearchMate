from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.engine import URL
from core.config import Config as DatabaseConfig

DATABASE_NAME = DatabaseConfig.DB_NAME
TABLE_NAME = "chat_messages"

CREATE_DATABASE_SQL = """
SELECT 1 FROM pg_database WHERE datname = :db_name;
"""

CREATE_TABLE_SQL = f"""
CREATE TABLE {TABLE_NAME} (
    id BIGSERIAL PRIMARY KEY,
    session_id VARCHAR(128) NOT NULL,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

CHECK_TABLE_SQL = """
SELECT EXISTS (
    SELECT 1
    FROM information_schema.tables
    WHERE table_schema = 'public'
      AND table_name = :table_name
);
"""


def get_server_engine():
    """
    PostgreSQL requires database creation from a server-level connection.
    """
    url = URL.create(
        drivername="postgresql+psycopg2",
        username=DatabaseConfig.DB_USER,
        password=DatabaseConfig.DB_PASSWORD,
        host=DatabaseConfig.DB_HOST,
        port=DatabaseConfig.DB_PORT,
        database="postgres",  # system database
    )
    return create_engine(url, isolation_level="AUTOCOMMIT")


def get_database_engine():
    return create_engine(DatabaseConfig.get_connection_url())


def database_exists(connection) -> bool:
    result = connection.execute(
        text(CREATE_DATABASE_SQL),
        {"db_name": DATABASE_NAME},
    )
    return result.scalar() is not None


def create_database() -> None:
    engine = get_server_engine()

    try:
        with engine.connect() as connection:
            if database_exists(connection):
                print(f"Database '{DATABASE_NAME}' already exists.")
                return

            connection.execute(
                text(f'CREATE DATABASE "{DATABASE_NAME}"')
            )
            print(f"Database '{DATABASE_NAME}' created successfully.")

    except SQLAlchemyError as exc:
        print(f"Error creating database '{DATABASE_NAME}': {exc}")


def table_exists(connection) -> bool:
    result = connection.execute(
        text(CHECK_TABLE_SQL),
        {"table_name": TABLE_NAME},
    )
    return result.scalar()


def create_table() -> None:
    engine = get_database_engine()

    try:
        with engine.begin() as connection:
            if table_exists(connection):
                print(f"Table '{TABLE_NAME}' already exists.")
                return

            connection.execute(text(CREATE_TABLE_SQL))
            print(f"Table '{TABLE_NAME}' created successfully.")

    except SQLAlchemyError as exc:
        print(f"Error creating table '{TABLE_NAME}': {exc}")


if __name__ == "__main__":
    create_database()
    create_table()
