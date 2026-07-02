# src/database/connection.py
"""Database connection and session configuration."""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Ensure environment variables are loaded from config
from src.core import config

DATABASE_URL = os.getenv("DATABASE_URL")

# Fallback to local SQLite database if DATABASE_URL is not set (e.g. for local testing/development)
if not DATABASE_URL:
    # Ensure local directory exists
    os.makedirs("data", exist_ok=True)
    DATABASE_URL = "sqlite:///./data/dev.db"

# Some platforms (like Heroku) output postgres:// which SQLAlchemy 1.4+ does not support
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# SQLite requires different arguments than PostgreSQL
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,  # Test connections before using them
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    """Dependency for getting a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
