import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from urllib.parse import quote



# engine = create_engine(database_url)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

engine = None
SessionLocal = None

def get_engine():
    global engine
    if engine is None:
        database_url = os.getenv("DATABASE_URL")
        if database_url is None:
            raise ValueError("No DATABASE_URL environment variable found.")
        engine = create_engine(database_url)
    return engine

def get_session_local():
    return sessionmaker(autocommit=False, autoflush=False, bind=get_engine())()
