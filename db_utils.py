from functools import lru_cache
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from loguru import logger

from schemas import Base

load_dotenv()


@lru_cache(maxsize=1)
def init_qdrant_client() -> QdrantClient:
    logger.info("Initializing qdrant client")
    qdrant_host = os.environ.get("QDRANT_HOST", None)
    qdrant_port = os.environ.get("QDRANT_PORT", None)
    qdrant_path = os.environ.get("QDRANT_PATH", None)
    qdrant_timeout = float(os.environ.get("QDRANT_TIMEOUT", 5.0))
    qd_client = QdrantClient(
        host=qdrant_host, port=qdrant_port, path=qdrant_path, timeout=qdrant_timeout
    )
    logger.success("Initialized qdrant client")
    return qd_client


def init_sql_session() -> sessionmaker:
    logger.info("Initializing SQL session")
    db_url = os.environ.get("DB_URL", "sqlite:///sonaveeb.db")
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    session_maker = sessionmaker()
    session_maker.configure(bind=engine)
    logger.info("SQL session initialized")
    return session_maker
