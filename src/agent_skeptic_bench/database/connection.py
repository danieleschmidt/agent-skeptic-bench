"""Database connection management for Agent Skeptic Bench."""

import logging
import os
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from .models import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and sessions."""

    def __init__(self, database_url: str | None = None, echo: bool = False):
        """Initialize database manager.
        
        Args:
            database_url: Database URL. If None, reads from environment.
            echo: Whether to echo SQL statements for debugging.
        """
        self.database_url = database_url or self._get_database_url()
        self.echo = echo

        # Create engines
        self.engine = sa.create_engine(
            self._get_sync_url(),
            echo=echo,
            pool_pre_ping=True,
            pool_recycle=300
        )

        self.async_engine = create_async_engine(
            self.database_url,
            echo=echo,
            pool_pre_ping=True,
            pool_recycle=300
        )

        # Create session factories
        self.SessionLocal = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False
        )

        self.AsyncSessionLocal = sessionmaker(
            bind=self.async_engine,
            class_=AsyncSession,
            autocommit=False,
            autoflush=False
        )

        logger.info(f"Database manager initialized with URL: {self._mask_url(self.database_url)}")

    def _get_database_url(self) -> str:
        """Get database URL from environment variables."""
        # Try DATABASE_URL first
        database_url = os.getenv("DATABASE_URL")
        if database_url:
            return self._ensure_async_driver(database_url)

        # Build from individual components
        user = os.getenv("POSTGRES_USER", "postgres")
        password = os.getenv("POSTGRES_PASSWORD", "")
        host = os.getenv("POSTGRES_HOST", "localhost")
        port = os.getenv("POSTGRES_PORT", "5432")
        database = os.getenv("POSTGRES_DB", "skeptic_bench")

        return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}"

    def _ensure_async_driver(self, url: str) -> str:
        """Ensure the URL uses an async driver."""
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+asyncpg://")
        elif url.startswith("postgresql+psycopg2://"):
            return url.replace("postgresql+psycopg2://", "postgresql+asyncpg://")
        return url

    def _get_sync_url(self) -> str:
        """Get synchronous database URL."""
        return self.database_url.replace("+asyncpg", "").replace("+psycopg2", "")

    def _mask_url(self, url: str) -> str:
        """Mask sensitive information in URL for logging."""
        if "@" in url:
            parts = url.split("@")
            if "//" in parts[0]:
                protocol_and_auth = parts[0].split("//")
                if ":" in protocol_and_auth[1]:
                    user_pass = protocol_and_auth[1].split(":")
                    masked = f"{protocol_and_auth[0]}//{user_pass[0]}:***"
                    return f"{masked}@{parts[1]}"
        return url

    async def create_tables(self) -> None:
        """Create all database tables."""
        async with self.async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully")

    async def drop_tables(self) -> None:
        """Drop all database tables."""
        async with self.async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        logger.info("Database tables dropped successfully")

    def create_tables_sync(self) -> None:
        """Create all database tables synchronously."""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created successfully (sync)")

    def drop_tables_sync(self) -> None:
        """Drop all database tables synchronously."""
        Base.metadata.drop_all(bind=self.engine)
        logger.info("Database tables dropped successfully (sync)")

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get a synchronous database session."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get an asynchronous database session."""
        async with self.AsyncSessionLocal() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def health_check(self) -> bool:
        """Check if database is healthy."""
        try:
            async with self.async_engine.begin() as conn:
                await conn.execute(sa.text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

    async def close(self) -> None:
        """Close database connections."""
        await self.async_engine.dispose()
        self.engine.dispose()
        logger.info("Database connections closed")


# Global database manager instance
_db_manager: DatabaseManager | None = None


def get_database(database_url: str | None = None, echo: bool = False) -> DatabaseManager:
    """Get global database manager instance."""
    global _db_manager

    if _db_manager is None:
        _db_manager = DatabaseManager(database_url, echo)

    return _db_manager


def reset_database() -> None:
    """Reset global database manager (for testing)."""
    global _db_manager
    _db_manager = None


# Dependency injection for FastAPI
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for database sessions."""
    db = get_database()
    async with db.get_async_session() as session:
        yield session
