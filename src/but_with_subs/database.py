"""Database layer for storing processed-file references.

Uses SQLModel with a Postgres engine (configured via the ``DATABASE_URL``
environment variable). Falls back to a local SQLite file when the env var
is not set, so the API also runs outside Docker for local development.
"""

import collections.abc as c
import os
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.engine import Engine
from sqlmodel import Field, Session, SQLModel, create_engine

from .logging_config import logger


class FileRecord(SQLModel, table=True):
    """Database row mapping a source URL to its downloaded media + subtitles."""

    __tablename__ = "files"

    url: str = Field(primary_key=True)
    video_path: str | None = None
    audio_path: str | None = None
    subtitles_path: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def _default_sqlite_url() -> str:
    """Return a sqlite URL pointing at ``./data/files.db``.

    Returns:
        SQLAlchemy-style sqlite URL.
    """
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{data_dir / 'files.db'}"


def build_engine(url: str | None = None) -> Engine:
    """Construct a SQLModel engine.

    Args:
        url: Optional explicit database URL. If ``None``, reads ``DATABASE_URL``
            from the environment, then falls back to a local sqlite file.

    Returns:
        Configured SQLAlchemy engine ready for SQLModel sessions.
    """
    db_url = url or os.environ.get("DATABASE_URL") or _default_sqlite_url()
    connect_args: dict[str, object] = {}
    if db_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    logger.info(f"Connecting to database at {db_url}")
    return create_engine(db_url, connect_args=connect_args)


def init_db(engine: Engine) -> None:
    """Create tables if they do not yet exist."""
    SQLModel.metadata.create_all(engine)


def get_session(engine: Engine) -> c.Iterator[Session]:
    """Yield a SQLModel session bound to ``engine``."""
    with Session(engine) as session:
        yield session


def upsert_file(
    session: Session,
    url: str,
    video_path: Path | None = None,
    audio_path: Path | None = None,
    subtitles_path: Path | None = None,
) -> FileRecord:
    """Insert or update the ``files`` row for ``url``.

    Only non-``None`` arguments overwrite existing values, so partial updates
    (e.g. setting just ``subtitles_path`` later) are safe.

    Returns:
        The persisted ``FileRecord``.
    """
    record = session.get(FileRecord, url)
    if record is None:
        record = FileRecord(
            url=url,
            video_path=str(video_path) if video_path else None,
            audio_path=str(audio_path) if audio_path else None,
            subtitles_path=str(subtitles_path) if subtitles_path else None,
        )
        session.add(record)
    else:
        if video_path is not None:
            record.video_path = str(video_path)
        if audio_path is not None:
            record.audio_path = str(audio_path)
        if subtitles_path is not None:
            record.subtitles_path = str(subtitles_path)
    session.commit()
    session.refresh(record)
    return record
