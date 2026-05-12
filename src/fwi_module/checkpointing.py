"""SQLite-backed cache catalog and processing checkpoint metadata."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .exceptions import CacheError
from .utils import ensure_directory


@dataclass(slots=True)
class DownloadRecord:
    cache_key: str
    dataset_id: str
    window_start: str
    window_end: str
    status: str
    file_path: str | None = None
    request_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class WindowRecord:
    window_id: str
    window_start: str
    window_end: str
    status: str
    output_path: str | None = None
    checkpoint_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class CatalogStore:
    """Persist cache metadata and processing progress in a local SQLite DB."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        ensure_directory(db_path.parent)
        self._initialize()

    def upsert_download(self, record: DownloadRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO downloads(cache_key, dataset_id, window_start, window_end, status, file_path, request_id, metadata_json, updated_at)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    dataset_id = excluded.dataset_id,
                    window_start = excluded.window_start,
                    window_end = excluded.window_end,
                    status = excluded.status,
                    file_path = excluded.file_path,
                    request_id = excluded.request_id,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    record.cache_key,
                    record.dataset_id,
                    record.window_start,
                    record.window_end,
                    record.status,
                    record.file_path,
                    record.request_id,
                    json.dumps(record.metadata, sort_keys=True),
                    _utc_now(),
                ),
            )

    def get_download(self, cache_key: str) -> DownloadRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT cache_key, dataset_id, window_start, window_end, status, file_path, request_id, metadata_json FROM downloads WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        return DownloadRecord(
            cache_key=row[0],
            dataset_id=row[1],
            window_start=row[2],
            window_end=row[3],
            status=row[4],
            file_path=row[5],
            request_id=row[6],
            metadata=json.loads(row[7] or "{}"),
        )

    def upsert_window(self, record: WindowRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO processing_windows(window_id, window_start, window_end, status, output_path, checkpoint_path, metadata_json, updated_at)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(window_id) DO UPDATE SET
                    window_start = excluded.window_start,
                    window_end = excluded.window_end,
                    status = excluded.status,
                    output_path = excluded.output_path,
                    checkpoint_path = excluded.checkpoint_path,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    record.window_id,
                    record.window_start,
                    record.window_end,
                    record.status,
                    record.output_path,
                    record.checkpoint_path,
                    json.dumps(record.metadata, sort_keys=True),
                    _utc_now(),
                ),
            )

    def get_window(self, window_id: str) -> WindowRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT window_id, window_start, window_end, status, output_path, checkpoint_path, metadata_json FROM processing_windows WHERE window_id = ?",
                (window_id,),
            ).fetchone()
        if row is None:
            return None
        return WindowRecord(
            window_id=row[0],
            window_start=row[1],
            window_end=row[2],
            status=row[3],
            output_path=row[4],
            checkpoint_path=row[5],
            metadata=json.loads(row[6] or "{}"),
        )

    def latest_completed_window(self) -> WindowRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT window_id, window_start, window_end, status, output_path, checkpoint_path, metadata_json
                FROM processing_windows
                WHERE status = 'completed'
                ORDER BY window_end DESC, updated_at DESC
                LIMIT 1
                """
            ).fetchone()
        if row is None:
            return None
        return WindowRecord(
            window_id=row[0],
            window_start=row[1],
            window_end=row[2],
            status=row[3],
            output_path=row[4],
            checkpoint_path=row[5],
            metadata=json.loads(row[6] or "{}"),
        )

    def summary(self) -> dict[str, int]:
        with self._connect() as connection:
            downloads = connection.execute("SELECT COUNT(*) FROM downloads").fetchone()[0]
            windows = connection.execute("SELECT COUNT(*) FROM processing_windows").fetchone()[0]
        return {"downloads": downloads, "processing_windows": windows}

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS downloads(
                    cache_key TEXT PRIMARY KEY,
                    dataset_id TEXT NOT NULL,
                    window_start TEXT NOT NULL,
                    window_end TEXT NOT NULL,
                    status TEXT NOT NULL,
                    file_path TEXT,
                    request_id TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS processing_windows(
                    window_id TEXT PRIMARY KEY,
                    window_start TEXT NOT NULL,
                    window_end TEXT NOT NULL,
                    status TEXT NOT NULL,
                    output_path TEXT,
                    checkpoint_path TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    updated_at TEXT NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        try:
            return sqlite3.connect(self.db_path)
        except sqlite3.Error as exc:
            raise CacheError(f"unable to open catalog database: {self.db_path}") from exc


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")