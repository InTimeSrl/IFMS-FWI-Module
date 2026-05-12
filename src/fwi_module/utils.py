"""Utility helpers shared across the package."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ProcessingWindow:
    """Inclusive date window used for downloads and processing."""

    start: date
    end: date

    @property
    def identifier(self) -> str:
        return f"{self.start.isoformat()}_{self.end.isoformat()}"


def month_windows(start: date, end: date) -> list[ProcessingWindow]:
    """Split a date range into month-bounded inclusive windows."""

    windows: list[ProcessingWindow] = []
    current = start
    while current <= end:
        month_end = _month_end(current)
        window_end = month_end if month_end <= end else end
        windows.append(ProcessingWindow(start=current, end=window_end))
        if window_end == end:
            break
        current = window_end + timedelta(days=1)
    return windows


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def request_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _month_end(value: date) -> date:
    if value.month == 12:
        return date(value.year, 12, 31)
    next_month = date(value.year, value.month + 1, 1)
    return next_month - timedelta(days=1)