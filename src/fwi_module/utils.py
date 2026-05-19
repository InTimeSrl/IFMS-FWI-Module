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


@dataclass(frozen=True, slots=True)
class DownloadChunk:
    """Inclusive cached download chunk that may serve multiple processing windows."""

    window: ProcessingWindow
    request_months: tuple[int, ...] | None = None


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


def download_chunk_for_window(
    window: ProcessingWindow,
    *,
    chunking: str,
    period_start: date,
    period_end: date,
    allowed_months: tuple[int, ...] | None = None,
) -> DownloadChunk:
    """Resolve the cached download chunk that should back a processing window."""

    if chunking == "monthly":
        return DownloadChunk(window=window)

    if chunking == "quarterly":
        candidate_months = _quarter_months(window.start.month)
    elif chunking == "yearly":
        candidate_months = tuple(range(1, 13))
    else:  # pragma: no cover - guarded by configuration validation
        raise ValueError(f"unsupported download chunking mode: {chunking}")

    effective_start = max(period_start, _chunk_start(window.start, chunking))
    effective_end = min(period_end, _chunk_end(window.start, chunking))
    selected_months = _bounded_months_for_year(
        window.start.year,
        effective_start=effective_start,
        effective_end=effective_end,
        candidate_months=candidate_months,
        allowed_months=allowed_months,
    )
    return DownloadChunk(
        window=ProcessingWindow(start=effective_start, end=effective_end),
        request_months=selected_months,
    )


def months_are_contiguous(months: tuple[int, ...]) -> bool:
    """Return True when the month tuple defines a continuous month range."""

    return bool(months) and months == tuple(range(months[0], months[-1] + 1))


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def request_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _chunk_start(value: date, chunking: str) -> date:
    if chunking == "quarterly":
        return date(value.year, _quarter_months(value.month)[0], 1)
    return date(value.year, 1, 1)


def _chunk_end(value: date, chunking: str) -> date:
    if chunking == "quarterly":
        return _month_end(date(value.year, _quarter_months(value.month)[-1], 1))
    return date(value.year, 12, 31)


def _quarter_months(month: int) -> tuple[int, ...]:
    quarter_start = ((month - 1) // 3) * 3 + 1
    return tuple(range(quarter_start, quarter_start + 3))


def _bounded_months_for_year(
    year: int,
    *,
    effective_start: date,
    effective_end: date,
    candidate_months: tuple[int, ...],
    allowed_months: tuple[int, ...] | None,
) -> tuple[int, ...]:
    available_months = set(range(max(effective_start, date(year, 1, 1)).month, min(effective_end, date(year, 12, 31)).month + 1))
    selected_months = tuple(month for month in candidate_months if month in available_months)
    if allowed_months is not None:
        allowed = set(allowed_months)
        selected_months = tuple(month for month in selected_months if month in allowed)
    if not selected_months:  # pragma: no cover - defensive guard for invalid period/chunk combinations
        raise ValueError(f"download chunk for year {year} does not intersect the requested months")
    return selected_months


def _month_end(value: date) -> date:
    if value.month == 12:
        return date(value.year, 12, 31)
    next_month = date(value.year, value.month + 1, 1)
    return next_month - timedelta(days=1)