"""CDS download layer with caching and dataset availability checks."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Protocol

from .checkpointing import CatalogStore, DownloadRecord
from .config import AppConfig, DatasetRequestConfig
from .exceptions import CredentialError, DatasetUnavailableError, DownloadError
from .runtime_logging import bind_log_context
from .utils import ProcessingWindow, ensure_directory, request_hash


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RequestSpec:
    label: str
    collection_id: str
    request_base: dict[str, Any]
    variables: list[str]
    times: list[str]
    data_format: str
    download_format: str


@dataclass(frozen=True, slots=True)
class DownloadedWindow:
    atmosphere_path: Path
    land_path: Path


class DataStoreBackend(Protocol):
    def check_authentication(self) -> None: ...

    def get_collection_end_date(self, collection_id: str) -> date | None: ...

    def retrieve(self, collection_id: str, request: dict[str, Any], target_path: Path) -> str | None: ...


class ECMWFDataStoresBackend:
    """Advanced CDS backend using ecmwf-datastores-client."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._client = None

    def check_authentication(self) -> None:
        try:
            self._get_client().check_authentication()
        except Exception as exc:  # pragma: no cover - depends on external service
            raise CredentialError("unable to authenticate against the CDS using ecmwf-datastores-client") from exc

    def get_collection_end_date(self, collection_id: str) -> date | None:
        try:
            collection = self._get_client().get_collection(collection_id)
        except Exception as exc:  # pragma: no cover - depends on external service
            raise DownloadError(f"unable to inspect collection metadata for {collection_id}") from exc
        end_datetime = getattr(collection, "end_datetime", None)
        if end_datetime is None:
            return None
        return end_datetime.date() if hasattr(end_datetime, "date") else date.fromisoformat(str(end_datetime)[:10])

    def retrieve(self, collection_id: str, request: dict[str, Any], target_path: Path) -> str | None:
        try:
            remote = self._get_client().submit(collection_id, request)
            remote.download(str(target_path))
        except Exception as exc:  # pragma: no cover - depends on external service
            raise DownloadError(f"download failed for collection {collection_id}: {exc}") from exc
        return getattr(remote, "request_id", None)

    def _get_client(self):
        if self._client is None:
            from ecmwf.datastores import Client

            url, key = _resolve_credentials(
                self.config,
                default_url="https://cds.climate.copernicus.eu/api",
                default_rc_name=".ecmwfdatastoresrc",
                fallback_url_envs=("CDSAPI_URL",),
                fallback_key_envs=("CDSAPI_KEY",),
            )
            kwargs: dict[str, str] = {}
            if url is not None:
                kwargs["url"] = url
            if key is not None:
                kwargs["key"] = key
            self._client = Client(**kwargs) if kwargs else Client()
        return self._client


class CdsApiBackend:
    """Compatibility backend using cdsapi."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._client = None

    def check_authentication(self) -> None:
        self._get_client()

    def get_collection_end_date(self, collection_id: str) -> date | None:
        return None

    def retrieve(self, collection_id: str, request: dict[str, Any], target_path: Path) -> str | None:
        try:
            self._get_client().retrieve(collection_id, request, str(target_path))
        except Exception as exc:  # pragma: no cover - depends on external service
            raise DownloadError(f"download failed for collection {collection_id}: {exc}") from exc
        return None

    def _get_client(self):
        if self._client is None:
            import cdsapi

            url, key = _resolve_credentials(
                self.config,
                default_url="https://cds.climate.copernicus.eu/api",
                default_rc_name=".cdsapirc",
                fallback_url_envs=("CDSAPI_URL", "ECMWF_DATASTORES_URL"),
                fallback_key_envs=("CDSAPI_KEY", "ECMWF_DATASTORES_KEY"),
            )
            kwargs: dict[str, str] = {}
            if url is not None:
                kwargs["url"] = url
            if key is not None:
                kwargs["key"] = key
            self._client = cdsapi.Client(**kwargs)
        return self._client


class CERRADataDownloader:
    """Download monthly CERRA subsets with local caching."""

    def __init__(self, config: AppConfig, catalog: CatalogStore, backend: DataStoreBackend | None = None) -> None:
        self.config = config
        self.catalog = catalog
        self.backend = backend or create_backend(config)

    def check_authentication(self) -> None:
        self.backend.check_authentication()

    def fetch_window(self, window: ProcessingWindow) -> DownloadedWindow:
        specs = self._build_specs(window)
        with bind_log_context(phase="download"):
            logger.info("Fetching CDS inputs for window %s -> %s", window.start.isoformat(), window.end.isoformat())
        atmosphere_path = self._download_spec(window, specs["atmosphere"])
        land_path = self._download_spec(window, specs["land"])
        return DownloadedWindow(atmosphere_path=atmosphere_path, land_path=land_path)

    def _download_spec(self, window: ProcessingWindow, spec: RequestSpec) -> Path:
        with bind_log_context(phase=f"download-{spec.label}"):
            request = self._build_request(spec, window)
            cache_key = request_hash({"collection_id": spec.collection_id, "request": request})
            extension = ".nc" if spec.data_format == "netcdf" else ".grib"
            target_path = ensure_directory(self.config.paths.cache_dir / spec.label) / f"{spec.collection_id}_{window.start:%Y%m%d}_{window.end:%Y%m%d}_{cache_key[:12]}{extension}"

            logger.info(
                "Prepared %s request for collection %s with %s variables and %s days",
                spec.label,
                spec.collection_id,
                len(spec.variables),
                len(request["day"]),
            )
            logger.info("Download target path: %s", target_path)

            cached = self.catalog.get_download(cache_key)
            if cached is not None and cached.status == "completed" and cached.file_path and Path(cached.file_path).exists():
                logger.info("Cache hit for %s dataset: %s", spec.label, cached.file_path)
                return Path(cached.file_path)

            logger.info("Cache miss for %s dataset; validating collection availability", spec.label)
            try:
                self._ensure_collection_available(spec.collection_id, window.end)
                self.catalog.upsert_download(
                    DownloadRecord(
                        cache_key=cache_key,
                        dataset_id=spec.collection_id,
                        window_start=window.start.isoformat(),
                        window_end=window.end.isoformat(),
                        status="running",
                        file_path=str(target_path),
                        metadata={"request": request, "label": spec.label},
                    )
                )
                logger.info("Submitting download for %s dataset", spec.label)
                request_id = self.backend.retrieve(spec.collection_id, request, target_path)
            except Exception:
                logger.exception("Download failed for %s dataset (%s)", spec.label, spec.collection_id)
                raise

            self.catalog.upsert_download(
                DownloadRecord(
                    cache_key=cache_key,
                    dataset_id=spec.collection_id,
                    window_start=window.start.isoformat(),
                    window_end=window.end.isoformat(),
                    status="completed",
                    file_path=str(target_path),
                    request_id=request_id,
                    metadata={"request": request, "label": spec.label},
                )
            )
            size_bytes = target_path.stat().st_size if target_path.exists() else "unknown"
            logger.info(
                "Download completed for %s dataset at %s (request_id=%s, size_bytes=%s)",
                spec.label,
                target_path,
                request_id,
                size_bytes,
            )
            return target_path

    def _ensure_collection_available(self, collection_id: str, requested_end: date) -> None:
        if not self.config.processing.fail_on_dataset_gap:
            return
        collection_end = self.backend.get_collection_end_date(collection_id)
        if collection_end is not None and requested_end > collection_end:
            raise DatasetUnavailableError(
                f"requested end date {requested_end.isoformat()} exceeds availability for {collection_id} (latest available: {collection_end.isoformat()})"
            )

    def _build_specs(self, window: ProcessingWindow) -> dict[str, RequestSpec]:
        return {
            "atmosphere": _request_spec_from_dataset("atmosphere", self.config.datasets.atmosphere),
            "land": _request_spec_from_dataset("land", self.config.datasets.land),
        }

    def _build_request(self, spec: RequestSpec, window: ProcessingWindow) -> dict[str, Any]:
        dates = _window_dates(window)
        request = dict(spec.request_base)
        request.update(
            {
                "variable": spec.variables,
                "year": sorted({f"{current.year:04d}" for current in dates}),
                "month": sorted({f"{current.month:02d}" for current in dates}),
                "day": [f"{current.day:02d}" for current in dates],
                "data_format": spec.data_format,
                "download_format": spec.download_format,
            }
        )
        if self.config.download.remote_area_subset:
            request["area"] = self.config.region.bbox.as_cds_area()
        if spec.times:
            request["time"] = spec.times
        return request


def create_backend(config: AppConfig) -> DataStoreBackend:
    if config.cds.client == "cdsapi":
        return CdsApiBackend(config)
    return ECMWFDataStoresBackend(config)


def _resolve_credentials(
    config: AppConfig,
    *,
    default_url: str,
    default_rc_name: str,
    fallback_url_envs: tuple[str, ...] = (),
    fallback_key_envs: tuple[str, ...] = (),
) -> tuple[str | None, str | None]:
    url = config.cds.url or _first_env(config.cds.url_env, *fallback_url_envs) or default_url
    key = config.cds.key or _first_env(config.cds.key_env, *fallback_key_envs)
    rc_override = os.getenv(config.cds.rc_file_env)
    rc_path = Path(rc_override).expanduser() if rc_override else Path.home() / default_rc_name

    if key is None and not rc_path.exists():
        raise CredentialError(
            "CDS credentials not found. Set the configured environment variables or provide a protected RC file."
        )
    return url, key


def _first_env(*names: str) -> str | None:
    for name in names:
        if not name:
            continue
        value = os.getenv(name)
        if value:
            return value
    return None


def _request_spec_from_dataset(label: str, dataset: DatasetRequestConfig) -> RequestSpec:
    return RequestSpec(
        label=label,
        collection_id=dataset.collection_id,
        request_base=dataset.request_base,
        variables=sorted(set(dataset.variable_map.values())),
        times=dataset.times,
        data_format=dataset.data_format,
        download_format=dataset.download_format,
    )


def _window_dates(window: ProcessingWindow) -> list[date]:
    dates: list[date] = []
    current = window.start
    while current <= window.end:
        dates.append(current)
        current = current.fromordinal(current.toordinal() + 1)
    return dates