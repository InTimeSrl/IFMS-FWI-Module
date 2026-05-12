"""YAML configuration models for the FWI processing pipeline."""

from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .exceptions import ConfigError


class BoundingBox(BaseModel):
    """Geographic bounding box expressed as north, west, south, east."""

    model_config = ConfigDict(extra="forbid")

    north: float
    west: float
    south: float
    east: float

    @model_validator(mode="after")
    def validate_bounds(self) -> "BoundingBox":
        if self.north <= self.south:
            raise ValueError("bbox.north must be greater than bbox.south")
        if self.east <= self.west:
            raise ValueError("bbox.east must be greater than bbox.west")
        return self

    def as_cds_area(self) -> list[float]:
        return [self.north, self.west, self.south, self.east]


class RegionConfig(BaseModel):
    """Region of interest for the computation."""

    model_config = ConfigDict(extra="forbid")

    bbox: BoundingBox
    country_name: str = "Greece"


class PeriodConfig(BaseModel):
    """Target time interval and spinup settings."""

    model_config = ConfigDict(extra="forbid")

    start: date
    end: date
    spinup_days: int = Field(default=90, ge=0, le=366)

    @model_validator(mode="after")
    def validate_period(self) -> "PeriodConfig":
        if self.end < self.start:
            raise ValueError("period.end must be on or after period.start")
        return self

    @property
    def extended_start(self) -> date:
        return self.start - timedelta(days=self.spinup_days)


class CDSConfig(BaseModel):
    """CDS client and credential settings."""

    model_config = ConfigDict(extra="forbid")

    client: Literal["ecmwf_datastores", "cdsapi"] = "ecmwf_datastores"
    url: str | None = None
    key: str | None = None
    url_env: str = "ECMWF_DATASTORES_URL"
    key_env: str = "ECMWF_DATASTORES_KEY"
    rc_file_env: str = "ECMWF_DATASTORES_RC_FILE"
    timeout_seconds: int = Field(default=600, ge=30, le=86_400)


class DatasetRequestConfig(BaseModel):
    """Dataset-specific CDS request details."""

    model_config = ConfigDict(extra="forbid")

    collection_id: str
    request_base: dict[str, Any] = Field(default_factory=dict)
    variable_map: dict[str, str] = Field(default_factory=dict)
    times: list[str] = Field(default_factory=list)
    data_format: Literal["grib", "netcdf"] = "grib"
    download_format: str = "unarchived"

    @field_validator("times")
    @classmethod
    def validate_times(cls, values: list[str]) -> list[str]:
        for value in values:
            if len(value) != 5 or value[2] != ":":
                raise ValueError(f"invalid time value: {value!r}")
        return values


class DatasetsConfig(BaseModel):
    """Default CERRA collections used by the pipeline."""

    model_config = ConfigDict(extra="forbid")

    atmosphere: DatasetRequestConfig = Field(
        default_factory=lambda: DatasetRequestConfig(
            collection_id="reanalysis-cerra-single-levels",
            request_base={"data_type": ["reanalysis"]},
            variable_map={
                "temperature": "2m_temperature",
                "relative_humidity": "2m_relative_humidity",
                "wind_speed": "10m_wind_speed",
            },
            times=["12:00"],
            data_format="grib",
        )
    )
    land: DatasetRequestConfig = Field(
        default_factory=lambda: DatasetRequestConfig(
            collection_id="reanalysis-cerra-land",
            request_base={},
            variable_map={
                "precipitation": "total_precipitation",
                "land_sea_mask": "land_sea_mask",
            },
            times=["06:00"],
            data_format="grib",
        )
    )

    @model_validator(mode="after")
    def validate_required_variables(self) -> "DatasetsConfig":
        missing_atmos = {"temperature", "relative_humidity", "wind_speed"} - set(self.atmosphere.variable_map)
        if missing_atmos:
            missing = ", ".join(sorted(missing_atmos))
            raise ValueError(f"datasets.atmosphere.variable_map is missing: {missing}")
        missing_land = {"precipitation", "land_sea_mask"} - set(self.land.variable_map)
        if missing_land:
            missing = ", ".join(sorted(missing_land))
            raise ValueError(f"datasets.land.variable_map is missing: {missing}")
        return self


class DownloadConfig(BaseModel):
    """Download chunking, retries and cache behaviour."""

    model_config = ConfigDict(extra="forbid")

    chunking: Literal["monthly", "quarterly"] = "monthly"
    retry_attempts: int = Field(default=4, ge=1, le=20)
    retry_wait_seconds: int = Field(default=30, ge=1, le=3_600)


class PathsConfig(BaseModel):
    """Filesystem locations used by the pipeline."""

    model_config = ConfigDict(extra="forbid")

    cache_dir: Path = Path("data/cache")
    output_dir: Path = Path("data/output")
    state_dir: Path = Path("data/state")
    catalog_db: Path | None = None

    @model_validator(mode="after")
    def apply_defaults(self) -> "PathsConfig":
        if self.catalog_db is None:
            self.catalog_db = self.state_dir / "catalog.sqlite"
        return self


class StorageConfig(BaseModel):
    """Output encoding and checkpoint naming."""

    model_config = ConfigDict(extra="forbid")

    format: Literal["netcdf4"] = "netcdf4"
    compression_level: int = Field(default=4, ge=0, le=9)
    chunk_shape: tuple[int, int, int] = (1, 128, 128)
    include_inputs: bool = True
    filename_template: str = "fwi_{year}{month:02d}.nc"
    state_template: str = "state_{year}{month:02d}{day:02d}.nc"


class ProcessingConfig(BaseModel):
    """Execution controls for masking, resume and resource usage."""

    model_config = ConfigDict(extra="forbid")

    land_sea_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_workers: int = Field(default=1, ge=1, le=128)
    resume: bool = True
    fail_on_dataset_gap: bool = True


class InitialStateConfig(BaseModel):
    """Initial values for the recursive FWI codes."""

    model_config = ConfigDict(extra="forbid")

    ffmc: float = 85.0
    dmc: float = 6.0
    dc: float = 15.0


class AppConfig(BaseModel):
    """Top-level application configuration."""

    model_config = ConfigDict(extra="forbid")

    region: RegionConfig
    period: PeriodConfig
    cds: CDSConfig = Field(default_factory=CDSConfig)
    datasets: DatasetsConfig = Field(default_factory=DatasetsConfig)
    download: DownloadConfig = Field(default_factory=DownloadConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    initial_state: InitialStateConfig = Field(default_factory=InitialStateConfig)

    def resolved(self, base_dir: Path) -> "AppConfig":
        data = self.model_dump(mode="python")

        for section_name in ("paths",):
            section = data[section_name]
            for key, value in section.items():
                if value is None:
                    continue
                section[key] = _resolve_path(base_dir, Path(value))

        return AppConfig.model_validate(data)

    def summary(self) -> str:
        return (
            f"Region: {self.region.country_name}\n"
            f"BBox: {self.region.bbox.north}, {self.region.bbox.west}, "
            f"{self.region.bbox.south}, {self.region.bbox.east}\n"
            f"Period: {self.period.start.isoformat()} -> {self.period.end.isoformat()} "
            f"(spinup {self.period.spinup_days} days)\n"
            f"CDS client: {self.cds.client}\n"
            f"Atmosphere dataset: {self.datasets.atmosphere.collection_id}\n"
            f"Land dataset: {self.datasets.land.collection_id}\n"
            f"Cache dir: {self.paths.cache_dir}\n"
            f"Output dir: {self.paths.output_dir}\n"
            f"Catalog DB: {self.paths.catalog_db}"
        )


def _resolve_path(base_dir: Path, value: Path) -> Path:
    if value.is_absolute():
        return value
    return (base_dir / value).resolve()


def load_config(path: str | Path) -> AppConfig:
    """Load and validate a YAML configuration file."""

    config_path = Path(path).expanduser().resolve()
    try:
        raw_content = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as exc:
        raise ConfigError(f"configuration file not found: {config_path}") from exc
    except OSError as exc:
        raise ConfigError(f"unable to read configuration file: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in configuration file: {config_path}") from exc

    try:
        config = AppConfig.model_validate(raw_content)
    except ValidationError as exc:
        raise ConfigError(str(exc)) from exc

    return config.resolved(config_path.parent)


def dump_example_config() -> dict[str, Any]:
    """Return the default example configuration as a Python dictionary."""

    return deepcopy(
        {
            "region": {
                "bbox": {"north": 41.9, "west": 19.0, "south": 34.5, "east": 29.7},
                "country_name": "Greece",
            },
            "period": {"start": "2023-04-01", "end": "2023-09-30", "spinup_days": 90},
            "cds": {
                "client": "ecmwf_datastores",
                "url_env": "ECMWF_DATASTORES_URL",
                "key_env": "ECMWF_DATASTORES_KEY",
                "rc_file_env": "ECMWF_DATASTORES_RC_FILE",
                "timeout_seconds": 600,
            },
            "datasets": {
                "atmosphere": {
                    "collection_id": "reanalysis-cerra-single-levels",
                    "request_base": {"data_type": ["reanalysis"]},
                    "variable_map": {
                        "temperature": "2m_temperature",
                        "relative_humidity": "2m_relative_humidity",
                        "wind_speed": "10m_wind_speed",
                    },
                    "times": ["12:00"],
                    "data_format": "grib",
                    "download_format": "unarchived",
                },
                "land": {
                    "collection_id": "reanalysis-cerra-land",
                    "request_base": {},
                    "variable_map": {
                        "precipitation": "total_precipitation",
                        "land_sea_mask": "land_sea_mask",
                    },
                    "times": ["06:00"],
                    "data_format": "grib",
                    "download_format": "unarchived",
                },
            },
            "download": {"chunking": "monthly", "retry_attempts": 4, "retry_wait_seconds": 30},
            "paths": {
                "cache_dir": "data/cache",
                "output_dir": "data/output",
                "state_dir": "data/state",
                "catalog_db": "data/state/catalog.sqlite",
            },
            "storage": {
                "format": "netcdf4",
                "compression_level": 4,
                "chunk_shape": [1, 128, 128],
                "include_inputs": True,
                "filename_template": "fwi_{year}{month:02d}.nc",
                "state_template": "state_{year}{month:02d}{day:02d}.nc",
            },
            "processing": {
                "land_sea_threshold": 0.5,
                "max_workers": 1,
                "resume": True,
                "fail_on_dataset_gap": True,
            },
            "initial_state": {"ffmc": 85.0, "dmc": 6.0, "dc": 15.0},
        }
    )