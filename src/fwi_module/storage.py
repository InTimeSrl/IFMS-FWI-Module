"""NetCDF storage helpers for outputs and checkpoints."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import xarray as xr

from .config import StorageConfig
from .utils import ensure_directory


def write_netcdf_atomic(dataset: xr.Dataset, target_path: Path, storage: StorageConfig) -> Path:
    """Write a dataset atomically to a NetCDF4 file."""

    ensure_directory(target_path.parent)
    encoding = build_encoding(dataset, storage)
    file_descriptor, temp_name = tempfile.mkstemp(suffix=target_path.suffix or ".nc", dir=target_path.parent)
    os.close(file_descriptor)
    temp_path = Path(temp_name)
    try:
        dataset.to_netcdf(temp_path, engine="netcdf4", format="NETCDF4", encoding=encoding)
        os.replace(temp_path, target_path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
    return target_path


def build_encoding(dataset: xr.Dataset, storage: StorageConfig) -> dict[str, dict[str, object]]:
    """Build per-variable NetCDF encoding settings."""

    encoding: dict[str, dict[str, object]] = {}
    for name, data_array in dataset.data_vars.items():
        variable_encoding: dict[str, object] = {
            "dtype": "float32" if data_array.dtype.kind == "f" else data_array.dtype,
            "zlib": storage.compression_level > 0,
            "complevel": storage.compression_level,
            "shuffle": True,
        }
        if len(data_array.dims) >= 3 and "time" in data_array.dims:
            variable_encoding["chunksizes"] = tuple(
                min(size, chunk) for size, chunk in zip(data_array.shape, storage.chunk_shape, strict=False)
            )
        elif len(data_array.dims) >= 2:
            spatial_chunks = storage.chunk_shape[-len(data_array.dims) :]
            variable_encoding["chunksizes"] = tuple(min(size, chunk) for size, chunk in zip(data_array.shape, spatial_chunks, strict=False))
        encoding[name] = variable_encoding
    return encoding