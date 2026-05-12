"""Open, normalize and merge downloaded meteorological data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

from .config import AppConfig, BoundingBox, DatasetRequestConfig
from .exceptions import ProcessingError
from .georeferencing import native_grid_projection_attrs, projected_axis_coordinates, projection_coordinate_attrs
from .masking import apply_spatial_mask


VARIABLE_NAME_GROUPS: tuple[tuple[str, ...], ...] = (
    ("2m_temperature", "t2m"),
    ("2m_relative_humidity", "r2"),
    ("10m_wind_speed", "si10"),
    ("total_precipitation", "tp"),
    ("land_sea_mask", "lsm"),
)


@dataclass(slots=True)
class PreparedInputs:
    dataset: xr.Dataset
    land_mask: xr.DataArray


def prepare_fwi_inputs(atmosphere_path: Path, land_path: Path, config: AppConfig) -> PreparedInputs:
    """Load raw files and turn them into daily FWI inputs."""

    atmosphere_raw = open_dataset_file(atmosphere_path)
    land_raw = open_dataset_file(land_path)
    try:
        atmosphere = extract_atmosphere_inputs(atmosphere_raw, config.datasets.atmosphere)
        land = extract_land_inputs(land_raw, config.datasets.land)
    finally:
        atmosphere_raw.close()
        land_raw.close()

    merged = xr.merge([atmosphere, land], compat="override", join="inner")
    merged = clip_to_bbox(merged, config.region.bbox)
    land_sea_mask = merged["land_sea_mask"]
    masked = apply_spatial_mask(
        merged.drop_vars("land_sea_mask"),
        land_sea_mask,
        config.region.country_name,
        config.processing.land_sea_threshold,
        config.processing.coastal_buffer_cells,
    )
    return PreparedInputs(dataset=masked, land_mask=masked["mask"])


def clip_to_bbox(dataset: xr.Dataset, bbox: BoundingBox) -> xr.Dataset:
    """Crop the dataset to the smallest grid rectangle intersecting the configured bbox."""

    lon, lat = _extract_lon_lat(dataset)
    bbox_mask = (lat >= bbox.south) & (lat <= bbox.north) & (lon >= bbox.west) & (lon <= bbox.east)

    if not bool(bbox_mask.any().item()):
        raise ProcessingError("configured bbox does not intersect the downloaded dataset domain")

    if bbox_mask.ndim >= 2:
        row_dim, col_dim = bbox_mask.dims[-2], bbox_mask.dims[-1]
        valid_rows = bbox_mask.any(dim=col_dim)
        valid_cols = bbox_mask.any(dim=row_dim)
        dataset = dataset.isel({row_dim: valid_rows, col_dim: valid_cols})
        lon, lat = _extract_lon_lat(dataset)
        bbox_mask = (lat >= bbox.south) & (lat <= bbox.north) & (lon >= bbox.west) & (lon <= bbox.east)

    return dataset.where(bbox_mask)


def open_dataset_file(path: Path) -> xr.Dataset:
    """Open NetCDF or GRIB files with xarray."""

    suffix = path.suffix.lower()
    if suffix in {".nc", ".nc4"}:
        dataset = xr.open_dataset(path)
    elif suffix in {".grib", ".grib2", ".grb"}:
        try:
            import cfgrib
        except ImportError as exc:
            raise ProcessingError(
                "GRIB support requires the optional dependencies 'cfgrib' and 'eccodes'. Install them with: uv sync --extra grib"
            ) from exc
        with xr.set_options(use_new_combine_kwarg_defaults=True):
            groups = cfgrib.open_datasets(path, backend_kwargs={"indexpath": "", "read_keys": ["radius", "shapeOfTheEarth"]})
        prepared_groups = [_standardize_dataset(_drop_auxiliary_grib_coords(group)) for group in groups]
        dataset = xr.merge(prepared_groups, compat="override", join="outer", combine_attrs="drop_conflicts")
    else:
        raise ProcessingError(f"unsupported data file format: {path}")

    return _assign_native_projection_coordinates(_standardize_dataset(dataset))


def extract_atmosphere_inputs(dataset: xr.Dataset, request: DatasetRequestConfig) -> xr.Dataset:
    variable_map = _resolve_variable_map(dataset, request.variable_map)
    subset = dataset[list(dict.fromkeys(variable_map.values()))]
    subset = _select_requested_times(subset, request.times)
    subset = subset.rename({variable_map["temperature"]: "temperature", variable_map["relative_humidity"]: "relative_humidity", variable_map["wind_speed"]: "wind_speed"})
    subset["temperature"] = _convert_temperature(subset["temperature"])
    subset["relative_humidity"] = subset["relative_humidity"].clip(min=0.0, max=100.0)
    subset["wind_speed"] = _convert_wind_speed(subset["wind_speed"])
    return subset[["temperature", "relative_humidity", "wind_speed"]].load()


def extract_land_inputs(dataset: xr.Dataset, request: DatasetRequestConfig) -> xr.Dataset:
    variable_map = _resolve_variable_map(dataset, request.variable_map)
    subset = dataset[list(dict.fromkeys(variable_map.values()))]
    subset = _select_requested_times(subset, request.times)
    renamed = subset.rename({variable_map["precipitation"]: "precipitation", variable_map["land_sea_mask"]: "land_sea_mask"})
    renamed["precipitation"] = _convert_precipitation(renamed["precipitation"])
    return renamed[["precipitation", "land_sea_mask"]].load()


def _resolve_variable_map(dataset: xr.Dataset, variable_map: dict[str, str]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    missing: list[str] = []

    for logical_name, source_name in variable_map.items():
        actual_name = next((candidate for candidate in _candidate_variable_names(source_name) if candidate in dataset.data_vars), None)
        if actual_name is None:
            missing.append(source_name)
            continue
        resolved[logical_name] = actual_name

    if missing:
        raise ProcessingError(f"dataset is missing variables: {', '.join(sorted(missing))}")
    return resolved


def _candidate_variable_names(source_name: str) -> tuple[str, ...]:
    for group in VARIABLE_NAME_GROUPS:
        if source_name in group:
            return group
    return (source_name,)


def _select_requested_times(dataset: xr.Dataset, requested_times: list[str]) -> xr.Dataset:
    if not requested_times or "time" not in dataset.coords:
        return dataset

    timestamps = dataset["time"].dt.strftime("%H:%M")
    selector = xr.DataArray(np.isin(timestamps, requested_times), coords={"time": dataset["time"]}, dims=("time",))
    selected = dataset.where(selector, drop=True)
    if selected.sizes.get("time", 0) == 0:
        joined = ", ".join(requested_times)
        raise ProcessingError(f"requested daily times not found in dataset: {joined}")

    normalized_time = selected["time"].values.astype("datetime64[D]").astype("datetime64[ns]")
    selected = selected.assign_coords(time=normalized_time)
    _, unique_indices = np.unique(normalized_time, return_index=True)
    return selected.isel(time=np.sort(unique_indices))


def _standardize_dataset(dataset: xr.Dataset) -> xr.Dataset:
    rename_map: dict[str, str] = {}
    if "latitude" in dataset.coords and "lat" not in dataset.coords:
        rename_map["latitude"] = "lat"
    if "longitude" in dataset.coords and "lon" not in dataset.coords:
        rename_map["longitude"] = "lon"
    if rename_map:
        dataset = dataset.rename(rename_map)

    time_values = _extract_time_values(dataset)
    dataset = _drop_scalar_coords(dataset, {"time", "valid_time", "step"})
    if time_values is not None:
        dataset = _assign_time_coordinate(dataset, time_values)
        return dataset.sortby("time")
    return dataset


def _drop_auxiliary_grib_coords(dataset: xr.Dataset) -> xr.Dataset:
    keep = {"time", "valid_time", "latitude", "longitude", "lat", "lon"}
    to_drop = [name for name in dataset.coords if name not in keep and name not in dataset.dims]
    return dataset.drop_vars(to_drop) if to_drop else dataset


def _extract_time_values(dataset: xr.Dataset) -> np.ndarray | None:
    for coord_name in ("valid_time", "time"):
        if coord_name in dataset.coords:
            values = np.asarray(dataset.coords[coord_name].values, dtype="datetime64[ns]")
            return values.reshape(1) if values.ndim == 0 else values.reshape(-1)
    return None


def _drop_scalar_coords(dataset: xr.Dataset, coord_names: set[str]) -> xr.Dataset:
    to_drop = [name for name in coord_names if name in dataset.coords and name not in dataset.dims]
    return dataset.drop_vars(to_drop) if to_drop else dataset


def _assign_time_coordinate(dataset: xr.Dataset, time_values: np.ndarray) -> xr.Dataset:
    if "time" in dataset.dims:
        return dataset.assign_coords(time=("time", time_values))

    expanded = {name: data_array.expand_dims(time=time_values) for name, data_array in dataset.data_vars.items()}
    coords = {name: coord for name, coord in dataset.coords.items()}
    coords["time"] = xr.DataArray(time_values, dims=("time",))
    return xr.Dataset(data_vars=expanded, coords=coords, attrs=dataset.attrs)


def _assign_native_projection_coordinates(dataset: xr.Dataset) -> xr.Dataset:
    if not {"x", "y"}.issubset(dataset.dims):
        return dataset
    if "x" in dataset.coords and "y" in dataset.coords:
        return dataset

    projection = native_grid_projection_attrs(dataset)
    if projection is None:
        return dataset

    x_values, y_values = projected_axis_coordinates(projection, x_size=dataset.sizes["x"], y_size=dataset.sizes["y"])
    return dataset.assign_coords(
        x=xr.DataArray(x_values, dims=("x",), attrs=projection_coordinate_attrs("x")),
        y=xr.DataArray(y_values, dims=("y",), attrs=projection_coordinate_attrs("y")),
    )


def _extract_lon_lat(dataset: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
    lon = _find_coord_or_var(dataset, ("lon", "longitude"))
    lat = _find_coord_or_var(dataset, ("lat", "latitude"))

    if lon.ndim == 1 and lat.ndim == 1:
        lon, lat = xr.broadcast(lon, lat)
    elif lon.dims != lat.dims:
        lon, lat = xr.broadcast(lon, lat)
    return lon, lat


def _find_coord_or_var(dataset: xr.Dataset, names: tuple[str, ...]) -> xr.DataArray:
    for name in names:
        if name in dataset.coords:
            value = dataset.coords[name]
            return value.isel(time=0, drop=True) if "time" in value.dims else value
        if name in dataset.data_vars:
            value = dataset[name]
            return value.isel(time=0, drop=True) if "time" in value.dims else value
    raise ProcessingError(f"dataset is missing coordinate(s): {', '.join(names)}")


def _convert_temperature(data_array: xr.DataArray) -> xr.DataArray:
    units = str(data_array.attrs.get("units", "")).lower()
    if units in {"k", "kelvin"} or float(data_array.max()) > 100.0:
        converted = data_array - 273.15
        converted.attrs["units"] = "degC"
        return converted
    return data_array


def _convert_wind_speed(data_array: xr.DataArray) -> xr.DataArray:
    units = str(data_array.attrs.get("units", "")).lower()
    if "m s" in units or "m/s" in units:
        converted = data_array * 3.6
        converted.attrs["units"] = "km h-1"
        return converted
    return data_array


def _convert_precipitation(data_array: xr.DataArray) -> xr.DataArray:
    units = str(data_array.attrs.get("units", "")).lower()
    if units in {"m", "meter", "metre"}:
        converted = data_array * 1000.0
        converted.attrs["units"] = "mm"
        return converted
    converted = data_array.copy()
    converted.attrs["units"] = "mm"
    return converted