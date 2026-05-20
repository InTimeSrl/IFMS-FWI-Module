"""Helpers for native-grid georeferencing metadata, coordinates and reprojection."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import xarray as xr
from pyproj import CRS, Transformer


DEFAULT_CERRA_LAMBERT_PROJECTION: dict[str, float | int] = {
    "earth_radius": 6_371_229.0,
    "latitude_of_projection_origin": 50.0,
    "standard_parallel_1": 50.0,
    "standard_parallel_2": 50.0,
    "longitude_of_central_meridian": 8.0,
    "latitude_of_first_grid_point": 20.292281,
    "longitude_of_first_grid_point": -17.485943,
    "dx": 5_500.0,
    "dy": 5_500.0,
    "i_scans_negatively": 0,
    "j_scans_positively": 1,
}


def default_cerra_lambert_projection() -> dict[str, float | int]:
    """Return the default Lambert grid definition used by CERRA."""

    return dict(DEFAULT_CERRA_LAMBERT_PROJECTION)


def native_grid_projection_attrs(dataset: xr.Dataset) -> dict[str, float | int] | None:
    """Extract the native Lambert grid definition from GRIB-backed variables."""

    for data_array in dataset.data_vars.values():
        attrs = data_array.attrs
        if attrs.get("GRIB_gridType") != "lambert":
            continue

        earth_radius = earth_radius_from_attrs(attrs)
        if earth_radius is None:
            continue

        required = (
            "GRIB_LaDInDegrees",
            "GRIB_Latin1InDegrees",
            "GRIB_Latin2InDegrees",
            "GRIB_LoVInDegrees",
            "GRIB_latitudeOfFirstGridPointInDegrees",
            "GRIB_longitudeOfFirstGridPointInDegrees",
            "GRIB_DxInMetres",
            "GRIB_DyInMetres",
        )
        if any(key not in attrs for key in required):
            continue

        return {
            "earth_radius": earth_radius,
            "latitude_of_projection_origin": float(attrs["GRIB_LaDInDegrees"]),
            "standard_parallel_1": float(attrs["GRIB_Latin1InDegrees"]),
            "standard_parallel_2": float(attrs["GRIB_Latin2InDegrees"]),
            "longitude_of_central_meridian": wrap_longitude(float(attrs["GRIB_LoVInDegrees"])),
            "latitude_of_first_grid_point": float(attrs["GRIB_latitudeOfFirstGridPointInDegrees"]),
            "longitude_of_first_grid_point": wrap_longitude(float(attrs["GRIB_longitudeOfFirstGridPointInDegrees"])),
            "dx": float(attrs["GRIB_DxInMetres"]),
            "dy": float(attrs["GRIB_DyInMetres"]),
            "i_scans_negatively": int(attrs.get("GRIB_iScansNegatively", 0)),
            "j_scans_positively": int(attrs.get("GRIB_jScansPositively", 0)),
        }
    return None


def native_lambert_crs(projection: Mapping[str, float | int]) -> CRS:
    """Build the native Lambert Conformal Conic CRS from GRIB metadata."""

    return CRS.from_proj4(
        " ".join(
            (
                "+proj=lcc",
                f"+lat_0={projection['latitude_of_projection_origin']}",
                f"+lat_1={projection['standard_parallel_1']}",
                f"+lat_2={projection['standard_parallel_2']}",
                f"+lon_0={projection['longitude_of_central_meridian']}",
                f"+R={projection['earth_radius']}",
                "+units=m",
                "+no_defs",
            )
        )
    )


def projected_axis_coordinates(
    projection: Mapping[str, float | int],
    *,
    x_size: int,
    y_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate native projected x/y coordinates for the regular Lambert grid."""

    transformer = Transformer.from_crs("EPSG:4326", native_lambert_crs(projection), always_xy=True)
    x_origin, y_origin = transformer.transform(
        float(projection["longitude_of_first_grid_point"]),
        float(projection["latitude_of_first_grid_point"]),
    )

    x_origin = _snap_projection_origin(x_origin)
    y_origin = _snap_projection_origin(y_origin)
    x_step = -float(projection["dx"]) if int(projection["i_scans_negatively"]) else float(projection["dx"])
    y_step = float(projection["dy"]) if int(projection["j_scans_positively"]) else -float(projection["dy"])

    x_values = x_origin + x_step * np.arange(x_size, dtype=np.float64)
    y_values = y_origin + y_step * np.arange(y_size, dtype=np.float64)
    return x_values, y_values


def projection_coordinate_attrs(axis: str) -> dict[str, str]:
    if axis == "x":
        return {
            "standard_name": "projection_x_coordinate",
            "long_name": "x coordinate of projection",
            "units": "m",
            "axis": "X",
        }
    return {
        "standard_name": "projection_y_coordinate",
        "long_name": "y coordinate of projection",
        "units": "m",
        "axis": "Y",
    }


def geographic_coordinate_attrs(axis: str) -> dict[str, str]:
    if axis == "x":
        return {
            "standard_name": "longitude",
            "long_name": "longitude",
            "units": "degrees_east",
            "axis": "X",
        }
    return {
        "standard_name": "latitude",
        "long_name": "latitude",
        "units": "degrees_north",
        "axis": "Y",
    }


def has_regular_geographic_grid(dataset: xr.Dataset) -> bool:
    lat = dataset.coords.get("lat")
    lon = dataset.coords.get("lon")
    if lat is not None and lon is not None and lat.ndim == 1 and lon.ndim == 1:
        return True

    x_coord = dataset.coords.get("x")
    y_coord = dataset.coords.get("y")
    if x_coord is None or y_coord is None or x_coord.ndim != 1 or y_coord.ndim != 1:
        return False

    x_units = str(x_coord.attrs.get("units", "")).lower()
    y_units = str(y_coord.attrs.get("units", "")).lower()
    if x_coord.attrs.get("standard_name") == "longitude" and y_coord.attrs.get("standard_name") == "latitude":
        return True
    return "degree" in x_units and "degree" in y_units


def reproject_dataset_to_wgs84(dataset: xr.Dataset) -> xr.Dataset:
    """Reproject a dataset from the native Lambert CERRA grid to a regular EPSG:4326 grid."""

    spatial_dims = spatial_dimensions(dataset)
    if spatial_dims is None:
        return dataset

    if has_regular_geographic_grid(dataset):
        target_lat, target_lon = geographic_axes_from_dataset(dataset)
        return _rebuild_geographic_dataset(dataset, spatial_dims, target_lat, target_lon)

    projection = native_grid_projection_attrs(dataset) or default_cerra_lambert_projection()
    source_x, source_y = projected_axes_for_dataset(dataset, projection)
    target_lat, target_lon = geographic_axes_from_dataset(dataset, projection=projection, source_x=source_x, source_y=source_y)

    source_crs = native_lambert_crs(projection)
    lon_grid, lat_grid = np.meshgrid(target_lon, target_lat)
    transformer = Transformer.from_crs("EPSG:4326", source_crs, always_xy=True)
    target_x, target_y = transformer.transform(lon_grid, lat_grid)

    coords = _build_geographic_coords(dataset, spatial_dims, target_lat, target_lon)
    data_vars: dict[str, xr.DataArray] = {}
    for name, data_array in dataset.data_vars.items():
        attrs = _clean_reprojected_attrs(data_array.attrs)
        if set(spatial_dims).issubset(data_array.dims):
            values = reproject_data_array(
                data_array,
                spatial_dims=spatial_dims,
                source_x=source_x,
                source_y=source_y,
                target_x=target_x,
                target_y=target_y,
            )
            data_vars[name] = xr.DataArray(values, dims=data_array.dims, coords=_coords_for_dims(coords, data_array.dims), attrs=attrs)
        else:
            data_vars[name] = xr.DataArray(
                np.asarray(data_array.values),
                dims=data_array.dims,
                coords=_coords_for_dims(coords, data_array.dims),
                attrs=attrs,
            )

    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=dict(dataset.attrs))


def reproject_data_array(
    data_array: xr.DataArray,
    *,
    spatial_dims: tuple[str, str],
    source_x: np.ndarray,
    source_y: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
) -> np.ndarray:
    """Resample a data array from a regular projected grid to the target grid."""

    spatial_axes = tuple(data_array.dims.index(dim) for dim in spatial_dims)
    values = np.asarray(data_array.values)
    moved = np.moveaxis(values, spatial_axes, (-2, -1))
    method = "nearest" if data_array.dtype.kind in {"b", "i", "u"} else "linear"
    reprojected = _resample_regular_grid(moved, source_x, source_y, target_x, target_y, method=method)

    if method == "nearest":
        fill_value = 0 if data_array.dtype.kind in {"b", "i", "u"} else np.nan
        reprojected = np.nan_to_num(reprojected, nan=fill_value)
        reprojected = reprojected.astype(data_array.dtype, copy=False)
    else:
        reprojected = reprojected.astype(data_array.dtype, copy=False)

    return np.moveaxis(reprojected, (-2, -1), spatial_axes)


def projected_axes_for_dataset(
    dataset: xr.Dataset,
    projection: Mapping[str, float | int],
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve the native projected x/y axes for the source grid."""

    x_coord = dataset.coords.get("x")
    y_coord = dataset.coords.get("y")
    if _looks_like_projected_axis(x_coord, "x") and _looks_like_projected_axis(y_coord, "y"):
        return np.asarray(x_coord.values, dtype=np.float64), np.asarray(y_coord.values, dtype=np.float64)

    lon, lat = extract_lon_lat_coordinates(dataset)
    transformer = Transformer.from_crs("EPSG:4326", native_lambert_crs(projection), always_xy=True)
    projected_x, projected_y = transformer.transform(np.asarray(lon.values, dtype=np.float64), np.asarray(lat.values, dtype=np.float64))
    return _collapse_projected_axis(projected_x, axis="x"), _collapse_projected_axis(projected_y, axis="y")


def geographic_axes_from_dataset(
    dataset: xr.Dataset,
    *,
    projection: Mapping[str, float | int] | None = None,
    source_x: np.ndarray | None = None,
    source_y: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the regular geographic target axes for an exported dataset."""

    lat_coord = dataset.coords.get("lat")
    lon_coord = dataset.coords.get("lon")
    if lat_coord is not None and lon_coord is not None and lat_coord.ndim == 1 and lon_coord.ndim == 1:
        return np.asarray(lat_coord.values, dtype=np.float64), np.asarray(lon_coord.values, dtype=np.float64)

    if has_regular_geographic_grid(dataset):
        x_coord = dataset.coords.get("x")
        y_coord = dataset.coords.get("y")
        if x_coord is not None and y_coord is not None:
            return np.asarray(y_coord.values, dtype=np.float64), np.asarray(x_coord.values, dtype=np.float64)

    if lat_coord is not None and lon_coord is not None:
        lon_values, lat_values = extract_lon_lat_coordinates(dataset)
        lon_array = np.asarray(lon_values.values, dtype=np.float64)
        lat_array = np.asarray(lat_values.values, dtype=np.float64)
    else:
        if projection is None:
            projection = native_grid_projection_attrs(dataset) or default_cerra_lambert_projection()
        if source_x is None or source_y is None:
            source_x, source_y = projected_axes_for_dataset(dataset, projection)
        x_grid, y_grid = np.meshgrid(source_x, source_y)
        transformer = Transformer.from_crs(native_lambert_crs(projection), "EPSG:4326", always_xy=True)
        lon_array, lat_array = transformer.transform(x_grid, y_grid)

    spatial_dims = spatial_dimensions(dataset)
    if spatial_dims is None:
        raise ValueError("dataset is missing a two-dimensional spatial grid")

    y_size = dataset.sizes[spatial_dims[0]]
    x_size = dataset.sizes[spatial_dims[1]]
    target_lat = _regular_axis_from_values(lat_array, axis="y", size=y_size)
    target_lon = _regular_axis_from_values(lon_array, axis="x", size=x_size)
    return target_lat, target_lon


def extract_lon_lat_coordinates(dataset: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
    lon = _find_coord_or_var(dataset, ("lon", "longitude"))
    lat = _find_coord_or_var(dataset, ("lat", "latitude"))

    if lon.ndim == 1 and lat.ndim == 1:
        lon, lat = xr.broadcast(lon, lat)
    elif lon.dims != lat.dims:
        lon, lat = xr.broadcast(lon, lat)
    return lon, lat


def spatial_dimensions(dataset: xr.Dataset) -> tuple[str, str] | None:
    if {"y", "x"}.issubset(dataset.dims):
        return ("y", "x")

    lat = dataset.coords.get("lat")
    lon = dataset.coords.get("lon")
    if lat is not None and lon is not None and lat.ndim == 2 and lon.ndim == 2:
        return (lat.dims[0], lon.dims[1])
    if lat is not None and lon is not None and lat.ndim == 1 and lon.ndim == 1:
        return (lat.dims[0], lon.dims[0])
    return None


def earth_radius_from_attrs(attrs: Mapping[str, object]) -> float | None:
    if "GRIB_radius" in attrs:
        return float(attrs["GRIB_radius"])
    shape_of_earth = attrs.get("GRIB_shapeOfTheEarth")
    if shape_of_earth is not None and int(shape_of_earth) == 6:
        return 6_371_229.0
    return None


def wrap_longitude(value: float) -> float:
    return value - 360.0 if value > 180.0 else value


def _snap_projection_origin(value: float) -> float:
    rounded = round(value)
    return float(rounded) if abs(value - rounded) <= 1.0 else float(value)


def _find_coord_or_var(dataset: xr.Dataset, names: tuple[str, ...]) -> xr.DataArray:
    for name in names:
        if name in dataset.coords:
            value = dataset.coords[name]
            return value.isel(time=0, drop=True) if "time" in value.dims else value
        if name in dataset.data_vars:
            value = dataset[name]
            return value.isel(time=0, drop=True) if "time" in value.dims else value
    raise ValueError(f"dataset is missing coordinate(s): {', '.join(names)}")


def _looks_like_projected_axis(coord: xr.DataArray | None, axis: str) -> bool:
    if coord is None or coord.ndim != 1:
        return False

    values = np.asarray(coord.values, dtype=np.float64)
    if values.size == 0:
        return False

    standard_name = coord.attrs.get("standard_name")
    units = str(coord.attrs.get("units", "")).lower()
    if standard_name == f"projection_{axis}_coordinate" or units == "m":
        return True
    return bool(np.nanmax(np.abs(values)) >= 1_000.0)


def _collapse_projected_axis(values: np.ndarray, *, axis: str) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError("projected grid coordinates must be two-dimensional")
    collapsed = np.nanmean(values, axis=0 if axis == "x" else 1)
    return collapsed.astype(np.float64, copy=False)


def _regular_axis_from_values(values: np.ndarray, *, axis: str, size: int) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("cannot infer a geographic axis from empty coordinates")

    minimum = float(finite.min())
    maximum = float(finite.max())
    if axis == "x":
        starts_at = float(values[0, 0])
        ends_at = float(values[0, -1])
        return np.linspace(minimum, maximum, size, dtype=np.float64) if ends_at >= starts_at else np.linspace(maximum, minimum, size, dtype=np.float64)

    starts_at = float(values[0, 0])
    ends_at = float(values[-1, 0])
    return np.linspace(minimum, maximum, size, dtype=np.float64) if ends_at >= starts_at else np.linspace(maximum, minimum, size, dtype=np.float64)


def _build_geographic_coords(
    dataset: xr.Dataset,
    spatial_dims: tuple[str, str],
    target_lat: np.ndarray,
    target_lon: np.ndarray,
) -> dict[str, xr.DataArray]:
    coords: dict[str, xr.DataArray] = {
        name: coord.copy(deep=True)
        for name, coord in dataset.coords.items()
        if name not in {"x", "y", "lat", "lon"} and not set(coord.dims).intersection(spatial_dims)
    }
    coords[spatial_dims[0]] = xr.DataArray(target_lat, dims=(spatial_dims[0],), attrs=geographic_coordinate_attrs("y"))
    coords[spatial_dims[1]] = xr.DataArray(target_lon, dims=(spatial_dims[1],), attrs=geographic_coordinate_attrs("x"))
    coords["lat"] = xr.DataArray(target_lat, dims=(spatial_dims[0],), attrs=geographic_coordinate_attrs("y"))
    coords["lon"] = xr.DataArray(target_lon, dims=(spatial_dims[1],), attrs=geographic_coordinate_attrs("x"))
    return coords


def _rebuild_geographic_dataset(
    dataset: xr.Dataset,
    spatial_dims: tuple[str, str],
    target_lat: np.ndarray,
    target_lon: np.ndarray,
) -> xr.Dataset:
    coords = _build_geographic_coords(dataset, spatial_dims, target_lat, target_lon)
    data_vars: dict[str, xr.DataArray] = {}
    for name, data_array in dataset.data_vars.items():
        data_vars[name] = xr.DataArray(
            np.asarray(data_array.values),
            dims=data_array.dims,
            coords=_coords_for_dims(coords, data_array.dims),
            attrs=_clean_reprojected_attrs(data_array.attrs),
        )
    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=dict(dataset.attrs))


def _coords_for_dims(coords: Mapping[str, xr.DataArray], dims: tuple[str, ...]) -> dict[str, xr.DataArray]:
    return {dim: coords[dim] for dim in dims if dim in coords}


def _clean_reprojected_attrs(attrs: Mapping[str, object]) -> dict[str, object]:
    cleaned = {
        key: value
        for key, value in attrs.items()
        if not key.startswith("GRIB_") and key not in {"coordinates", "grid_mapping"}
    }
    return dict(cleaned)


def _resample_regular_grid(
    values: np.ndarray,
    source_x: np.ndarray,
    source_y: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
    *,
    method: str,
) -> np.ndarray:
    source_x = np.asarray(source_x, dtype=np.float64)
    source_y = np.asarray(source_y, dtype=np.float64)
    reshaped = np.asarray(values, dtype=np.float64)
    source_x, reshaped = _ensure_increasing_axis(source_x, reshaped, axis=-1)
    source_y, reshaped = _ensure_increasing_axis(source_y, reshaped, axis=-2)

    ny = source_y.size
    nx = source_x.size
    leading_shape = reshaped.shape[:-2]
    flat = reshaped.reshape((-1, ny, nx))
    flat_x = target_x.reshape(-1)
    flat_y = target_y.reshape(-1)

    x_index = np.interp(flat_x, source_x, np.arange(nx, dtype=np.float64), left=np.nan, right=np.nan)
    y_index = np.interp(flat_y, source_y, np.arange(ny, dtype=np.float64), left=np.nan, right=np.nan)
    invalid = np.isnan(x_index) | np.isnan(y_index)
    safe_x_index = np.where(invalid, 0.0, x_index)
    safe_y_index = np.where(invalid, 0.0, y_index)

    if method == "nearest":
        x_nearest = np.clip(np.rint(safe_x_index), 0, nx - 1).astype(np.int64, copy=False)
        y_nearest = np.clip(np.rint(safe_y_index), 0, ny - 1).astype(np.int64, copy=False)
        result = flat[:, y_nearest, x_nearest].astype(np.float64, copy=False)
        result[:, invalid] = np.nan
    else:
        x0 = np.clip(np.floor(safe_x_index), 0, nx - 1).astype(np.int64, copy=False)
        y0 = np.clip(np.floor(safe_y_index), 0, ny - 1).astype(np.int64, copy=False)
        x1 = np.clip(x0 + 1, 0, nx - 1)
        y1 = np.clip(y0 + 1, 0, ny - 1)
        wx = safe_x_index - x0
        wy = safe_y_index - y0

        v00 = flat[:, y0, x0]
        v10 = flat[:, y0, x1]
        v01 = flat[:, y1, x0]
        v11 = flat[:, y1, x1]

        weights = (
            ((1.0 - wx) * (1.0 - wy), v00),
            (wx * (1.0 - wy), v10),
            ((1.0 - wx) * wy, v01),
            (wx * wy, v11),
        )
        result = np.zeros_like(v00, dtype=np.float64)
        weight_sum = np.zeros_like(v00, dtype=np.float64)
        for weight, corner in weights:
            broadcast_weight = weight[None, :]
            valid = np.isfinite(corner)
            result += np.where(valid, corner * broadcast_weight, 0.0)
            weight_sum += np.where(valid, broadcast_weight, 0.0)

        with np.errstate(invalid="ignore", divide="ignore"):
            result = np.divide(result, weight_sum, out=np.full_like(result, np.nan), where=weight_sum > 0.0)
        result[:, invalid] = np.nan

    return result.reshape((*leading_shape, *target_x.shape))


def _ensure_increasing_axis(axis_values: np.ndarray, values: np.ndarray, *, axis: int) -> tuple[np.ndarray, np.ndarray]:
    if axis_values.size <= 1 or axis_values[0] <= axis_values[-1]:
        return axis_values, values
    return axis_values[::-1].copy(), np.flip(values, axis=axis)