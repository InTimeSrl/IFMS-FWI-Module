"""Helpers for native-grid georeferencing metadata and coordinates."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import xarray as xr
from pyproj import CRS, Transformer


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