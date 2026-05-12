from __future__ import annotations

import numpy as np
import xarray as xr

from fwi_module.config import AppConfig, dump_example_config
from fwi_module.preprocess import (
    _assign_native_projection_coordinates,
    _drop_auxiliary_grib_coords,
    _standardize_dataset,
    extract_atmosphere_inputs,
    extract_land_inputs,
)


def test_standardize_dataset_promotes_scalar_valid_time_to_time_dimension() -> None:
    dataset = xr.Dataset(
        data_vars={"t2m": (("y", "x"), np.full((2, 2), 300.0, dtype=float))},
        coords={
            "y": [0, 1],
            "x": [0, 1],
            "latitude": (("y", "x"), np.array([[39.0, 39.1], [38.9, 39.0]], dtype=float)),
            "longitude": (("y", "x"), np.array([[22.0, 22.1], [22.1, 22.2]], dtype=float)),
            "time": np.datetime64("2023-04-01T00:00:00"),
            "valid_time": np.datetime64("2023-04-01T12:00:00"),
            "step": np.timedelta64(12, "h"),
            "heightAboveGround": 2.0,
        },
    )

    standardized = _standardize_dataset(_drop_auxiliary_grib_coords(dataset))

    assert standardized.sizes["time"] == 1
    assert standardized["t2m"].dims == ("time", "y", "x")
    assert standardized.time.values[0] == np.datetime64("2023-04-01T12:00:00")
    assert "lat" in standardized.coords
    assert "lon" in standardized.coords
    assert "heightAboveGround" not in standardized.coords


def test_extract_atmosphere_inputs_accepts_grib_short_names() -> None:
    config = AppConfig.model_validate(dump_example_config())
    dataset = xr.Dataset(
        data_vars={
            "t2m": (("time", "y", "x"), np.full((1, 2, 2), 296.15, dtype=float)),
            "r2": (("time", "y", "x"), np.full((1, 2, 2), 35.0, dtype=float)),
            "si10": (("time", "y", "x"), np.full((1, 2, 2), 5.0, dtype=float)),
        },
        coords={
            "time": [np.datetime64("2023-04-01T12:00:00")],
            "y": [0, 1],
            "x": [0, 1],
            "lat": (("y", "x"), np.array([[39.0, 39.1], [38.9, 39.0]], dtype=float)),
            "lon": (("y", "x"), np.array([[22.0, 22.1], [22.1, 22.2]], dtype=float)),
        },
    )
    dataset["t2m"].attrs["units"] = "K"
    dataset["si10"].attrs["units"] = "m s-1"

    prepared = extract_atmosphere_inputs(dataset, config.datasets.atmosphere)

    assert list(prepared.data_vars) == ["temperature", "relative_humidity", "wind_speed"]
    np.testing.assert_allclose(prepared["temperature"].values, 23.0)
    np.testing.assert_allclose(prepared["relative_humidity"].values, 35.0)
    np.testing.assert_allclose(prepared["wind_speed"].values, 18.0)


def test_extract_land_inputs_accepts_grib_short_names() -> None:
    config = AppConfig.model_validate(dump_example_config())
    dataset = xr.Dataset(
        data_vars={
            "tp": (("time", "y", "x"), np.full((1, 2, 2), 0.001, dtype=float)),
            "lsm": (("time", "y", "x"), np.ones((1, 2, 2), dtype=float)),
        },
        coords={
            "time": [np.datetime64("2023-04-01T06:00:00")],
            "y": [0, 1],
            "x": [0, 1],
            "lat": (("y", "x"), np.array([[39.0, 39.1], [38.9, 39.0]], dtype=float)),
            "lon": (("y", "x"), np.array([[22.0, 22.1], [22.1, 22.2]], dtype=float)),
        },
    )
    dataset["tp"].attrs["units"] = "m"

    prepared = extract_land_inputs(dataset, config.datasets.land)

    assert list(prepared.data_vars) == ["precipitation", "land_sea_mask"]
    np.testing.assert_allclose(prepared["precipitation"].values, 1.0)
    np.testing.assert_allclose(prepared["land_sea_mask"].values, 1.0)


def test_assign_native_projection_coordinates_uses_cerra_lambert_grid() -> None:
    dataset = xr.Dataset(
        data_vars={"t2m": (("y", "x"), np.zeros((2, 3), dtype=float))},
        coords={
            "lat": (("y", "x"), np.array([[20.292281, 20.292281, 20.292281], [20.3418, 20.3418, 20.3418]], dtype=float)),
            "lon": (("y", "x"), np.array([[-17.485943, -17.4267, -17.3674], [-17.4886, -17.4294, -17.3701]], dtype=float)),
        },
    )
    dataset["t2m"].attrs.update(
        {
            "GRIB_gridType": "lambert",
            "GRIB_radius": 6_371_229,
            "GRIB_LaDInDegrees": 50.0,
            "GRIB_Latin1InDegrees": 50.0,
            "GRIB_Latin2InDegrees": 50.0,
            "GRIB_LoVInDegrees": 8.0,
            "GRIB_latitudeOfFirstGridPointInDegrees": 20.292281,
            "GRIB_longitudeOfFirstGridPointInDegrees": 342.514057,
            "GRIB_DxInMetres": 5_500.0,
            "GRIB_DyInMetres": 5_500.0,
            "GRIB_iScansNegatively": 0,
            "GRIB_jScansPositively": 1,
        }
    )

    projected = _assign_native_projection_coordinates(dataset)

    np.testing.assert_allclose(projected.coords["x"].values, [-2_937_000.0, -2_931_500.0, -2_926_000.0])
    np.testing.assert_allclose(projected.coords["y"].values, [-2_937_000.0, -2_931_500.0])
    assert projected.coords["x"].attrs["standard_name"] == "projection_x_coordinate"
    assert projected.coords["y"].attrs["standard_name"] == "projection_y_coordinate"