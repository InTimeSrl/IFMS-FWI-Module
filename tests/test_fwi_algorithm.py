from __future__ import annotations

import numpy as np
import xarray as xr

from fwi_module.fwi_algorithm import FWIState, compute_fwi_indices


def test_compute_fwi_indices_produces_expected_variables() -> None:
    dataset = _sample_inputs()

    result = compute_fwi_indices(dataset)

    assert set(result.dataset.data_vars) == {"ffmc", "dmc", "dc", "isi", "bui", "fwi", "dsr"}
    assert result.dataset.sizes["time"] == 3
    assert result.dataset["fwi"].shape == (3, 2, 2)
    assert np.isfinite(result.dataset["fwi"].values).all()


def test_resume_state_matches_full_run() -> None:
    dataset = _sample_inputs()

    full_result = compute_fwi_indices(dataset)
    first_window = compute_fwi_indices(dataset.isel(time=slice(0, 2)))
    resumed_result = compute_fwi_indices(dataset.isel(time=slice(2, 3)), initial_state=first_window.state)

    np.testing.assert_allclose(full_result.dataset["ffmc"].isel(time=2).values, resumed_result.dataset["ffmc"].isel(time=0).values)
    np.testing.assert_allclose(full_result.dataset["dmc"].isel(time=2).values, resumed_result.dataset["dmc"].isel(time=0).values)
    np.testing.assert_allclose(full_result.dataset["dc"].isel(time=2).values, resumed_result.dataset["dc"].isel(time=0).values)
    np.testing.assert_allclose(full_result.dataset["fwi"].isel(time=2).values, resumed_result.dataset["fwi"].isel(time=0).values)


def test_rainfall_reduces_fire_danger_signal() -> None:
    dry = _single_cell_inputs(temperature=[28.0, 29.0], relative_humidity=[25.0, 20.0], wind_speed=[20.0, 25.0], precipitation=[0.0, 0.0])
    wet = _single_cell_inputs(temperature=[28.0, 29.0], relative_humidity=[25.0, 20.0], wind_speed=[20.0, 25.0], precipitation=[0.0, 15.0])

    dry_result = compute_fwi_indices(dry)
    wet_result = compute_fwi_indices(wet)

    assert wet_result.dataset["ffmc"].isel(time=1).item() < dry_result.dataset["ffmc"].isel(time=1).item()
    assert wet_result.dataset["fwi"].isel(time=1).item() < dry_result.dataset["fwi"].isel(time=1).item()


def _sample_inputs() -> xr.Dataset:
    times = np.array(["2023-04-01", "2023-04-02", "2023-04-03"], dtype="datetime64[ns]")
    lat = xr.DataArray(np.array([[39.0, 39.1], [38.9, 39.0]]), dims=("y", "x"))
    coords = {"time": times, "y": [0, 1], "x": [0, 1], "lat": lat}

    return xr.Dataset(
        data_vars={
            "temperature": (("time", "y", "x"), np.array([[[17.0, 18.0], [19.0, 20.0]], [[21.0, 22.0], [23.0, 24.0]], [[18.0, 19.0], [20.0, 21.0]]])),
            "relative_humidity": (("time", "y", "x"), np.array([[[42.0, 40.0], [38.0, 36.0]], [[30.0, 28.0], [26.0, 24.0]], [[35.0, 34.0], [33.0, 32.0]]])),
            "wind_speed": (("time", "y", "x"), np.array([[[25.0, 22.0], [18.0, 17.0]], [[20.0, 18.0], [16.0, 15.0]], [[24.0, 22.0], [20.0, 19.0]]])),
            "precipitation": (("time", "y", "x"), np.array([[[0.0, 0.0], [0.0, 0.0]], [[2.4, 2.0], [1.8, 1.5]], [[0.0, 0.0], [0.0, 0.0]]])),
        },
        coords=coords,
    )


def _single_cell_inputs(*, temperature: list[float], relative_humidity: list[float], wind_speed: list[float], precipitation: list[float]) -> xr.Dataset:
    times = np.array(["2023-07-01", "2023-07-02"], dtype="datetime64[ns]")
    lat = xr.DataArray(np.array([[39.0]]), dims=("y", "x"))
    return xr.Dataset(
        data_vars={
            "temperature": (("time", "y", "x"), np.array(temperature, dtype=float).reshape(2, 1, 1)),
            "relative_humidity": (("time", "y", "x"), np.array(relative_humidity, dtype=float).reshape(2, 1, 1)),
            "wind_speed": (("time", "y", "x"), np.array(wind_speed, dtype=float).reshape(2, 1, 1)),
            "precipitation": (("time", "y", "x"), np.array(precipitation, dtype=float).reshape(2, 1, 1)),
        },
        coords={"time": times, "y": [0], "x": [0], "lat": lat},
    )