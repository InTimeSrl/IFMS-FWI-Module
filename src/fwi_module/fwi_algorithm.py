"""Vectorized Canadian Fire Weather Index calculations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Iterable

import numpy as np
import xarray as xr

from .exceptions import ProcessingError


FFMC_COEFFICIENT = 250.0 * 59.5 / 101.0
DMC_DAY_LENGTH_46N = np.array([6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0])
DMC_DAY_LENGTH_20N = np.array([7.9, 8.4, 8.9, 9.5, 9.9, 10.2, 10.1, 9.7, 9.1, 8.6, 8.1, 7.8])
DMC_DAY_LENGTH_20S = np.array([10.1, 9.6, 9.1, 8.5, 8.1, 7.8, 7.9, 8.3, 8.9, 9.4, 9.9, 10.2])
DMC_DAY_LENGTH_40S = np.array([11.5, 10.5, 9.2, 7.9, 6.8, 6.2, 6.5, 7.4, 8.7, 10.0, 11.2, 11.8])
DC_DAY_FACTOR_20N = np.array([-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6])
DC_DAY_FACTOR_20S = np.array([6.4, 5.0, 2.4, 0.4, -1.6, -1.6, -1.6, -1.6, -1.6, 0.9, 3.8, 5.8])


@dataclass(slots=True)
class FWIState:
    """Recursive state variables needed to continue the FWI computation."""

    ffmc: np.ndarray
    dmc: np.ndarray
    dc: np.ndarray

    @classmethod
    def from_shape(cls, shape: tuple[int, ...], ffmc: float = 85.0, dmc: float = 6.0, dc: float = 15.0) -> "FWIState":
        return cls(
            ffmc=np.full(shape, ffmc, dtype=np.float64),
            dmc=np.full(shape, dmc, dtype=np.float64),
            dc=np.full(shape, dc, dtype=np.float64),
        )

    def copy(self) -> "FWIState":
        return FWIState(ffmc=self.ffmc.copy(), dmc=self.dmc.copy(), dc=self.dc.copy())

    def to_dataset(self, template: xr.DataArray) -> xr.Dataset:
        return xr.Dataset(
            data_vars={
                "ffmc": xr.DataArray(self.ffmc.astype(np.float32), coords=template.coords, dims=template.dims),
                "dmc": xr.DataArray(self.dmc.astype(np.float32), coords=template.coords, dims=template.dims),
                "dc": xr.DataArray(self.dc.astype(np.float32), coords=template.coords, dims=template.dims),
            },
            attrs={"stateful": "true"},
        )

    @classmethod
    def from_dataset(cls, dataset: xr.Dataset) -> "FWIState":
        return cls(
            ffmc=np.asarray(dataset["ffmc"].values, dtype=np.float64),
            dmc=np.asarray(dataset["dmc"].values, dtype=np.float64),
            dc=np.asarray(dataset["dc"].values, dtype=np.float64),
        )


@dataclass(slots=True)
class ComputationResult:
    """Outputs and end-state for a processing window."""

    dataset: xr.Dataset
    state: FWIState


def compute_fwi_indices(
    inputs: xr.Dataset,
    initial_state: FWIState | None = None,
    *,
    lat_adjust: bool = True,
) -> ComputationResult:
    """Compute daily FWI variables for each point of the input grid."""

    _validate_inputs(inputs)
    spatial_template = inputs["temperature"].isel(time=0, drop=True)
    state = initial_state.copy() if initial_state is not None else FWIState.from_shape(spatial_template.shape)
    latitudes = _extract_latitudes(inputs, spatial_template)

    outputs: dict[str, list[xr.DataArray]] = {name: [] for name in ("ffmc", "dmc", "dc", "isi", "bui", "fwi", "dsr")}
    for index, raw_time in enumerate(inputs["time"].values):
        timestamp = _normalize_timestamp(raw_time)
        month = timestamp.month
        temperature = np.asarray(inputs["temperature"].isel(time=index).values, dtype=np.float64)
        relative_humidity = np.asarray(inputs["relative_humidity"].isel(time=index).values, dtype=np.float64)
        wind_speed = np.asarray(inputs["wind_speed"].isel(time=index).values, dtype=np.float64)
        precipitation = np.asarray(inputs["precipitation"].isel(time=index).values, dtype=np.float64)

        relative_humidity = np.clip(relative_humidity, 0.0, 99.9999)
        wind_speed = np.maximum(wind_speed, 0.0)
        precipitation = np.maximum(precipitation, 0.0)

        state.ffmc = fine_fuel_moisture_code(state.ffmc, temperature, relative_humidity, wind_speed, precipitation)
        state.dmc = duff_moisture_code(state.dmc, temperature, relative_humidity, precipitation, latitudes, month, lat_adjust)
        state.dc = drought_code(state.dc, temperature, precipitation, latitudes, month, lat_adjust)

        isi = initial_spread_index(state.ffmc, wind_speed)
        bui = buildup_index(state.dmc, state.dc)
        fwi = fire_weather_index(isi, bui)
        dsr = daily_severity_rating(fwi)

        step_values = {
            "ffmc": state.ffmc,
            "dmc": state.dmc,
            "dc": state.dc,
            "isi": isi,
            "bui": bui,
            "fwi": fwi,
            "dsr": dsr,
        }
        for name, values in step_values.items():
            outputs[name].append(_as_step_dataarray(values, spatial_template, timestamp, name))

    dataset = xr.Dataset({name: xr.concat(data_arrays, dim="time") for name, data_arrays in outputs.items()})
    dataset.attrs.update(
        {
            "title": "Canadian Fire Weather Index outputs",
            "source": "fwi-module",
            "time_reference": "daily noon local standard time inputs, gridded processing",
        }
    )
    for name in dataset.data_vars:
        dataset[name] = dataset[name].astype(np.float32)
    return ComputationResult(dataset=dataset, state=state)


def fine_fuel_moisture_code(
    previous_ffmc: np.ndarray,
    temperature: np.ndarray,
    relative_humidity: np.ndarray,
    wind_speed: np.ndarray,
    precipitation: np.ndarray,
) -> np.ndarray:
    wmo = FFMC_COEFFICIENT * (101.0 - previous_ffmc) / (59.5 + previous_ffmc)
    ra = np.where(precipitation > 0.5, precipitation - 0.5, precipitation)
    rain_effect = 42.5 * ra * np.exp(-100.0 / (251.0 - wmo)) * (1.0 - np.exp(-6.93 / np.maximum(ra, 1e-12)))
    wmo = np.where(
        precipitation > 0.5,
        np.where(wmo > 150.0, wmo + rain_effect + 0.0015 * (wmo - 150.0) ** 2 * np.sqrt(ra), wmo + rain_effect),
        wmo,
    )
    wmo = np.minimum(wmo, 250.0)

    drying_equilibrium = (
        0.942 * (relative_humidity**0.679)
        + 11.0 * np.exp((relative_humidity - 100.0) / 10.0)
        + 0.18 * (21.1 - temperature) * (1.0 - np.exp(-0.115 * relative_humidity))
    )
    wetting_equilibrium = (
        0.618 * (relative_humidity**0.753)
        + 10.0 * np.exp((relative_humidity - 100.0) / 10.0)
        + 0.18 * (21.1 - temperature) * (1.0 - np.exp(-0.115 * relative_humidity))
    )

    z = np.where(
        (wmo < drying_equilibrium) & (wmo < wetting_equilibrium),
        0.424 * (1.0 - ((100.0 - relative_humidity) / 100.0) ** 1.7)
        + 0.0694 * np.sqrt(wind_speed) * (1.0 - ((100.0 - relative_humidity) / 100.0) ** 8),
        0.0,
    )
    x = z * 0.581 * np.exp(0.0365 * temperature)
    wm = np.where((wmo < drying_equilibrium) & (wmo < wetting_equilibrium), wetting_equilibrium - (wetting_equilibrium - wmo) / (10.0**x), wmo)

    z = np.where(
        wmo > drying_equilibrium,
        0.424 * (1.0 - (relative_humidity / 100.0) ** 1.7) + 0.0694 * np.sqrt(wind_speed) * (1.0 - (relative_humidity / 100.0) ** 8),
        z,
    )
    x = z * 0.581 * np.exp(0.0365 * temperature)
    wm = np.where(wmo > drying_equilibrium, drying_equilibrium + (wmo - drying_equilibrium) / (10.0**x), wm)

    ffmc = (59.5 * (250.0 - wm)) / (FFMC_COEFFICIENT + wm)
    return np.clip(ffmc, 0.0, 101.0)


def duff_moisture_code(
    previous_dmc: np.ndarray,
    temperature: np.ndarray,
    relative_humidity: np.ndarray,
    precipitation: np.ndarray,
    latitude: np.ndarray,
    month: int,
    lat_adjust: bool = True,
) -> np.ndarray:
    temperature = np.maximum(temperature, -1.1)
    day_length = np.full_like(temperature, DMC_DAY_LENGTH_46N[month - 1], dtype=np.float64)
    if lat_adjust:
        day_length = np.where((latitude <= 30.0) & (latitude > 10.0), DMC_DAY_LENGTH_20N[month - 1], day_length)
        day_length = np.where((latitude <= -10.0) & (latitude > -30.0), DMC_DAY_LENGTH_20S[month - 1], day_length)
        day_length = np.where((latitude <= -30.0) & (latitude >= -90.0), DMC_DAY_LENGTH_40S[month - 1], day_length)
        day_length = np.where((latitude <= 10.0) & (latitude > -10.0), 9.0, day_length)

    rk = 1.894 * (temperature + 1.1) * (100.0 - relative_humidity) * day_length * 1.0e-4
    pr = np.where(precipitation <= 1.5, previous_dmc, _dmc_post_rain(previous_dmc, precipitation))
    pr = np.maximum(pr, 0.0)
    return np.maximum(pr + rk, 0.0)


def drought_code(
    previous_dc: np.ndarray,
    temperature: np.ndarray,
    precipitation: np.ndarray,
    latitude: np.ndarray,
    month: int,
    lat_adjust: bool = True,
) -> np.ndarray:
    temperature = np.maximum(temperature, -2.8)
    potential_evapotranspiration = (0.36 * (temperature + 2.8) + DC_DAY_FACTOR_20N[month - 1]) / 2.0
    if lat_adjust:
        potential_evapotranspiration = np.where(
            latitude <= -20.0,
            (0.36 * (temperature + 2.8) + DC_DAY_FACTOR_20S[month - 1]) / 2.0,
            potential_evapotranspiration,
        )
        potential_evapotranspiration = np.where(
            (latitude > -20.0) & (latitude <= 20.0),
            (0.36 * (temperature + 2.8) + 1.4) / 2.0,
            potential_evapotranspiration,
        )
    potential_evapotranspiration = np.maximum(potential_evapotranspiration, 0.0)

    effective_rain = 0.83 * precipitation - 1.27
    soil_moisture_index = 800.0 * np.exp(-previous_dc / 400.0)
    drought_after_rain = previous_dc - 400.0 * np.log(1.0 + 3.937 * effective_rain / soil_moisture_index)
    drought_after_rain = np.maximum(drought_after_rain, 0.0)
    drought_start = np.where(precipitation <= 2.8, previous_dc, drought_after_rain)

    return np.maximum(drought_start + potential_evapotranspiration, 0.0)


def initial_spread_index(ffmc: np.ndarray, wind_speed: np.ndarray) -> np.ndarray:
    fuel_moisture = FFMC_COEFFICIENT * (101.0 - ffmc) / (59.5 + ffmc)
    wind_effect = np.exp(0.05039 * wind_speed)
    fine_fuel_moisture = 91.9 * np.exp(-0.1386 * fuel_moisture) * (1.0 + (fuel_moisture**5.31) / 49_300_000.0)
    return 0.208 * wind_effect * fine_fuel_moisture


def buildup_index(dmc: np.ndarray, dc: np.ndarray) -> np.ndarray:
    bui1 = np.where((dmc == 0.0) & (dc == 0.0), 0.0, 0.8 * dc * dmc / np.maximum(dmc + 0.4 * dc, 1e-12))
    p = np.where(dmc == 0.0, 0.0, (dmc - bui1) / dmc)
    cc = 0.92 + (0.0114 * dmc) ** 1.7
    bui0 = np.maximum(dmc - cc * p, 0.0)
    return np.where(bui1 < dmc, bui0, bui1)


def fire_weather_index(isi: np.ndarray, bui: np.ndarray) -> np.ndarray:
    bb = np.where(
        bui > 80.0,
        0.1 * isi * (1000.0 / (25.0 + 108.64 / np.exp(0.023 * bui))),
        0.1 * isi * (0.626 * (bui**0.809) + 2.0),
    )
    return np.where(bb <= 1.0, bb, np.exp(2.72 * ((0.434 * np.log(bb)) ** 0.647)))


def daily_severity_rating(fwi: np.ndarray) -> np.ndarray:
    return 0.0272 * (fwi**1.77)


def _dmc_post_rain(previous_dmc: np.ndarray, precipitation: np.ndarray) -> np.ndarray:
    net_rain = 0.92 * precipitation - 1.27
    initial_moisture = 20.0 + 280.0 / np.exp(0.023 * previous_dmc)
    b = np.where(
        previous_dmc <= 33.0,
        100.0 / (0.5 + 0.3 * previous_dmc),
        np.where(previous_dmc <= 65.0, 14.0 - 1.3 * np.log(previous_dmc), 6.2 * np.log(previous_dmc) - 17.2),
    )
    moisture_after_rain = initial_moisture + 1000.0 * net_rain / (48.77 + b * net_rain)
    return 43.43 * (5.6348 - np.log(np.maximum(moisture_after_rain - 20.0, 1e-12)))


def _as_step_dataarray(values: np.ndarray, template: xr.DataArray, timestamp: datetime, name: str) -> xr.DataArray:
    data_array = xr.DataArray(values.astype(np.float32), coords=template.coords, dims=template.dims, name=name)
    return data_array.expand_dims(time=[np.datetime64(timestamp)])


def _extract_latitudes(inputs: xr.Dataset, template: xr.DataArray) -> np.ndarray:
    for candidate in ("lat", "latitude"):
        if candidate in inputs.coords:
            latitude = inputs.coords[candidate]
            if "time" in latitude.dims:
                latitude = latitude.isel(time=0, drop=True)
            return np.broadcast_to(np.asarray(latitude.values, dtype=np.float64), template.shape)
        if candidate in inputs.data_vars:
            latitude = inputs[candidate]
            if "time" in latitude.dims:
                latitude = latitude.isel(time=0, drop=True)
            return np.broadcast_to(np.asarray(latitude.values, dtype=np.float64), template.shape)
    raise ProcessingError("input dataset must expose latitude coordinates as 'lat' or 'latitude'")


def _normalize_timestamp(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, np.datetime64):
        return datetime.fromtimestamp(value.astype("datetime64[s]").astype(int), tz=UTC).replace(tzinfo=None)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    if hasattr(value, "item"):
        item = value.item()
        if isinstance(item, datetime):
            return item
    raise ProcessingError(f"unsupported time coordinate value: {value!r}")


def _validate_inputs(inputs: xr.Dataset) -> None:
    required = {"temperature", "relative_humidity", "wind_speed", "precipitation"}
    missing = required - set(inputs.data_vars)
    if missing:
        raise ProcessingError(f"input dataset is missing variables: {', '.join(sorted(missing))}")
    if "time" not in inputs.dims:
        raise ProcessingError("input dataset must include a 'time' dimension")