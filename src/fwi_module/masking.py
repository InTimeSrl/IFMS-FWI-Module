"""Spatial masking utilities for Greek land pixels."""

from __future__ import annotations

import xarray as xr

from .exceptions import ProcessingError


def apply_spatial_mask(
    dataset: xr.Dataset,
    land_sea_mask: xr.DataArray,
    country_name: str,
    land_sea_threshold: float,
) -> xr.Dataset:
    """Mask out non-land and non-Greek cells."""

    land_mask = land_sea_mask
    if "time" in land_mask.dims:
        land_mask = land_mask.isel(time=0, drop=True)
    land_mask = land_mask >= land_sea_threshold
    country_mask = build_country_mask(dataset, country_name)
    combined_mask = (land_mask.astype(bool) & country_mask.astype(bool)).rename("mask")

    masked = dataset.where(combined_mask)
    masked["mask"] = combined_mask.astype("uint8")
    return masked


def build_country_mask(dataset: xr.Dataset, country_name: str) -> xr.DataArray:
    """Build a boolean mask for a country using Natural Earth boundaries."""

    import regionmask

    lon, lat = _extract_lon_lat(dataset)
    countries = regionmask.defined_regions.natural_earth_v5_0_0.countries_50

    try:
        region_index = countries.names.index(country_name)
    except ValueError as exc:
        raise ProcessingError(f"country not found in Natural Earth regions: {country_name}") from exc

    mask = countries.mask_3D(lon=lon, lat=lat)
    country_mask = mask.isel(region=region_index)
    if "region" in country_mask.coords:
        country_mask = country_mask.drop_vars("region")
    return country_mask.fillna(False).astype(bool).rename("country_mask")


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