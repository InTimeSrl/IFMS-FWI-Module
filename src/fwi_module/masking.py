"""Spatial masking utilities for Greek land pixels."""

from __future__ import annotations

import numpy as np
import xarray as xr

from .exceptions import ProcessingError


def apply_spatial_mask(
    dataset: xr.Dataset,
    land_sea_mask: xr.DataArray,
    country_name: str,
    land_sea_threshold: float,
    coastal_buffer_cells: int = 0,
) -> xr.Dataset:
    """Mask out non-land and non-Greek cells."""

    land_mask = land_sea_mask
    if "time" in land_mask.dims:
        land_mask = land_mask.isel(time=0, drop=True)
    land_mask = land_mask >= land_sea_threshold
    country_mask = build_country_mask(dataset, country_name)
    combined_mask = (land_mask.astype(bool) & country_mask.astype(bool)).rename("mask")
    combined_mask = _expand_mask(combined_mask, coastal_buffer_cells)

    masked = dataset.where(combined_mask)
    masked["mask"] = combined_mask.astype("uint8")
    return masked


def build_country_mask(dataset: xr.Dataset, country_name: str) -> xr.DataArray:
    """Build a boolean mask for a country using Natural Earth boundaries."""

    import regionmask

    lon, lat = _extract_lon_lat(dataset)
    natural_earth = getattr(regionmask.defined_regions, "natural_earth_v5_1_2", regionmask.defined_regions.natural_earth_v5_0_0)
    countries = natural_earth.countries_50

    try:
        region_index = countries.names.index(country_name)
    except ValueError as exc:
        raise ProcessingError(f"country not found in Natural Earth regions: {country_name}") from exc

    mask = countries.mask_3D(lon, lat, drop=False)
    country_mask = mask.isel(region=region_index)
    extra_coords = [name for name in ("region", "abbrevs", "names") if name in country_mask.coords]
    if extra_coords:
        country_mask = country_mask.drop_vars(extra_coords)
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


def _expand_mask(mask: xr.DataArray, cells: int) -> xr.DataArray:
    if cells <= 0:
        return mask.astype(bool)

    expanded_values = np.asarray(mask.fillna(False).values, dtype=bool)
    for _ in range(cells):
        padded = np.pad(expanded_values, 1, mode="constant", constant_values=False)
        next_values = expanded_values.copy()
        for row_shift in (-1, 0, 1):
            for col_shift in (-1, 0, 1):
                next_values |= padded[
                    1 + row_shift : 1 + row_shift + expanded_values.shape[0],
                    1 + col_shift : 1 + col_shift + expanded_values.shape[1],
                ]
        expanded_values = next_values

    return xr.DataArray(expanded_values, coords=mask.coords, dims=mask.dims, name=mask.name)