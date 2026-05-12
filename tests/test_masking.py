from __future__ import annotations

import numpy as np
import xarray as xr

from fwi_module.masking import _expand_mask, apply_spatial_mask


def test_expand_mask_grows_by_one_cell_neighborhood() -> None:
    mask = xr.DataArray(
        np.array(
            [
                [False, False, False],
                [False, True, False],
                [False, False, False],
            ],
            dtype=bool,
        ),
        dims=("y", "x"),
        name="mask",
    )

    expanded = _expand_mask(mask, 1)

    assert expanded.dtype == bool
    assert expanded.values.all()


def test_apply_spatial_mask_expands_coastal_pixels(monkeypatch) -> None:
    dataset = xr.Dataset(
        data_vars={"temperature": (("time", "y", "x"), np.ones((1, 3, 3), dtype=float))},
        coords={"time": [np.datetime64("2023-04-01")], "y": [0, 1, 2], "x": [0, 1, 2]},
    )
    land_sea_mask = xr.DataArray(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
        dims=("y", "x"),
        name="land_sea_mask",
    )
    country_mask = xr.DataArray(
        np.array(
            [
                [False, False, False],
                [False, True, False],
                [False, False, False],
            ],
            dtype=bool,
        ),
        dims=("y", "x"),
        name="country_mask",
    )

    monkeypatch.setattr("fwi_module.masking.build_country_mask", lambda dataset, country_name: country_mask)

    masked = apply_spatial_mask(dataset, land_sea_mask, "Greece", land_sea_threshold=0.5, coastal_buffer_cells=1)

    assert masked["mask"].sum().item() == 9
