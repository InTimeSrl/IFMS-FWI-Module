from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import xarray as xr

from fwi_module.cds_client import DataStoreBackend
from fwi_module.config import AppConfig, dump_example_config
from fwi_module.processor import FWIProcessor


class FakeBackend(DataStoreBackend):
    def __init__(self, atmosphere_collection: str, land_collection: str) -> None:
        self.atmosphere_collection = atmosphere_collection
        self.land_collection = land_collection
        self.requests: list[tuple[str, dict[str, object], Path]] = []

    def check_authentication(self) -> None:
        return None

    def get_collection_end_date(self, collection_id: str) -> date | None:
        return date(2024, 12, 31)

    def retrieve(self, collection_id: str, request: dict[str, object], target_path: Path) -> str | None:
        self.requests.append((collection_id, request, target_path))
        _write_fake_dataset(collection_id, request, target_path, self.atmosphere_collection, self.land_collection)
        return f"req-{len(self.requests)}"


def test_processor_runs_monthly_pipeline_and_resumes(tmp_path: Path, monkeypatch) -> None:
    raw = dump_example_config()
    raw["period"] = {"start": "2023-04-01", "end": "2023-05-05", "spinup_days": 0}
    raw["paths"] = {
        "cache_dir": str(tmp_path / "cache"),
        "output_dir": str(tmp_path / "output"),
        "state_dir": str(tmp_path / "state"),
        "catalog_db": str(tmp_path / "state" / "catalog.sqlite"),
    }
    raw["datasets"]["atmosphere"]["data_format"] = "netcdf"
    raw["datasets"]["land"]["data_format"] = "netcdf"
    config = AppConfig.model_validate(raw).resolved(tmp_path)

    monkeypatch.setattr(
        "fwi_module.preprocess.apply_spatial_mask",
        lambda dataset, land_sea_mask, country_name, land_sea_threshold, coastal_buffer_cells: dataset.assign(
            mask=((land_sea_mask.isel(time=0, drop=True) >= land_sea_threshold).astype("uint8"))
        ),
    )

    backend = FakeBackend(config.datasets.atmosphere.collection_id, config.datasets.land.collection_id)
    processor = FWIProcessor(config, backend=backend)

    outputs = processor.run(resume=False)
    first_request_count = len(backend.requests)

    assert len(outputs) == 2
    assert all(path.exists() for path in outputs)
    assert first_request_count == 4

    with xr.open_dataset(outputs[0]) as dataset:
        assert dataset.attrs["Conventions"] == "CF-1.8"
        assert "spatial_ref" in dataset.data_vars
        assert dataset["spatial_ref"].attrs["grid_mapping_name"] == "latitude_longitude"
        assert dataset["fwi"].attrs["grid_mapping"] == "spatial_ref"
        assert "abbrevs" not in dataset.coords
        assert "names" not in dataset.coords

    resumed_outputs = FWIProcessor(config, backend=backend).run(resume=True)

    assert resumed_outputs == []
    assert len(backend.requests) == first_request_count


def test_processor_annotates_native_lambert_grid_when_projection_metadata_is_present(tmp_path: Path) -> None:
    config = AppConfig.model_validate(dump_example_config()).resolved(tmp_path)
    processor = FWIProcessor(config, backend=FakeBackend(config.datasets.atmosphere.collection_id, config.datasets.land.collection_id))

    dataset = xr.Dataset(
        data_vars={"fwi": (("time", "y", "x"), np.ones((1, 2, 3), dtype=float))},
        coords={
            "time": [np.datetime64("2023-04-01")],
            "y": [-2_937_000.0, -2_931_500.0],
            "x": [-2_937_000.0, -2_931_500.0, -2_926_000.0],
            "lat": (("y", "x"), np.array([[20.292281, 20.292281, 20.292281], [20.3418, 20.3418, 20.3418]], dtype=float)),
            "lon": (("y", "x"), np.array([[-17.485943, -17.4267, -17.3674], [-17.4886, -17.4294, -17.3701]], dtype=float)),
        },
    )
    dataset["fwi"].attrs.update(
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

    annotated = processor._annotate_output_georeferencing(dataset)

    assert annotated["spatial_ref"].attrs["grid_mapping_name"] == "lambert_conformal_conic"
    assert annotated["spatial_ref"].attrs["longitude_of_central_meridian"] == 8.0
    assert annotated["fwi"].attrs["grid_mapping"] == "spatial_ref"
    assert annotated.coords["x"].attrs["standard_name"] == "projection_x_coordinate"
    assert annotated.coords["y"].attrs["standard_name"] == "projection_y_coordinate"


def _write_fake_dataset(
    collection_id: str,
    request: dict[str, object],
    target_path: Path,
    atmosphere_collection: str,
    land_collection: str,
) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    days = [int(day) for day in request["day"]]
    month = int(request["month"][0])
    year = int(request["year"][0])
    time_values = request.get("time", ["00:00"])
    timestamps = [np.datetime64(f"{year:04d}-{month:02d}-{day:02d}T{hour}") for day in days for hour in time_values]
    lat = xr.DataArray(np.array([[39.0, 39.2], [38.8, 39.1]]), dims=("y", "x"))
    lon = xr.DataArray(np.array([[22.0, 22.2], [22.1, 22.3]]), dims=("y", "x"))

    if collection_id == atmosphere_collection:
        dataset = xr.Dataset(
            data_vars={
                "2m_temperature": (("time", "y", "x"), np.full((len(timestamps), 2, 2), 296.15, dtype=float)),
                "2m_relative_humidity": (("time", "y", "x"), np.full((len(timestamps), 2, 2), 35.0, dtype=float)),
                "10m_wind_speed": (("time", "y", "x"), np.full((len(timestamps), 2, 2), 5.0, dtype=float)),
            },
            coords={"time": timestamps, "y": [0, 1], "x": [0, 1], "lat": lat, "lon": lon},
        )
        dataset["2m_temperature"].attrs["units"] = "K"
        dataset["10m_wind_speed"].attrs["units"] = "m s-1"
    elif collection_id == land_collection:
        precipitation = np.zeros((len(timestamps), 2, 2), dtype=float)
        if len(timestamps) > 0:
            precipitation[0, :, :] = 1.0
        dataset = xr.Dataset(
            data_vars={
                "total_precipitation": (("time", "y", "x"), precipitation),
                "land_sea_mask": (("time", "y", "x"), np.ones((len(timestamps), 2, 2), dtype=float)),
            },
            coords={"time": timestamps, "y": [0, 1], "x": [0, 1], "lat": lat, "lon": lon},
        )
        dataset["total_precipitation"].attrs["units"] = "kg m-2"
    else:  # pragma: no cover - defensive branch
        raise AssertionError(f"unexpected collection: {collection_id}")

    dataset.to_netcdf(target_path)