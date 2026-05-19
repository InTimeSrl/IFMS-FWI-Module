from __future__ import annotations

from calendar import monthrange
from datetime import date
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from fwi_module.cds_client import DataStoreBackend
from fwi_module.config import AppConfig, dump_example_config
from fwi_module.processor import FWIProcessor
from fwi_module.runtime_logging import bind_log_context, close_run_logging, configure_run_logging


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

    log_session = configure_run_logging(config.logging, command="run")
    try:
        with bind_log_context(command="run", run_id=log_session.run_id):
            outputs = processor.run(resume=False)
    finally:
        close_run_logging(log_session.logger)
    first_request_count = len(backend.requests)

    assert len(outputs) == 2
    assert all(path.exists() for path in outputs)
    assert first_request_count == 4
    assert log_session.log_path is not None and log_session.log_path.exists()
    log_content = log_session.log_path.read_text(encoding="utf-8")
    assert "Checking CDS authentication before processing run" in log_content
    assert "Starting processing window 1/2" in log_content
    assert "Cache miss for atmosphere dataset" in log_content
    assert "Preparing FWI inputs" in log_content
    assert "Processing daily FWI step 1/30" in log_content
    assert "Writing NetCDF dataset to" in log_content
    assert "Checkpoint written to" in log_content
    assert "Window processing completed successfully" in log_content

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


def test_processor_reuses_yearly_downloads_across_monthly_windows(tmp_path: Path, monkeypatch) -> None:
    raw = dump_example_config()
    raw["period"] = {"start": "2023-04-01", "end": "2023-05-05", "spinup_days": 0}
    raw["download"]["chunking"] = "yearly"
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
    outputs = FWIProcessor(config, backend=backend).run(resume=False)

    assert len(outputs) == 2
    assert len(backend.requests) == 2
    atmosphere_request = next(
        request for collection_id, request, _ in backend.requests if collection_id == config.datasets.atmosphere.collection_id
    )
    assert atmosphere_request["year"] == ["2023"]
    assert atmosphere_request["month"] == ["04", "05"]
    assert atmosphere_request["day"] == [f"{day:02d}" for day in range(1, 31)]


def test_run_percentile_product_with_yearly_chunking_reuses_one_download_per_dataset_and_year(
    tmp_path: Path, monkeypatch
) -> None:
    raw = dump_example_config()
    raw["period"] = {"start": "2023-05-01", "end": "2023-06-30", "spinup_days": 90}
    raw["download"]["chunking"] = "yearly"
    raw["paths"] = {
        "cache_dir": str(tmp_path / "cache"),
        "output_dir": str(tmp_path / "output"),
        "state_dir": str(tmp_path / "state"),
        "catalog_db": str(tmp_path / "state" / "catalog.sqlite"),
    }
    raw["datasets"]["atmosphere"]["data_format"] = "netcdf"
    raw["datasets"]["land"]["data_format"] = "netcdf"
    raw["storage"]["intermediate_output"] = "climatology"
    raw["percentile"] = {
        "start_year": 2020,
        "end_year": 2021,
        "months": [5, 6],
        "percentile": 90.0,
        "method": "exact",
        "block_shape": [1, 1],
        "output_template": "fwi_p{percentile:.0f}_{start_year}_{end_year}_{months}.nc",
    }
    config = AppConfig.model_validate(raw).resolved(tmp_path)

    monkeypatch.setattr(
        "fwi_module.preprocess.apply_spatial_mask",
        lambda dataset, land_sea_mask, country_name, land_sea_threshold, coastal_buffer_cells: dataset.assign(
            mask=((land_sea_mask.isel(time=0, drop=True) >= land_sea_threshold).astype("uint8"))
        ),
    )

    backend = FakeBackend(config.datasets.atmosphere.collection_id, config.datasets.land.collection_id)
    output_path = FWIProcessor(config, backend=backend).run_percentile_product(resume=False)

    assert output_path.exists()
    assert len(backend.requests) == 4
    assert all(request["month"] == ["05", "06"] for _, request, _ in backend.requests)
    assert (config.paths.output_dir / config.storage.filename_template.format(year=2020, month=5)).exists()
    assert (config.paths.output_dir / config.storage.filename_template.format(year=2020, month=6)).exists()
    assert not (config.paths.output_dir / config.storage.filename_template.format(year=2020, month=4)).exists()


def test_run_percentile_product_rejects_non_contiguous_months_for_yearly_chunking(tmp_path: Path) -> None:
    raw = dump_example_config()
    raw["download"]["chunking"] = "yearly"
    raw["percentile"] = {
        "start_year": 2020,
        "end_year": 2021,
        "months": [5, 7],
        "percentile": 90.0,
        "method": "exact",
        "block_shape": [1, 1],
        "output_template": "fwi_p{percentile:.0f}_{start_year}_{end_year}_{months}.nc",
    }
    config = AppConfig.model_validate(raw).resolved(tmp_path)

    processor = FWIProcessor(
        config,
        backend=FakeBackend(config.datasets.atmosphere.collection_id, config.datasets.land.collection_id),
    )

    with pytest.raises(Exception, match="contiguous month range"):
        processor.run_percentile_product(resume=False)


def test_processor_writes_climatology_intermediate_outputs(tmp_path: Path, monkeypatch) -> None:
    raw = dump_example_config()
    raw["period"] = {"start": "2023-04-01", "end": "2023-04-07", "spinup_days": 0}
    raw["paths"] = {
        "cache_dir": str(tmp_path / "cache"),
        "output_dir": str(tmp_path / "output"),
        "state_dir": str(tmp_path / "state"),
        "catalog_db": str(tmp_path / "state" / "catalog.sqlite"),
    }
    raw["datasets"]["atmosphere"]["data_format"] = "netcdf"
    raw["datasets"]["land"]["data_format"] = "netcdf"
    raw["storage"]["intermediate_output"] = "climatology"
    config = AppConfig.model_validate(raw).resolved(tmp_path)

    monkeypatch.setattr(
        "fwi_module.preprocess.apply_spatial_mask",
        lambda dataset, land_sea_mask, country_name, land_sea_threshold, coastal_buffer_cells: dataset.assign(
            mask=((land_sea_mask.isel(time=0, drop=True) >= land_sea_threshold).astype("uint8"))
        ),
    )

    backend = FakeBackend(config.datasets.atmosphere.collection_id, config.datasets.land.collection_id)
    outputs = FWIProcessor(config, backend=backend).run(resume=False)

    assert len(outputs) == 1
    with xr.open_dataset(outputs[0]) as dataset:
        assert set(dataset.data_vars) == {"fwi", "mask", "spatial_ref"}
        assert "temperature" not in dataset.data_vars
        assert "ffmc" not in dataset.data_vars
        assert dataset["fwi"].attrs["grid_mapping"] == "spatial_ref"


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


def test_processor_aggregates_multi_year_percentile_in_spatial_blocks(tmp_path: Path) -> None:
    raw = dump_example_config()
    raw["paths"] = {
        "cache_dir": str(tmp_path / "cache"),
        "output_dir": str(tmp_path / "output"),
        "state_dir": str(tmp_path / "state"),
        "catalog_db": str(tmp_path / "state" / "catalog.sqlite"),
    }
    raw["percentile"] = {
        "start_year": 2020,
        "end_year": 2021,
        "months": [5, 6],
        "percentile": 90.0,
        "method": "exact",
        "block_shape": [1, 1],
        "output_template": "fwi_p{percentile:.0f}_{start_year}_{end_year}_{months}.nc",
    }
    config = AppConfig.model_validate(raw).resolved(tmp_path)
    processor = FWIProcessor(config, backend=FakeBackend(config.datasets.atmosphere.collection_id, config.datasets.land.collection_id))

    monthly_values = {
        (2020, 5): np.array(
            [
                [[10.0, 20.0], [30.0, 40.0]],
                [[12.0, 22.0], [32.0, 42.0]],
            ],
            dtype=np.float32,
        ),
        (2020, 6): np.array(
            [
                [[14.0, 24.0], [34.0, 44.0]],
                [[16.0, 26.0], [36.0, 46.0]],
            ],
            dtype=np.float32,
        ),
        (2021, 5): np.array(
            [
                [[18.0, 28.0], [38.0, 48.0]],
                [[20.0, 30.0], [40.0, 50.0]],
            ],
            dtype=np.float32,
        ),
        (2021, 6): np.array(
            [
                [[22.0, 32.0], [42.0, 52.0]],
                [[24.0, 34.0], [44.0, 54.0]],
            ],
            dtype=np.float32,
        ),
    }

    for (year, month), values in monthly_values.items():
        _write_monthly_fwi_output(config.paths.output_dir / config.storage.filename_template.format(year=year, month=month), year, month, values)

    output_path = processor.aggregate_percentile_product()

    assert output_path.exists()
    stacked = np.concatenate(list(monthly_values.values()), axis=0)
    expected = np.nanpercentile(stacked, 90.0, axis=0)

    with xr.open_dataset(output_path) as dataset:
        np.testing.assert_allclose(dataset["fwi_p90"].values, expected)
        assert dataset.attrs["Conventions"] == "CF-1.8"
        assert dataset.attrs["baseline_years"] == "2020-2021"
        assert dataset["fwi_p90"].attrs["grid_mapping"] == "spatial_ref"
        assert dataset["mask"].dtype.kind in {"i", "u"}


def test_run_percentile_product_processes_each_year_and_writes_final_raster(tmp_path: Path, monkeypatch) -> None:
    raw = dump_example_config()
    raw["period"] = {"start": "2023-05-01", "end": "2023-05-31", "spinup_days": 0}
    raw["paths"] = {
        "cache_dir": str(tmp_path / "cache"),
        "output_dir": str(tmp_path / "output"),
        "state_dir": str(tmp_path / "state"),
        "catalog_db": str(tmp_path / "state" / "catalog.sqlite"),
    }
    raw["datasets"]["atmosphere"]["data_format"] = "netcdf"
    raw["datasets"]["land"]["data_format"] = "netcdf"
    raw["storage"]["intermediate_output"] = "climatology"
    raw["percentile"] = {
        "start_year": 2020,
        "end_year": 2021,
        "months": [5],
        "percentile": 90.0,
        "method": "exact",
        "block_shape": [1, 1],
        "output_template": "fwi_p{percentile:.0f}_{start_year}_{end_year}_{months}.nc",
    }
    config = AppConfig.model_validate(raw).resolved(tmp_path)

    monkeypatch.setattr(
        "fwi_module.preprocess.apply_spatial_mask",
        lambda dataset, land_sea_mask, country_name, land_sea_threshold, coastal_buffer_cells: dataset.assign(
            mask=((land_sea_mask.isel(time=0, drop=True) >= land_sea_threshold).astype("uint8"))
        ),
    )

    backend = FakeBackend(config.datasets.atmosphere.collection_id, config.datasets.land.collection_id)
    processor = FWIProcessor(config, backend=backend)

    log_session = configure_run_logging(config.logging, command="run-percentile")
    try:
        with bind_log_context(command="run-percentile", run_id=log_session.run_id):
            output_path = processor.run_percentile_product(resume=False)
    finally:
        close_run_logging(log_session.logger)

    may_2020 = config.paths.output_dir / config.storage.filename_template.format(year=2020, month=5)
    may_2021 = config.paths.output_dir / config.storage.filename_template.format(year=2021, month=5)

    assert output_path.exists()
    assert may_2020.exists()
    assert may_2021.exists()
    assert (config.paths.state_dir / "percentile" / "2020" / "catalog.sqlite").exists()
    assert (config.paths.state_dir / "percentile" / "2021" / "catalog.sqlite").exists()
    assert len(backend.requests) == 4
    assert log_session.log_path is not None and log_session.log_path.exists()
    log_content = log_session.log_path.read_text(encoding="utf-8")
    assert "Starting percentile workflow" in log_content
    assert "Processing percentile year 1/2" in log_content
    assert "Processing percentile year 2/2" in log_content
    assert "Starting final percentile aggregation step" in log_content
    assert "Aggregating percentile block 1/4" in log_content
    assert "Percentile product written to" in log_content

    with xr.open_dataset(may_2020) as first_year, xr.open_dataset(may_2021) as second_year, xr.open_dataset(output_path) as aggregated:
        assert set(first_year.data_vars) == {"fwi", "mask", "spatial_ref"}
        assert set(second_year.data_vars) == {"fwi", "mask", "spatial_ref"}
        expected = np.nanpercentile(np.concatenate([first_year["fwi"].values, second_year["fwi"].values], axis=0), 90.0, axis=0)
        np.testing.assert_allclose(aggregated["fwi_p90"].values, expected)


def _write_fake_dataset(
    collection_id: str,
    request: dict[str, object],
    target_path: Path,
    atmosphere_collection: str,
    land_collection: str,
) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    time_values = request.get("time", ["00:00"])
    timestamps = _request_timestamps(request, time_values)
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


def _request_timestamps(request: dict[str, object], time_values: list[str]) -> list[np.datetime64]:
    timestamps: list[np.datetime64] = []
    for year_value in sorted({int(value) for value in request["year"]}):
        for month_value in sorted({int(value) for value in request["month"]}):
            last_day = monthrange(year_value, month_value)[1]
            for day_value in sorted({int(value) for value in request["day"]}):
                if day_value > last_day:
                    continue
                for hour in time_values:
                    timestamps.append(np.datetime64(f"{year_value:04d}-{month_value:02d}-{day_value:02d}T{hour}"))
    return timestamps


def _write_monthly_fwi_output(path: Path, year: int, month: int, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    times = np.array([f"{year:04d}-{month:02d}-01", f"{year:04d}-{month:02d}-02"], dtype="datetime64[ns]")
    lat = xr.DataArray(np.array([[39.0, 39.2], [38.8, 39.1]], dtype=float), dims=("y", "x"))
    lon = xr.DataArray(np.array([[22.0, 22.2], [22.1, 22.3]], dtype=float), dims=("y", "x"))
    mask = np.array([[1, 1], [1, 0]], dtype=np.uint8)

    dataset = xr.Dataset(
        data_vars={
            "fwi": (("time", "y", "x"), values),
            "mask": (("y", "x"), mask),
        },
        coords={
            "time": times,
            "y": [0, 1],
            "x": [0, 1],
            "lat": lat,
            "lon": lon,
        },
    )
    dataset.to_netcdf(path)