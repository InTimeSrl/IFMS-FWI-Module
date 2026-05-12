"""End-to-end orchestration for monthly FWI processing windows."""

from __future__ import annotations

import warnings
from contextlib import ExitStack
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import xarray as xr
from pyproj import CRS

from .cds_client import CERRADataDownloader, DataStoreBackend
from .checkpointing import CatalogStore, WindowRecord
from .config import AppConfig, PercentileConfig
from .exceptions import ConfigError, ProcessingError
from .fwi_algorithm import FWIState, compute_fwi_indices
from .georeferencing import native_grid_projection_attrs, native_lambert_crs, projection_coordinate_attrs
from .preprocess import PreparedInputs, prepare_fwi_inputs
from .storage import write_netcdf_atomic
from .utils import ProcessingWindow, month_windows


class FWIProcessor:
    """Coordinate CDS download, preprocessing, FWI computation and output writing."""

    def __init__(
        self,
        config: AppConfig,
        *,
        backend: DataStoreBackend | None = None,
        prepare_inputs=prepare_fwi_inputs,
    ) -> None:
        self.config = config
        self.catalog = CatalogStore(config.paths.catalog_db)
        self.downloader = CERRADataDownloader(config, self.catalog, backend=backend)
        self.prepare_inputs = prepare_inputs

    def run(self, *, resume: bool | None = None) -> list[Path]:
        resume_enabled = self.config.processing.resume if resume is None else resume
        self.downloader.check_authentication()
        return self._run_windows(resume_enabled=resume_enabled)

    def run_percentile_product(self, *, resume: bool | None = None) -> Path:
        percentile = self._require_percentile_config()
        resume_enabled = self.config.processing.resume if resume is None else resume
        self.downloader.check_authentication()

        for year in range(percentile.start_year, percentile.end_year + 1):
            year_processor = self._processor_for_percentile_year(year)
            year_processor._run_windows(resume_enabled=resume_enabled)

        return self.aggregate_percentile_product()

    def aggregate_percentile_product(self) -> Path:
        percentile = self._require_percentile_config()
        source_paths = self._percentile_source_paths(percentile)
        output_path = self._percentile_output_path(percentile)

        with ExitStack() as stack:
            datasets = [stack.enter_context(xr.open_dataset(path, engine="netcdf4")) for path in source_paths]
            aggregated = self._build_percentile_dataset(datasets, percentile)
            write_netcdf_atomic(aggregated, output_path, self.config.storage)

        return output_path

    def _run_windows(self, *, resume_enabled: bool) -> list[Path]:
        windows = month_windows(self.config.period.extended_start, self.config.period.end)
        state, start_index = self._restore_state(windows) if resume_enabled else (None, 0)

        outputs: list[Path] = []
        for index in range(start_index, len(windows)):
            window = windows[index]
            output_path = self._process_window(window, state)
            if output_path is not None:
                outputs.append(output_path)
            checkpoint = self.catalog.get_window(window.identifier)
            if checkpoint and checkpoint.checkpoint_path:
                with xr.open_dataset(checkpoint.checkpoint_path) as dataset:
                    state = FWIState.from_dataset(dataset.load())
        return outputs

    def _require_percentile_config(self) -> PercentileConfig:
        if self.config.percentile is None:
            raise ConfigError("configuration is missing the percentile section required by percentile commands")
        return self.config.percentile

    def _processor_for_percentile_year(self, year: int) -> "FWIProcessor":
        percentile = self._require_percentile_config()
        raw = self.config.model_dump(mode="python")
        start_month = min(percentile.months)
        end_month = max(percentile.months)
        raw["period"]["start"] = date(year, start_month, 1)
        raw["period"]["end"] = _month_end(date(year, end_month, 1))

        year_state_dir = self.config.paths.state_dir / "percentile" / f"{year:04d}"
        raw["paths"]["state_dir"] = year_state_dir
        raw["paths"]["catalog_db"] = year_state_dir / "catalog.sqlite"

        year_config = AppConfig.model_validate(raw)
        return FWIProcessor(year_config, backend=self.downloader.backend, prepare_inputs=self.prepare_inputs)

    def _percentile_source_paths(self, percentile: PercentileConfig) -> list[Path]:
        paths: list[Path] = []
        missing: list[Path] = []

        for year in range(percentile.start_year, percentile.end_year + 1):
            for month in percentile.months:
                path = self.config.paths.output_dir / self.config.storage.filename_template.format(year=year, month=month)
                if path.exists():
                    paths.append(path)
                else:
                    missing.append(path)

        if missing:
            missing_list = ", ".join(str(path) for path in missing[:6])
            if len(missing) > 6:
                missing_list = f"{missing_list}, ..."
            raise ProcessingError(f"missing monthly outputs required for percentile aggregation: {missing_list}")
        return paths

    def _build_percentile_dataset(self, datasets: list[xr.Dataset], percentile: PercentileConfig) -> xr.Dataset:
        if not datasets:
            raise ProcessingError("no monthly outputs available for percentile aggregation")

        reference = datasets[0]
        if "fwi" not in reference.data_vars:
            raise ProcessingError("monthly output is missing the fwi variable required for percentile aggregation")

        reference_fwi = reference["fwi"]
        if "time" not in reference_fwi.dims:
            raise ProcessingError("monthly output variable fwi must have a time dimension")

        spatial_template = reference_fwi.isel(time=0, drop=True)
        if spatial_template.ndim != 2:
            raise ProcessingError("percentile aggregation currently requires a two-dimensional spatial grid")

        for dataset in datasets[1:]:
            if "fwi" not in dataset.data_vars:
                raise ProcessingError("monthly output is missing the fwi variable required for percentile aggregation")
            current = dataset["fwi"]
            if current.dims != reference_fwi.dims:
                raise ProcessingError("monthly outputs have inconsistent fwi dimensions for percentile aggregation")
            if current.isel(time=0, drop=True).shape != spatial_template.shape:
                raise ProcessingError("monthly outputs have inconsistent spatial shapes for percentile aggregation")

        block_shape = percentile.block_shape
        aggregated_values = np.full(spatial_template.shape, np.nan, dtype=np.float32)
        for row_start in range(0, spatial_template.shape[0], block_shape[0]):
            row_end = min(row_start + block_shape[0], spatial_template.shape[0])
            for col_start in range(0, spatial_template.shape[1], block_shape[1]):
                col_end = min(col_start + block_shape[1], spatial_template.shape[1])
                indexers = {
                    spatial_template.dims[0]: slice(row_start, row_end),
                    spatial_template.dims[1]: slice(col_start, col_end),
                }
                block_values = [np.asarray(dataset["fwi"].isel(indexers).values, dtype=np.float32) for dataset in datasets]
                stacked = np.concatenate(block_values, axis=0)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
                    percentile_block = np.nanpercentile(stacked, percentile.percentile, axis=0)
                aggregated_values[row_start:row_end, col_start:col_end] = percentile_block.astype(np.float32, copy=False)

        percentile_name = _percentile_variable_name(percentile.percentile)
        result = xr.Dataset(
            data_vars={
                percentile_name: xr.DataArray(aggregated_values, coords=spatial_template.coords, dims=spatial_template.dims)
            },
            attrs={
                "title": "FWI seasonal percentile product",
                "source": "fwi-module",
                "aggregation_method": "exact blocked np.nanpercentile over daily FWI outputs",
                "baseline_years": f"{percentile.start_year}-{percentile.end_year}",
                "season_months": ",".join(f"{month:02d}" for month in percentile.months),
                "percentile": percentile.percentile,
            },
        )
        result[percentile_name].attrs.update(
            {
                "long_name": f"FWI {percentile.percentile:g}th percentile",
                "source_variable": "fwi",
            }
        )

        if "mask" in reference.data_vars:
            mask = reference["mask"]
            result["mask"] = mask.isel(time=0, drop=True) if "time" in mask.dims else mask.copy(deep=True)

        for name in result.data_vars:
            result[name].encoding = {}

        return self._annotate_output_georeferencing(result)

    def _percentile_output_path(self, percentile: PercentileConfig) -> Path:
        months_label = "-".join(f"{month:02d}" for month in percentile.months)
        filename = percentile.output_template.format(
            percentile=percentile.percentile,
            start_year=percentile.start_year,
            end_year=percentile.end_year,
            months=months_label,
        )
        return self.config.paths.output_dir / filename

    def _process_window(self, window: ProcessingWindow, state: FWIState | None) -> Path | None:
        self.catalog.upsert_window(
            WindowRecord(
                window_id=window.identifier,
                window_start=window.start.isoformat(),
                window_end=window.end.isoformat(),
                status="running",
            )
        )
        downloads = self.downloader.fetch_window(window)
        prepared = self.prepare_inputs(downloads.atmosphere_path, downloads.land_path, self.config)
        compute_inputs = prepared.dataset.drop_vars("mask", errors="ignore")
        result = compute_fwi_indices(compute_inputs, initial_state=state)
        output_dataset = self._compose_output_dataset(prepared, result.dataset)
        trimmed_dataset = self._trim_to_requested_period(output_dataset)

        output_path: Path | None = None
        if trimmed_dataset.sizes.get("time", 0) > 0:
            output_path = self._output_path(window)
            write_netcdf_atomic(trimmed_dataset, output_path, self.config.storage)

        checkpoint_path = self._checkpoint_path(window)
        checkpoint_template = prepared.dataset["temperature"].isel(time=-1, drop=True)
        checkpoint_dataset = result.state.to_dataset(checkpoint_template)
        checkpoint_dataset.attrs.update({"window_end": window.end.isoformat()})
        write_netcdf_atomic(checkpoint_dataset, checkpoint_path, self.config.storage)

        self.catalog.upsert_window(
            WindowRecord(
                window_id=window.identifier,
                window_start=window.start.isoformat(),
                window_end=window.end.isoformat(),
                status="completed",
                output_path=str(output_path) if output_path is not None else None,
                checkpoint_path=str(checkpoint_path),
                metadata={
                    "atmosphere_path": str(downloads.atmosphere_path),
                    "land_path": str(downloads.land_path),
                },
            )
        )
        return output_path

    def _restore_state(self, windows: list[ProcessingWindow]) -> tuple[FWIState | None, int]:
        latest = self.catalog.latest_completed_window()
        if latest is None or latest.checkpoint_path is None:
            return None, 0

        checkpoint_path = Path(latest.checkpoint_path)
        if not checkpoint_path.exists():
            return None, 0

        with xr.open_dataset(checkpoint_path) as dataset:
            state = FWIState.from_dataset(dataset.load())

        window_ids = [window.identifier for window in windows]
        try:
            next_index = window_ids.index(latest.window_id) + 1
        except ValueError:
            return None, 0
        return state, next_index

    def _compose_output_dataset(self, prepared: PreparedInputs, fwi_outputs: xr.Dataset) -> xr.Dataset:
        if self.config.storage.include_inputs:
            meteorology = prepared.dataset[["temperature", "relative_humidity", "wind_speed", "precipitation", "mask"]]
            output = xr.merge([meteorology, fwi_outputs], compat="override", join="inner")
        else:
            output = xr.merge([prepared.dataset[["mask"]], fwi_outputs], compat="override", join="inner")
        return self._annotate_output_georeferencing(output)

    def _annotate_output_georeferencing(self, dataset: xr.Dataset) -> xr.Dataset:
        cleaned = dataset.drop_vars(["region", "abbrevs", "names"], errors="ignore")
        if "lat" not in cleaned.coords or "lon" not in cleaned.coords:
            return cleaned

        annotated = cleaned.copy()
        spatial_ref_attrs = self._spatial_ref_attrs(annotated)
        annotated["spatial_ref"] = xr.DataArray(np.int32(0), attrs=spatial_ref_attrs)
        annotated.coords["time"].attrs.setdefault("standard_name", "time") if "time" in annotated.coords else None
        if "time" in annotated.coords:
            annotated.coords["time"].attrs.setdefault("axis", "T")
        if "x" in annotated.coords:
            for key, value in projection_coordinate_attrs("x").items():
                annotated.coords["x"].attrs.setdefault(key, value)
        if "y" in annotated.coords:
            for key, value in projection_coordinate_attrs("y").items():
                annotated.coords["y"].attrs.setdefault(key, value)

        for name in annotated.data_vars:
            if name == "spatial_ref":
                continue
            variable_attrs = dict(annotated[name].attrs)
            variable_attrs["coordinates"] = "lat lon"
            variable_attrs["grid_mapping"] = "spatial_ref"
            annotated[name].attrs = variable_attrs

        annotated.attrs["Conventions"] = "CF-1.8"
        return annotated

    def _spatial_ref_attrs(self, dataset: xr.Dataset) -> dict[str, object]:
        projection = native_grid_projection_attrs(dataset)
        if projection is not None and "x" in dataset.coords and "y" in dataset.coords:
            native_crs = native_lambert_crs(projection)
            spatial_ref_attrs = dict(native_crs.to_cf())
            wkt = native_crs.to_wkt()
            spatial_ref_attrs.update(
                {
                    "projected_crs_name": "CERRA native Lambert conformal grid",
                    "long_name": "CERRA native Lambert conformal grid",
                    "spatial_ref": wkt,
                    "crs_wkt": wkt,
                }
            )
            return spatial_ref_attrs

        geographic_crs = CRS.from_epsg(4326)
        spatial_ref_attrs = dict(geographic_crs.to_cf())
        wkt = geographic_crs.to_wkt()
        spatial_ref_attrs.update(
            {
                "long_name": "WGS 84 geographic CRS for auxiliary geolocation coordinates",
                "spatial_ref": wkt,
                "crs_wkt": wkt,
            }
        )
        return spatial_ref_attrs

    def _trim_to_requested_period(self, dataset: xr.Dataset) -> xr.Dataset:
        if "time" not in dataset.coords:
            return dataset
        time_values = dataset["time"].values.astype("datetime64[D]")
        start = np.datetime64(self.config.period.start.isoformat())
        end = np.datetime64(self.config.period.end.isoformat())
        selector = (time_values >= start) & (time_values <= end)
        return dataset.isel(time=selector)

    def _output_path(self, window: ProcessingWindow) -> Path:
        filename = self.config.storage.filename_template.format(year=window.start.year, month=window.start.month)
        return self.config.paths.output_dir / filename

    def _checkpoint_path(self, window: ProcessingWindow) -> Path:
        filename = self.config.storage.state_template.format(year=window.end.year, month=window.end.month, day=window.end.day)
        return self.config.paths.state_dir / filename


def _month_end(value: date) -> date:
    if value.month == 12:
        return date(value.year, 12, 31)
    return date(value.year, value.month + 1, 1) - timedelta(days=1)


def _percentile_variable_name(percentile: float) -> str:
    label = f"{percentile:g}".replace(".", "_")
    return f"fwi_p{label}"