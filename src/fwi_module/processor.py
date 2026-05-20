"""End-to-end orchestration for monthly FWI processing windows."""

from __future__ import annotations

import inspect
import logging
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
from .georeferencing import geographic_coordinate_attrs, reproject_dataset_to_wgs84
from .preprocess import PreparedInputs, prepare_fwi_inputs
from .runtime_logging import bind_log_context
from .storage import write_netcdf_atomic
from .utils import ProcessingWindow, month_windows, months_are_contiguous


logger = logging.getLogger(__name__)


class FWIProcessor:
    """Coordinate CDS download, preprocessing, FWI computation and output writing."""

    def __init__(
        self,
        config: AppConfig,
        *,
        backend: DataStoreBackend | None = None,
        prepare_inputs=prepare_fwi_inputs,
        percentile_year: int | None = None,
    ) -> None:
        self.config = config
        self.catalog = CatalogStore(config.paths.catalog_db)
        self._prepare_inputs_accepts_window = _prepare_inputs_supports_window(prepare_inputs)
        self.downloader = CERRADataDownloader(config, self.catalog, backend=backend, percentile_year=percentile_year)
        self.prepare_inputs = prepare_inputs

    def run(self, *, resume: bool | None = None) -> list[Path]:
        resume_enabled = self.config.processing.resume if resume is None else resume
        logger.info("Starting processing run with resume=%s", resume_enabled)
        logger.info("Checking CDS authentication before processing run")
        self.downloader.check_authentication()
        logger.info("CDS authentication completed successfully")
        return self._run_windows(resume_enabled=resume_enabled)

    def run_percentile_product(self, *, resume: bool | None = None) -> Path:
        percentile = self._require_percentile_config()
        resume_enabled = self.config.processing.resume if resume is None else resume
        logger.info(
            "Starting percentile workflow for years %s-%s, months=%s, percentile=%s with resume=%s",
            percentile.start_year,
            percentile.end_year,
            list(percentile.months),
            percentile.percentile,
            resume_enabled,
        )
        logger.info("Checking CDS authentication before percentile workflow")
        self.downloader.check_authentication()
        logger.info("CDS authentication completed successfully")

        total_years = percentile.end_year - percentile.start_year + 1
        for index, year in enumerate(range(percentile.start_year, percentile.end_year + 1), start=1):
            year_processor = self._processor_for_percentile_year(year)
            with bind_log_context(year=year, phase="percentile-year"):
                logger.info("Processing percentile year %s/%s", index, total_years)
                logger.info("Percentile year state directory: %s", year_processor.config.paths.state_dir)
                year_processor._run_windows(resume_enabled=resume_enabled)
                logger.info("Completed percentile year %s", year)

        logger.info("Starting final percentile aggregation step")
        return self.aggregate_percentile_product()

    def aggregate_percentile_product(self) -> Path:
        percentile = self._require_percentile_config()
        source_paths = self._percentile_source_paths(percentile)
        output_path = self._percentile_output_path(percentile)

        with bind_log_context(phase="percentile-aggregation"):
            logger.info("Starting percentile aggregation from %s monthly outputs", len(source_paths))
            logger.info("Percentile aggregation target: %s", output_path)
        with ExitStack() as stack:
            datasets = [stack.enter_context(xr.open_dataset(path, engine="netcdf4")) for path in source_paths]
            aggregated = self._build_percentile_dataset(datasets, percentile)
            write_netcdf_atomic(aggregated, output_path, self.config.storage)

        with bind_log_context(phase="percentile-aggregation"):
            logger.info("Percentile product written to %s", output_path)

        return output_path

    def _run_windows(self, *, resume_enabled: bool) -> list[Path]:
        windows = month_windows(self.config.period.extended_start, self.config.period.end)
        state, start_index = self._restore_state(windows) if resume_enabled else (None, 0)

        logger.info("Prepared %s monthly processing windows", len(windows))
        if resume_enabled:
            logger.info("Resume is enabled; starting from window index %s", start_index)
        else:
            logger.info("Resume is disabled; processing starts from the first window")

        outputs: list[Path] = []
        for index in range(start_index, len(windows)):
            window = windows[index]
            with bind_log_context(window=window.identifier, phase="window"):
                logger.info("Starting processing window %s/%s", index + 1, len(windows))
                output_path = self._process_window(window, state)
            if output_path is not None:
                outputs.append(output_path)
            checkpoint = self.catalog.get_window(window.identifier)
            if checkpoint and checkpoint.checkpoint_path:
                with xr.open_dataset(checkpoint.checkpoint_path) as dataset:
                    state = FWIState.from_dataset(dataset.load())
                with bind_log_context(window=window.identifier, phase="checkpoint"):
                    logger.info("Loaded checkpoint state from %s", checkpoint.checkpoint_path)
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
        if self.config.download.chunking == "yearly":
            if not months_are_contiguous(percentile.months):
                raise ConfigError(
                    "download.chunking=yearly requires percentile.months to define a contiguous month range"
                )
            raw["period"]["spinup_days"] = 0
        raw["period"]["start"] = date(year, start_month, 1)
        raw["period"]["end"] = _month_end(date(year, end_month, 1))

        year_state_dir = self.config.paths.state_dir / "percentile" / f"{year:04d}"
        raw["paths"]["state_dir"] = year_state_dir
        raw["paths"]["catalog_db"] = year_state_dir / "catalog.sqlite"

        year_config = AppConfig.model_validate(raw)
        return FWIProcessor(
            year_config,
            backend=self.downloader.backend,
            prepare_inputs=self.prepare_inputs,
            percentile_year=year,
        )

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
            logger.error("Percentile aggregation is missing %s monthly outputs", len(missing))
            raise ProcessingError(f"missing monthly outputs required for percentile aggregation: {missing_list}")
        logger.info("Collected %s monthly outputs for percentile aggregation", len(paths))
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
        total_blocks = ((spatial_template.shape[0] + block_shape[0] - 1) // block_shape[0]) * (
            (spatial_template.shape[1] + block_shape[1] - 1) // block_shape[1]
        )
        logger.info(
            "Validated percentile aggregation inputs on grid shape=%s using block_shape=%s (%s blocks)",
            spatial_template.shape,
            block_shape,
            total_blocks,
        )
        block_index = 0
        for row_start in range(0, spatial_template.shape[0], block_shape[0]):
            row_end = min(row_start + block_shape[0], spatial_template.shape[0])
            for col_start in range(0, spatial_template.shape[1], block_shape[1]):
                col_end = min(col_start + block_shape[1], spatial_template.shape[1])
                block_index += 1
                logger.info(
                    "Aggregating percentile block %s/%s rows=%s:%s cols=%s:%s",
                    block_index,
                    total_blocks,
                    row_start,
                    row_end,
                    col_start,
                    col_end,
                )
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

        logger.info("Percentile aggregation dataset ready with variables: %s", ", ".join(result.data_vars))
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
        with bind_log_context(window=window.identifier, phase="window"):
            logger.info("Marking processing window as running in the catalog")
            self.catalog.upsert_window(
                WindowRecord(
                    window_id=window.identifier,
                    window_start=window.start.isoformat(),
                    window_end=window.end.isoformat(),
                    status="running",
                )
            )
            downloads = self.downloader.fetch_window(window)
            logger.info("Downloads ready: atmosphere=%s land=%s", downloads.atmosphere_path, downloads.land_path)
            prepared = self._prepare_window_inputs(downloads.atmosphere_path, downloads.land_path, window)
            logger.info("Prepared input dataset with sizes=%s", dict(prepared.dataset.sizes))
            compute_inputs = prepared.dataset.drop_vars("mask", errors="ignore")
            result = compute_fwi_indices(compute_inputs, initial_state=state)
            logger.info("Computed FWI outputs with sizes=%s", dict(result.dataset.sizes))
            output_dataset = self._compose_output_dataset(prepared, result.dataset)
            trimmed_dataset = self._trim_to_requested_period(output_dataset)
            logger.info("Trimmed output dataset to requested period with sizes=%s", dict(trimmed_dataset.sizes))

            output_path: Path | None = None
            if trimmed_dataset.sizes.get("time", 0) > 0:
                output_path = self._output_path(window)
                with bind_log_context(phase="output"):
                    write_netcdf_atomic(trimmed_dataset, output_path, self.config.storage)
                logger.info("Window output written to %s", output_path)
            else:
                logger.info("Window produced no samples inside the requested period; skipping monthly output write")

            checkpoint_path = self._checkpoint_path(window)
            checkpoint_template = prepared.dataset["temperature"].isel(time=-1, drop=True)
            checkpoint_dataset = result.state.to_dataset(checkpoint_template)
            checkpoint_dataset.attrs.update({"window_end": window.end.isoformat()})
            with bind_log_context(phase="checkpoint"):
                write_netcdf_atomic(checkpoint_dataset, checkpoint_path, self.config.storage)
            logger.info("Checkpoint written to %s", checkpoint_path)

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
            logger.info("Window processing completed successfully")
            return output_path

    def _prepare_window_inputs(self, atmosphere_path: Path, land_path: Path, window: ProcessingWindow) -> PreparedInputs:
        if self._prepare_inputs_accepts_window:
            return self.prepare_inputs(atmosphere_path, land_path, self.config, window=window)
        return self.prepare_inputs(atmosphere_path, land_path, self.config)

    def _restore_state(self, windows: list[ProcessingWindow]) -> tuple[FWIState | None, int]:
        latest = self.catalog.latest_completed_window()
        if latest is None or latest.checkpoint_path is None:
            logger.info("No completed window checkpoint found; starting from scratch")
            return None, 0

        checkpoint_path = Path(latest.checkpoint_path)
        if not checkpoint_path.exists():
            logger.warning("Checkpoint path recorded in catalog is missing: %s", checkpoint_path)
            return None, 0

        with xr.open_dataset(checkpoint_path) as dataset:
            state = FWIState.from_dataset(dataset.load())

        window_ids = [window.identifier for window in windows]
        try:
            next_index = window_ids.index(latest.window_id) + 1
        except ValueError:
            logger.warning("Latest completed window %s is outside the current run plan; restarting from the beginning", latest.window_id)
            return None, 0
        logger.info("Restored state from %s and will resume at window index %s", checkpoint_path, next_index)
        return state, next_index

    def _compose_output_dataset(self, prepared: PreparedInputs, fwi_outputs: xr.Dataset) -> xr.Dataset:
        if self.config.storage.intermediate_output == "climatology":
            output = xr.merge([prepared.dataset[["mask"]], fwi_outputs[["fwi"]]], compat="override", join="inner")
            return self._annotate_output_georeferencing(output)

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

        annotated = reproject_dataset_to_wgs84(cleaned)
        spatial_ref_attrs = self._spatial_ref_attrs(annotated)
        annotated["spatial_ref"] = xr.DataArray(np.int32(0), attrs=spatial_ref_attrs)
        annotated.coords["time"].attrs.setdefault("standard_name", "time") if "time" in annotated.coords else None
        if "time" in annotated.coords:
            annotated.coords["time"].attrs.setdefault("axis", "T")
        if "x" in annotated.coords:
            for key, value in geographic_coordinate_attrs("x").items():
                annotated.coords["x"].attrs.setdefault(key, value)
        if "y" in annotated.coords:
            for key, value in geographic_coordinate_attrs("y").items():
                annotated.coords["y"].attrs.setdefault(key, value)
        if "lon" in annotated.coords:
            for key, value in geographic_coordinate_attrs("x").items():
                annotated.coords["lon"].attrs.setdefault(key, value)
        if "lat" in annotated.coords:
            for key, value in geographic_coordinate_attrs("y").items():
                annotated.coords["lat"].attrs.setdefault(key, value)

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
        geographic_crs = CRS.from_epsg(4326)
        spatial_ref_attrs = dict(geographic_crs.to_cf())
        wkt = geographic_crs.to_wkt()
        spatial_ref_attrs.update(
            {
                "long_name": "WGS 84 geographic CRS for reprojected export grid",
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


def _prepare_inputs_supports_window(prepare_inputs) -> bool:
    try:
        parameters = inspect.signature(prepare_inputs).parameters
    except (TypeError, ValueError):
        return False
    return "window" in parameters