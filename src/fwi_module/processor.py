"""End-to-end orchestration for monthly FWI processing windows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from .cds_client import CERRADataDownloader, DataStoreBackend
from .checkpointing import CatalogStore, WindowRecord
from .config import AppConfig
from .fwi_algorithm import FWIState, compute_fwi_indices
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
            return xr.merge([meteorology, fwi_outputs], compat="override", join="inner")
        return xr.merge([prepared.dataset[["mask"]], fwi_outputs], compat="override", join="inner")

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