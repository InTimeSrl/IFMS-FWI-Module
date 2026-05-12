from __future__ import annotations

from pathlib import Path

from fwi_module.checkpointing import CatalogStore, DownloadRecord, WindowRecord


def test_catalog_store_persists_downloads_and_windows(tmp_path: Path) -> None:
    store = CatalogStore(tmp_path / "catalog.sqlite")

    store.upsert_download(
        DownloadRecord(
            cache_key="abc123",
            dataset_id="reanalysis-cerra-single-levels",
            window_start="2023-04-01",
            window_end="2023-04-30",
            status="completed",
            file_path="data/cache/atmos_202304.grib",
            metadata={"collection": "atmosphere"},
        )
    )
    store.upsert_window(
        WindowRecord(
            window_id="2023-04-01_2023-04-30",
            window_start="2023-04-01",
            window_end="2023-04-30",
            status="completed",
            output_path="data/output/fwi_202304.nc",
            checkpoint_path="data/state/state_20230430.nc",
        )
    )

    download = store.get_download("abc123")
    window = store.latest_completed_window()
    summary = store.summary()

    assert download is not None
    assert download.status == "completed"
    assert window is not None
    assert window.output_path == "data/output/fwi_202304.nc"
    assert summary == {"downloads": 1, "processing_windows": 1}