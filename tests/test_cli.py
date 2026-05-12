from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pytest

from fwi_module.cli import main
from fwi_module.config import dump_example_config


def test_probe_cds_requires_both_window_dates(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)

    exit_code = main(["probe-cds", str(config_path), "--start", "2023-04-01"])

    assert exit_code == 2


def test_run_requires_both_override_dates(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)

    exit_code = main(["run", str(config_path), "--start", "2023-04-01", "--no-resume"])

    assert exit_code == 2


def test_probe_cds_download_uses_first_month_when_no_dates(tmp_path: Path, monkeypatch, capsys) -> None:
    config_path = _write_config(tmp_path)
    state: dict[str, object] = {}

    class FakeCatalog:
        def __init__(self, db_path):
            state["db_path"] = db_path

    class FakeBackend:
        def get_collection_end_date(self, collection_id: str):
            return date(2024, 12, 31)

    class FakeDownloader:
        def __init__(self, config, catalog):
            self.backend = FakeBackend()

        def check_authentication(self) -> None:
            state["auth_checked"] = True

        def fetch_window(self, window):
            state["window"] = window

            class Result:
                atmosphere_path = Path("cache/atmosphere.nc")
                land_path = Path("cache/land.nc")

            return Result()

    monkeypatch.setattr("fwi_module.cli.CatalogStore", FakeCatalog)
    monkeypatch.setattr("fwi_module.cli.CERRADataDownloader", FakeDownloader)

    exit_code = main(["probe-cds", str(config_path), "--download"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert state["auth_checked"] is True
    assert state["window"].start.isoformat() == "2023-04-01"
    assert state["window"].end.isoformat() == "2023-04-30"
    assert "CDS authentication: OK" in output
    assert "Downloaded atmosphere file:" in output


def test_cdsapi_env_fallback_is_accepted(tmp_path: Path, monkeypatch) -> None:
    config_path = _write_config(tmp_path, client="cdsapi")
    monkeypatch.setenv("CDSAPI_KEY", "test-token")
    monkeypatch.delenv("ECMWF_DATASTORES_KEY", raising=False)

    class FakeCatalog:
        def __init__(self, db_path):
            self.db_path = db_path

    class FakeBackend:
        def get_collection_end_date(self, collection_id: str):
            return date(2024, 12, 31)

    class FakeDownloader:
        def __init__(self, config, catalog):
            assert config.cds.client == "cdsapi"
            self.backend = FakeBackend()

        def check_authentication(self) -> None:
            return None

    monkeypatch.setattr("fwi_module.cli.CatalogStore", FakeCatalog)
    monkeypatch.setattr("fwi_module.cli.CERRADataDownloader", FakeDownloader)

    exit_code = main(["probe-cds", str(config_path)])

    assert exit_code == 0


def test_run_period_override_is_passed_to_processor(tmp_path: Path, monkeypatch) -> None:
    import fwi_module.processor as processor_module

    config_path = _write_config(tmp_path)
    state: dict[str, object] = {}

    class FakeProcessor:
        def __init__(self, config) -> None:
            state["start"] = config.period.start.isoformat()
            state["end"] = config.period.end.isoformat()

        def run(self, *, resume: bool | None = None):
            state["resume"] = resume
            return []

    monkeypatch.setattr(processor_module, "FWIProcessor", FakeProcessor)

    exit_code = main(["run", str(config_path), "--start", "2023-04-02", "--end", "2023-04-07", "--no-resume"])

    assert exit_code == 0
    assert state["start"] == "2023-04-02"
    assert state["end"] == "2023-04-07"
    assert state["resume"] is False


def _write_config(tmp_path: Path, *, client: str = "ecmwf_datastores") -> Path:
    import yaml

    raw = dump_example_config()
    raw["period"] = {"start": "2023-04-01", "end": "2023-05-05", "spinup_days": 0}
    raw["cds"]["client"] = client
    raw["paths"] = {
        "cache_dir": str(tmp_path / "cache"),
        "output_dir": str(tmp_path / "output"),
        "state_dir": str(tmp_path / "state"),
        "catalog_db": str(tmp_path / "state" / "catalog.sqlite"),
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path