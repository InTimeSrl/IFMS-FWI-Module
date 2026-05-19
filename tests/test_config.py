from __future__ import annotations

from pathlib import Path

import pytest

from fwi_module.config import AppConfig, dump_example_config, load_config
from fwi_module.exceptions import ConfigError


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_example_config_loads_and_resolves_paths() -> None:
    config = load_config(PROJECT_ROOT / "examples" / "greece.yaml")

    assert config.region.country_name == "Greece"
    assert config.period.extended_start.isoformat() == "2023-01-01"
    assert config.datasets.atmosphere.collection_id == "reanalysis-cerra-single-levels"
    assert config.paths.cache_dir.is_absolute()
    assert config.paths.catalog_db == config.paths.state_dir / "catalog.sqlite"
    assert config.logging.directory == config.paths.state_dir / "logs"
    assert config.storage.intermediate_output == "full"
    assert config.percentile is not None
    assert config.percentile.start_year == 1991
    assert config.percentile.months == (5, 6, 7, 8, 9)


def test_logging_config_defaults_to_state_log_dir() -> None:
    config = AppConfig.model_validate(dump_example_config())

    assert config.logging.directory == config.paths.state_dir / "logs"


def test_invalid_bbox_is_rejected() -> None:
    data = dump_example_config()
    data["region"]["bbox"]["north"] = 34.0
    data["region"]["bbox"]["south"] = 35.0

    with pytest.raises(Exception):
        AppConfig.model_validate(data)


def test_invalid_yaml_path_raises_config_error() -> None:
    with pytest.raises(ConfigError):
        load_config(PROJECT_ROOT / "examples" / "missing.yaml")


def test_percentile_config_loads_and_normalizes_months() -> None:
    data = dump_example_config()
    data["percentile"] = {
        "start_year": 1991,
        "end_year": 2020,
        "months": [9, 5, 7],
        "percentile": 90.0,
        "method": "exact",
        "block_shape": [64, 96],
        "output_template": "fwi_p{percentile:.0f}_{start_year}_{end_year}.nc",
    }

    config = AppConfig.model_validate(data)

    assert config.percentile is not None
    assert config.percentile.months == (5, 7, 9)
    assert config.percentile.block_shape == (64, 96)


def test_download_chunking_accepts_yearly() -> None:
    data = dump_example_config()
    data["download"]["chunking"] = "yearly"

    config = AppConfig.model_validate(data)

    assert config.download.chunking == "yearly"


def test_invalid_percentile_config_is_rejected() -> None:
    data = dump_example_config()
    data["percentile"] = {
        "start_year": 2020,
        "end_year": 1991,
        "months": [5, 13],
        "percentile": 90.0,
        "method": "exact",
        "block_shape": [128, 128],
    }

    with pytest.raises(Exception):
        AppConfig.model_validate(data)


def test_logging_directory_is_resolved_with_config_file() -> None:
    config = load_config(PROJECT_ROOT / "examples" / "greece.yaml")

    assert config.logging.directory == config.paths.state_dir / "logs"