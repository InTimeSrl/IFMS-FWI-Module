"""Command line interface for the FWI package."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import date
from pathlib import Path

from .cds_client import CERRADataDownloader
from .checkpointing import CatalogStore
from .config import AppConfig, load_config
from .exceptions import ConfigError, FWIError
from .runtime_logging import bind_log_context, close_run_logging, configure_run_logging
from .utils import ProcessingWindow, month_windows


RUN_COMMANDS = {"run", "resume", "run-percentile", "aggregate-percentile"}
LOG_LEVEL_CHOICES = ("DEBUG", "INFO", "WARNING", "ERROR")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fwi-module", description="FWI processing pipeline for Greece")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate-config", help="Validate a YAML configuration file")
    validate_parser.add_argument("config", type=Path, help="Path to the YAML configuration file")

    inspect_parser = subparsers.add_parser("inspect-cache", help="Inspect the local catalog database")
    inspect_parser.add_argument("config", type=Path, help="Path to the YAML configuration file")

    probe_parser = subparsers.add_parser("probe-cds", help="Check CDS authentication and optionally download a sample window")
    probe_parser.add_argument("config", type=Path, help="Path to the YAML configuration file")
    probe_parser.add_argument("--start", type=str, help="Sample window start date in ISO format (YYYY-MM-DD)")
    probe_parser.add_argument("--end", type=str, help="Sample window end date in ISO format (YYYY-MM-DD)")
    probe_parser.add_argument("--download", action="store_true", help="Download the sample window after validating authentication")

    run_parser = subparsers.add_parser("run", help="Run the processing pipeline")
    run_parser.add_argument("config", type=Path, help="Path to the YAML configuration file")
    run_parser.add_argument("--start", type=str, help="Override the processing start date (YYYY-MM-DD)")
    run_parser.add_argument("--end", type=str, help="Override the processing end date (YYYY-MM-DD)")
    run_parser.add_argument("--no-resume", action="store_true", help="Disable resume even if configured")
    _add_logging_arguments(run_parser)

    run_percentile_parser = subparsers.add_parser("run-percentile", help="Run the multi-year seasonal percentile workflow")
    run_percentile_parser.add_argument("config", type=Path, help="Path to the YAML configuration file")
    run_percentile_parser.add_argument("--no-resume", action="store_true", help="Disable resume even if configured")
    _add_logging_arguments(run_percentile_parser)

    aggregate_percentile_parser = subparsers.add_parser(
        "aggregate-percentile",
        help="Aggregate existing monthly outputs into the configured percentile raster",
    )
    aggregate_percentile_parser.add_argument("config", type=Path, help="Path to the YAML configuration file")
    _add_logging_arguments(aggregate_percentile_parser)

    resume_parser = subparsers.add_parser("resume", help="Resume the processing pipeline")
    resume_parser.add_argument("config", type=Path, help="Path to the YAML configuration file")
    resume_parser.add_argument("--start", type=str, help="Override the processing start date (YYYY-MM-DD)")
    resume_parser.add_argument("--end", type=str, help="Override the processing end date (YYYY-MM-DD)")
    _add_logging_arguments(resume_parser)

    return parser


def _add_logging_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--log-level", choices=LOG_LEVEL_CHOICES, help="Override the runtime logging level")
    parser.add_argument(
        "--log-dir",
        type=lambda value: Path(value).expanduser().resolve(),
        help="Override the directory where the per-run log file is written",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    log_session = None

    try:
        if args.command == "validate-config":
            config = load_config(args.config)
            print(config.summary())
            return 0

        if args.command == "inspect-cache":
            config = load_config(args.config)
            return _inspect_cache(config.paths.catalog_db)

        if args.command == "probe-cds":
            config = load_config(args.config)
            return _probe_cds(config, start=args.start, end=args.end, download=args.download)

        if args.command in {"run", "resume"}:
            config = load_config(args.config)
            config = _override_processing_period(config, start=args.start, end=args.end)
            config = _override_logging_settings(config, level=args.log_level, directory=args.log_dir)
            log_session = configure_run_logging(config.logging, command=args.command)
            from .processor import FWIProcessor

            processor = FWIProcessor(config)
            with bind_log_context(command=args.command, run_id=log_session.run_id):
                _log_command_start(log_session.log_path, args.command, config, config_path=args.config)
                processor.run(resume=(args.command == "resume") or (not args.no_resume and config.processing.resume))
                _log_command_success(args.command)
            _print_log_location(log_session.log_path)
            return 0

        if args.command == "run-percentile":
            config = load_config(args.config)
            config = _override_logging_settings(config, level=args.log_level, directory=args.log_dir)
            log_session = configure_run_logging(config.logging, command=args.command)
            from .processor import FWIProcessor

            processor = FWIProcessor(config)
            with bind_log_context(command=args.command, run_id=log_session.run_id):
                _log_command_start(log_session.log_path, args.command, config, config_path=args.config)
                output_path = processor.run_percentile_product(resume=not args.no_resume and config.processing.resume)
                _log_command_success(args.command)
            print(f"Percentile output: {output_path}")
            _print_log_location(log_session.log_path)
            return 0

        if args.command == "aggregate-percentile":
            config = load_config(args.config)
            config = _override_logging_settings(config, level=args.log_level, directory=args.log_dir)
            log_session = configure_run_logging(config.logging, command=args.command)
            from .processor import FWIProcessor

            processor = FWIProcessor(config)
            with bind_log_context(command=args.command, run_id=log_session.run_id):
                _log_command_start(log_session.log_path, args.command, config, config_path=args.config)
                output_path = processor.aggregate_percentile_product()
                _log_command_success(args.command)
            print(f"Percentile output: {output_path}")
            _print_log_location(log_session.log_path)
            return 0
    except ConfigError as exc:
        _log_command_error(log_session, f"Configuration error: {exc}")
        print(f"Configuration error: {exc}", file=sys.stderr)
        _print_log_location(log_session.log_path if log_session is not None else None, stream=sys.stderr)
        return 2
    except FWIError as exc:
        _log_command_error(log_session, f"Processing error: {exc}")
        print(f"Processing error: {exc}", file=sys.stderr)
        _print_log_location(log_session.log_path if log_session is not None else None, stream=sys.stderr)
        return 3
    finally:
        if log_session is not None:
            close_run_logging(log_session.logger)

    parser.print_help(sys.stderr)
    return 1


def _log_command_start(log_path: Path | None, command: str, config: AppConfig, *, config_path: Path) -> None:
    import logging

    logger = logging.getLogger(__name__)
    with bind_log_context(phase="startup"):
        logger.info("Starting command %s", command)
        logger.info("Configuration file: %s", config_path)
        logger.info("Run log file: %s", log_path if log_path is not None else "disabled")
        logger.info("Output directory: %s", config.paths.output_dir)
        logger.info("State directory: %s", config.paths.state_dir)
        logger.info("Configured period: %s -> %s", config.period.start.isoformat(), config.period.end.isoformat())


def _log_command_success(command: str) -> None:
    import logging

    logger = logging.getLogger(__name__)
    with bind_log_context(phase="shutdown"):
        logger.info("Command %s completed successfully", command)


def _log_command_error(log_session, message: str) -> None:
    import logging

    if log_session is None:
        return
    logger = logging.getLogger(__name__)
    with bind_log_context(command=log_session.command, run_id=log_session.run_id, phase="error"):
        logger.exception(message)


def _print_log_location(log_path: Path | None, *, stream=None) -> None:
    if log_path is None:
        return
    if stream is None:
        stream = sys.stdout
    print(f"Run log: {log_path}", file=stream)


def _inspect_cache(db_path: Path) -> int:
    if not db_path.exists():
        print(f"Catalog database not found: {db_path}")
        return 0

    with sqlite3.connect(db_path) as connection:
        cursor = connection.cursor()
        tables = {
            row[0]
            for row in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if not tables:
            print(f"Catalog database exists but is empty: {db_path}")
            return 0

        print(f"Catalog database: {db_path}")
        for table_name in sorted(tables):
            count = cursor.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            print(f"{table_name}: {count}")

    return 0


def _probe_cds(config, *, start: str | None, end: str | None, download: bool) -> int:
    window = _probe_window(config, start=start, end=end) if download or start or end else None
    catalog = CatalogStore(config.paths.catalog_db)
    downloader = CERRADataDownloader(config, catalog)
    downloader.check_authentication()

    atmosphere_id = config.datasets.atmosphere.collection_id
    land_id = config.datasets.land.collection_id
    atmosphere_end = downloader.backend.get_collection_end_date(atmosphere_id)
    land_end = downloader.backend.get_collection_end_date(land_id)

    print(f"CDS authentication: OK")
    print(f"Atmosphere collection: {atmosphere_id}")
    print(f"Atmosphere latest date: {atmosphere_end.isoformat() if atmosphere_end else 'unknown'}")
    print(f"Land collection: {land_id}")
    print(f"Land latest date: {land_end.isoformat() if land_end else 'unknown'}")

    if not download:
        return 0

    assert window is not None
    downloaded = downloader.fetch_window(window)
    print(f"Downloaded atmosphere file: {downloaded.atmosphere_path}")
    print(f"Downloaded land file: {downloaded.land_path}")
    return 0


def _probe_window(config, *, start: str | None, end: str | None) -> ProcessingWindow:
    if bool(start) != bool(end):
        raise ConfigError("probe-cds requires both --start and --end when overriding the sample window")
    if start and end:
        try:
            return ProcessingWindow(start=date.fromisoformat(start), end=date.fromisoformat(end))
        except ValueError as exc:
            raise ConfigError("invalid ISO date passed to probe-cds") from exc

    windows = month_windows(config.period.start, config.period.end)
    if not windows:
        raise ConfigError("configuration period does not contain any monthly processing window")
    return windows[0]


def _override_processing_period(config: AppConfig, *, start: str | None, end: str | None) -> AppConfig:
    if start is None and end is None:
        return config
    if bool(start) != bool(end):
        raise ConfigError("run/resume requires both --start and --end when overriding the processing period")

    try:
        override_start = date.fromisoformat(start)
        override_end = date.fromisoformat(end)
    except ValueError as exc:
        raise ConfigError("invalid ISO date passed to run/resume") from exc

    raw = config.model_dump(mode="python")
    raw["period"]["start"] = override_start
    raw["period"]["end"] = override_end
    return AppConfig.model_validate(raw)


def _override_logging_settings(config: AppConfig, *, level: str | None, directory: Path | None) -> AppConfig:
    if level is None and directory is None:
        return config

    raw = config.model_dump(mode="python")
    raw.setdefault("logging", {})
    if level is not None:
        raw["logging"]["level"] = level
    if directory is not None:
        raw["logging"]["directory"] = directory
    raw["logging"]["enabled"] = True
    return AppConfig.model_validate(raw)