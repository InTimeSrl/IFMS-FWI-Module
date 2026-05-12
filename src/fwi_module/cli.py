"""Command line interface for the FWI package."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

from .config import load_config
from .exceptions import ConfigError, FWIError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fwi-module", description="FWI processing pipeline for Greece")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate-config", help="Validate a YAML configuration file")
    validate_parser.add_argument("config", type=Path, help="Path to the YAML configuration file")

    inspect_parser = subparsers.add_parser("inspect-cache", help="Inspect the local catalog database")
    inspect_parser.add_argument("config", type=Path, help="Path to the YAML configuration file")

    run_parser = subparsers.add_parser("run", help="Run the processing pipeline")
    run_parser.add_argument("config", type=Path, help="Path to the YAML configuration file")
    run_parser.add_argument("--no-resume", action="store_true", help="Disable resume even if configured")

    resume_parser = subparsers.add_parser("resume", help="Resume the processing pipeline")
    resume_parser.add_argument("config", type=Path, help="Path to the YAML configuration file")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "validate-config":
            config = load_config(args.config)
            print(config.summary())
            return 0

        if args.command == "inspect-cache":
            config = load_config(args.config)
            return _inspect_cache(config.paths.catalog_db)

        if args.command in {"run", "resume"}:
            config = load_config(args.config)
            from .processor import FWIProcessor

            processor = FWIProcessor(config)
            processor.run(resume=(args.command == "resume") or (not args.no_resume and config.processing.resume))
            return 0
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except FWIError as exc:
        print(f"Processing error: {exc}", file=sys.stderr)
        return 3

    parser.print_help(sys.stderr)
    return 1


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