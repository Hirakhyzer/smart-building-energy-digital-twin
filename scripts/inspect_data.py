"""Inspect local ASHRAE input files without generating model results."""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from buildingtwin.data_io import inspect_ashrae_tables, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect local smart-building energy data.")
    parser.add_argument("--raw-dir", default="data/raw/ashrae")
    parser.add_argument("--output", default="outputs/results/data_summary.json")
    args = parser.parse_args()
    summary = inspect_ashrae_tables(args.raw_dir)
    save_json(summary, args.output)
    print(f"Wrote data summary: {Path(args.output).resolve()}")
    print(f"Buildings: {summary['building_count']} | Sites: {summary['site_count']}")
    print("Meter counts:", summary["meter_counts"])

if __name__ == "__main__":
    main()
