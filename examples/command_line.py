"""Command-line options shared by the examples.

Price data lives outside git, in ``data/`` by default: one Yahoo
Finance-style CSV per stock, named ``<SYMBOL>.csv``.
"""

import argparse
from pathlib import Path

from investing import Portfolio

DEFAULT_DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "data"


def parse_arguments(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-directory", type=Path, default=DEFAULT_DATA_DIRECTORY,
                        help="folder of <SYMBOL>.csv price files (default: data/)")
    parser.add_argument("--symbols", nargs="+",
                        help="symbols to load (default: every CSV in the data folder)")
    return parser.parse_args()


def load_portfolio(arguments: argparse.Namespace) -> Portfolio:
    return Portfolio.from_directory(arguments.data_directory, arguments.symbols)
