"""Price download tests. They use made-up responses and never touch the network."""

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import pandas as pd

from investing import Stock
from investing import price_download
from investing.price_download import PriceDownloadError, parse_chart, save_prices

# Midnight UTC timestamps for three consecutive days; the exchange is in UTC.
DAY_ONE, DAY_TWO, DAY_THREE = 1704067200, 1704153600, 1704240000


def chart_payload(timestamps, closes, session=None):
    count = len(timestamps)
    meta = {"exchangeTimezoneName": "UTC"}
    if session:
        meta["currentTradingPeriod"] = {"regular": {"start": session[0], "end": session[1]}}
    return {"chart": {"error": None, "result": [{
        "meta": meta,
        "timestamp": timestamps,
        "indicators": {
            "quote": [{"open": closes, "high": closes, "low": closes, "close": closes,
                       "volume": [1000] * count}],
            "adjclose": [{"adjclose": [close * 0.5 if close is not None else None for close in closes]}],
        },
    }]}}


class TestParseChart(unittest.TestCase):
    def test_columns_and_order(self):
        table = parse_chart(chart_payload([DAY_TWO, DAY_ONE], [11.0, 10.0]))
        self.assertEqual(list(table.columns), price_download.COLUMNS)
        self.assertEqual(table["Close"].tolist(), [10.0, 11.0])
        self.assertEqual(table["Adj Close"].tolist(), [5.0, 5.5])
        self.assertEqual(str(table["Date"].iloc[0].date()), "2024-01-01")

    def test_drops_days_without_a_close(self):
        table = parse_chart(chart_payload([DAY_ONE, DAY_TWO], [10.0, None]))
        self.assertEqual(len(table), 1)

    def test_drops_unfinished_day_while_market_open(self):
        payload = chart_payload([DAY_ONE, DAY_TWO, DAY_THREE], [10.0, 11.0, 12.0],
                                session=(DAY_THREE + 3600, DAY_THREE + 7200))
        self.assertEqual(len(parse_chart(payload, now=DAY_THREE + 5000)), 2)   # open: drop today
        self.assertEqual(len(parse_chart(payload, now=DAY_THREE + 9000)), 3)   # closed: keep it

    def test_error_response(self):
        payload = {"chart": {"result": None, "error": {"description": "No data found"}}}
        with self.assertRaisesRegex(PriceDownloadError, "No data found"):
            parse_chart(payload)


class TestSaving(unittest.TestCase):
    def test_saved_file_loads_as_stock(self):
        payload = chart_payload([DAY_ONE, DAY_TWO, DAY_THREE], [10.0, 11.0, 12.1])
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(price_download, "fetch_chart", return_value=payload):
            path = save_prices("test", directory)
            self.assertEqual(path.name, "TEST.csv")
            stock = Stock("TEST", path)
            self.assertEqual(len(stock.prices), 3)
            self.assertAlmostEqual(stock.time_series.df[Stock.RETURN].iloc[-1], 0.1)

    def test_command_line_continues_after_a_failure(self):
        def fake_fetch(symbol, *arguments, **kwargs):
            if symbol == "BAD":
                raise PriceDownloadError("BAD: No data found")
            return chart_payload([DAY_ONE, DAY_TWO], [10.0, 11.0])

        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(price_download, "fetch_chart", side_effect=fake_fetch), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as errors:
            exit_code = price_download.main(["BAD", "GOOD", "--data-directory", directory, "--pause", "0"])
            self.assertEqual(exit_code, 1)
            self.assertTrue((Path(directory) / "GOOD.csv").exists())
            self.assertIn("FAILED BAD", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
