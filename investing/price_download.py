"""Download daily stock prices from Yahoo Finance.

Usage from the repo root:

    python -m investing.price_download SYMBOL [SYMBOL ...] [--data-directory data]

Each symbol is saved as ``<data-directory>/<SYMBOL>.csv`` in the format the
rest of the package reads:

    Date,Open,High,Low,Close,Adj Close,Volume

Prices come from Yahoo's public chart endpoint, which is unofficial: it can
change or rate-limit without notice, and the data is for personal research
under Yahoo's terms. Each download fetches the full history again, because
adjusted closes for past dates change whenever a dividend or split occurs.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

CHART_URL = "https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
EARLIEST_TIMESTAMP = -2208988800    # 1900-01-01, so the full history comes back
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (fe_investing price download)"}
COLUMNS = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]


class PriceDownloadError(Exception):
    """The symbol couldn't be downloaded (unknown symbol, no data, or network trouble)."""


def _timestamp(day: str | date | None, default: int) -> int:
    if day is None:
        return default
    day = pd.Timestamp(day)
    return int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp())


def fetch_chart(symbol: str, start=None, end=None, timeout: float = 30, retries: int = 2) -> dict:
    """Raw JSON response for one symbol's daily history. ``start``/``end`` are
    dates (inclusive start, exclusive end); omit them for the full history."""
    query = urllib.parse.urlencode({
        "period1": _timestamp(start, EARLIEST_TIMESTAMP),
        "period2": _timestamp(end, int(time.time())),
        "interval": "1d",
        "events": "div,split",
        "includeAdjustedClose": "true",
    })
    url = CHART_URL.format(symbol=urllib.parse.quote(symbol, safe="")) + "?" + query
    request = urllib.request.Request(url, headers=REQUEST_HEADERS)
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code == 429 and attempt < retries:  # rate-limited: wait and retry
                time.sleep(2 * (attempt + 1))
                continue
            try:
                description = json.load(error)["chart"]["error"]["description"]
            except Exception:
                description = f"HTTP {error.code}"
            raise PriceDownloadError(f"{symbol}: {description}") from error
        except urllib.error.URLError as error:
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
                continue
            raise PriceDownloadError(f"{symbol}: network error ({error.reason})") from error
    raise PriceDownloadError(f"{symbol}: download failed")


def parse_chart(payload: dict, now: float | None = None) -> pd.DataFrame:
    """Turn a chart response into a table with ``COLUMNS``, oldest first.

    Dates are in the exchange's own time zone. A bar for today is dropped
    while the market is still open, since its prices aren't final.
    """
    chart = payload.get("chart", {})
    if chart.get("error"):
        raise PriceDownloadError(chart["error"].get("description", "unknown error"))
    results = chart.get("result") or []
    if not results or not results[0].get("timestamp"):
        raise PriceDownloadError("no price data returned")
    result = results[0]
    meta = result.get("meta", {})
    quote = result["indicators"]["quote"][0]
    adjusted = (result["indicators"].get("adjclose") or [{}])[0].get("adjclose") or quote["close"]

    zone = meta.get("exchangeTimezoneName", "UTC")
    dates = pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_convert(zone).normalize().tz_localize(None)
    table = pd.DataFrame({
        "Date": dates,
        "Open": quote["open"], "High": quote["high"], "Low": quote["low"], "Close": quote["close"],
        "Adj Close": adjusted, "Volume": quote["volume"],
    }, columns=COLUMNS).astype({column: float for column in COLUMNS[1:]})

    regular_session = meta.get("currentTradingPeriod", {}).get("regular", {})
    now = time.time() if now is None else now
    if regular_session and now < regular_session.get("end", 0):
        session_day = pd.to_datetime(regular_session["start"], unit="s", utc=True).tz_convert(zone).normalize()
        table = table[table["Date"] != session_day.tz_localize(None)]

    table = table.dropna(subset=["Close"]).drop_duplicates("Date", keep="last")
    table["Volume"] = table["Volume"].round().astype("Int64")
    return table.sort_values("Date", ignore_index=True)


def download_prices(symbol: str, start=None, end=None) -> pd.DataFrame:
    """Daily prices for one symbol, oldest first, with ``COLUMNS``."""
    try:
        return parse_chart(fetch_chart(symbol, start, end))
    except PriceDownloadError as error:
        if str(error).startswith(f"{symbol}:"):
            raise
        raise PriceDownloadError(f"{symbol}: {error}") from error


def write_prices(prices: pd.DataFrame, path) -> Path:
    """Write a price table as a CSV the rest of the package can read."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    output = prices.copy()
    output["Date"] = pd.to_datetime(output["Date"]).dt.strftime("%Y-%m-%d")
    output.to_csv(path, index=False, float_format="%.6f")
    return path


def save_prices(symbol: str, directory="data", start=None, end=None) -> Path:
    """Download one symbol and save it as ``<directory>/<SYMBOL>.csv``."""
    return write_prices(download_prices(symbol, start, end), Path(directory) / f"{symbol.upper()}.csv")


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download daily prices from Yahoo Finance into CSV files.")
    parser.add_argument("symbols", nargs="+", help="ticker symbols as Yahoo writes them")
    parser.add_argument("--data-directory", default="data", help="where to save <SYMBOL>.csv (default: data)")
    parser.add_argument("--start", help="first date, YYYY-MM-DD (default: full history)")
    parser.add_argument("--end", help="last date, YYYY-MM-DD, exclusive (default: today)")
    parser.add_argument("--pause", type=float, default=1.0, help="seconds to wait between symbols")
    options = parser.parse_args(arguments)

    failures = 0
    for index, symbol in enumerate(options.symbols):
        if index:
            time.sleep(options.pause)
        try:
            path = save_prices(symbol, options.data_directory, options.start, options.end)
            prices = pd.read_csv(path)
            print(f"{symbol}: {len(prices)} days, {prices['Date'].iloc[0]} to {prices['Date'].iloc[-1]} -> {path}")
        except PriceDownloadError as error:
            failures += 1
            print(f"FAILED {error}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
