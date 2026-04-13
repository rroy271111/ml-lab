from datetime import datetime, timedelta
import pandas as pd
import lzma
import requests
import struct

# import dukascopy_python

BASE_URL = "https://datafeed.dukascopy.com/datafeed"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.dukascopy.com/",
    "Accept-Encoding": "gzip, deflate, br",
}


def _get_hour_url(symbol, dt):
    return (
        f"{BASE_URL}/{symbol}/{dt.year}/{dt.month - 1:02d}/"
        f"{dt.day:02d}/{dt.hour:02d}h_ticks.bi5"
    )


def _download_hour(symbol, dt):
    url = _get_hour_url(symbol, dt)
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
    except requests.RequestException as e:
        print(f"Request failed for {url}: {e}")
        return None

    if r.status_code != 200 or len(r.content) == 0:
        return None

    try:
        return lzma.decompress(r.content)
    except lzma.LZMAError:
        return None


def _parse_ticks(binary_data, base_dt):
    ticks = []
    size = 20  # each tick = 20 bytes

    base = base_dt.replace(minute=0, second=0, microsecond=0)

    for i in range(0, len(binary_data), size):
        chunk = binary_data[i : i + size]
        if len(chunk) < size:
            continue

        ms, ask, bid, ask_vol, bid_vol = struct.unpack(">IIIff", chunk)

        ticks.append(
            {
                "datetime": base + timedelta(milliseconds=ms),
                "ask": ask / 1e5,
                "bid": bid / 1e5,
                "ask_volume": round(ask_vol, 2),
                "bid_volume": round(bid_vol, 2),
            }
        )

    return ticks


def download_tick_data(symbol, start, end):
    current = start
    all_ticks = []

    while current < end:
        binary = _download_hour(symbol, current)

        if binary:
            ticks = _parse_ticks(binary, current)
            all_ticks.extend(ticks)
            print(f"{current.strftime('%Y-%m-%d %H:00')} - {len(ticks)} ticks")
        else:
            print(f"{current.strftime('%Y-%m-%d %H:00')} - No Data")

        current += timedelta(hours=1)

    df = pd.DataFrame(all_ticks)

    if df.empty:
        raise ValueError(
            f"No tick data downloaded for {symbol} " f"from {start} to {end}"
        )
    return df
