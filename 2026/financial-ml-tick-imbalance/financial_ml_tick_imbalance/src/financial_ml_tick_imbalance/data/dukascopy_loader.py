from datetime import datetime, timedelta
import pandas as pd
import lzma
import requests
import struct

# import dukascopy_python

BASE_URL = "https://datafeed.dukascopy.com/datafeed"


def _get_hour_url(symbol, dt):
    return (
        f"{BASE_URL}/{symbol}/{dt.year}/{dt.month-1:02d}"
        f"{dt.day:02d}/{dt.hour:02d}h_ticks.bi5"
    )


def _download_hour(symbol, dt):
    url = _get_hour_url(symbol, dt)
    r = requests.get(url)

    if r.status_code != 200:
        return None

    try:
        decompressed = lzma.decompress(r.content)
    except Exception:
        return None

    return decompressed


def _parse_ticks(binary_data):
    ticks = []
    size = 20  # each tick = 20 bytes

    for i in range(0, len(binary_data), size):
        chunk = binary_data[i : i + size]
        if len(chunk) < size:
            continue

        ms, ask, bid, ask_vol, bid_vol = struct.unpack(">IIIff", chunk)

        ticks.append(
            {
                "time": ms,
                "ask": ask / 1e5,
                "bid": bid / 1e5,
                "ask_volume": ask_vol,
                "bid_volume": bid_vol,
            }
        )

    return ticks


def download_tick_data(symbol, start, end):
    current = start
    all_ticks = []

    while current < end:
        binary = _download_hour(symbol, current)

        if binary:
            ticks = _parse_ticks(binary)

            for t in ticks:
                t["datetime"] = current.replace(
                    minute=0, second=0, microsecond=0
                ) + timedelta(milliseconds=t["time"])

            all_ticks.extend(ticks)

        current += timedelta(hours=1)

    return pd.DataFrame(all_ticks)
