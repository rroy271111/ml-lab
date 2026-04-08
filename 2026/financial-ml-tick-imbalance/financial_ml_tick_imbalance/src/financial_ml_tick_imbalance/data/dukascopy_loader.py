from datetime import datetime
import pandas as pd
import dukascopy_python


def download_tick_data(symbol, start, end):
    data = dukascopy_python.fetch(
        symbol=symbol,
        interval="tick",
        start=start,
        end=end,
    )

    return pd.DataFrame(data)
