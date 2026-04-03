def build_ohlc(prices, bars):
    ohlc = []

    for start, end in bars:
        chunk = prices[start : end + 1]
        ohlc.append(
            {
                "open": chunk[0],
                "high": np.max(chunk),
                "low": np.min(chunk),
                "close": chunk[-1],
                "length": len(chunk),
            }
        )

    return ohlc
