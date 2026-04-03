def compute_features(ohlc):
    features = []

    for bar in ohlc:
        ret = (bar["close"] - bar["open"]) / bar["open"]
        range_ = (bar["high"] - bar["low"]) / bar["open"]

        features.append({"return": ret, "range": range, "length": bar["length"]})

    return features
