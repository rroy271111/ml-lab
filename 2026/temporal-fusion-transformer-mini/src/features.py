import pandas as pd
from pathlib import Path

INTPUT_PATH = Path("data/btc_usd.csv")
OUTPUT_PATH = Path("data/btc_features.csv")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df["returns"] = df["Close"].pct_change()

    df["sma_7"] = df["Close"].rolling(7).mean()

    df["voltality_7"] = df["returns"].rolling(7).std()

    df["day_of_week"] = df["Date"].date.dayofweek

    df["month"] = df["Date"].dt.month

    return df
