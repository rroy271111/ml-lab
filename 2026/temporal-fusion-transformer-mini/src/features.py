import pandas as pd
from pathlib import Path

#INTPUT_PATH = Path("data/btc_usd.csv")
#OUTPUT_PATH = Path("data/btc_features.csv")
INTPUT_PATH = Path("data/aapl.csv")
OUTPUT_PATH = Path("data/aapl_features.csv")

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df["returns"] = df["Close"].pct_change(fill_method=None)

    df["sma_7"] = df["Close"].rolling(7).mean()

    df["volatility_7"] = df["returns"].rolling(7).std()

    df["day_of_week"] = df["Date"].dt.dayofweek

    df["month"] = df["Date"].dt.month

    return df

def main():

    df = pd.read_csv(INTPUT_PATH)

    df["Date"] = pd.to_datetime(df["Date"])

    df = add_features(df)   

    df = df.dropna().reset_index(drop=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print(df.head())  

if __name__ == "__main__":
    main()