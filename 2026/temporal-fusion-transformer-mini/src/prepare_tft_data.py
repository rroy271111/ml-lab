import pandas as pd

from pathlib import Path

INPUT_PATH = Path("data/aapl_features.csv")
OUTPUT_PATH = Path("data/aapl_tft.csv")

def main():
    df = pd.read_csv(INPUT_PATH)

    # convert back to datetime
    df["Date"] = pd.to_datetime(df["Date"])                   

    # required by PyTorch Forecasting
    df["time_idx"] = range(len(df))

    # group identifier
    df["series"] = "AAPL"   

    print(df[["Date", "time_idx", "series"]].head())

    print()
    print(df.shape)

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()