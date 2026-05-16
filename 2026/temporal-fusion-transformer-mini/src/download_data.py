import yfinance as yf

from pathlib import Path

#OUTPUT_PATH = Path("data/btc_usd.csv")
OUTPUT_PATH = Path("data/aapl.csv")

def main():
    #btc = yf.download("BTC-USD", start="2018-01-01", interval="1d")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = yf.download(
        "AAPL",
        start="2018-01-01",
        auto_adjust=True
    )

    df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(df.head())
    print(df.columns)

    # Flatten MultiIndex columns
    #btc.columns = btc.columns.get_level_values(0)

    # Move Date index into a normal column
    #btc.reset_index(inplace=True)

    #btc.to_csv(OUTPUT_PATH)

    print(f"Saved data to {OUTPUT_PATH}")
    #print(btc.head())
    #print(btc.columns)


if __name__ == "__main__":
    main()
