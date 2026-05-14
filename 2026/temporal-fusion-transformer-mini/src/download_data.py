import yfinance as yf

from pathlib import Path

OUTPUT_PATH = Path("data/btc_usd.csv")


def main():
    btc = yf.download("BTC-USD", start="2018-01-01", interval="1d")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    btc.to_csv(OUTPUT_PATH)

    print(f"Saved data to {OUTPUT_PATH}")
    print(btc.head())


if __name__ == "__main__":
    main()
