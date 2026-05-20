import pandas as pd

from pytorch_forecasting import TimeSeriesDataSet

DATA_PATH= "data/aapl_tft.csv"

def main():
    df = pd.read_csv(DATA_PATH)

    #print(df.columns.tolist())

    training = TimeSeriesDataSet(
        df, 
        time_idx="time_idx",
        target="Close",
        group_ids=["series"],
        max_encoder_length=30,
        max_prediction_length=7,
        static_categoricals=["series"],
        time_varying_known_reals=[
            "time_idx",
            "day_of_week",
            "month",
        ],

        time_varying_unknown_reals=[
            "Close",
            "High",
            "Low",
            "Open",
            "Volume",
            "returns",
            "sma_7",
            "sma_30",
            "volatility_7",
        ],

    )
    print(training)

    x, y = training[0]

    print("\nTarget:")
    print(y)

    print("\nKeys:")
    print(x.keys())

    print()
    print("Number of samples:")
    print(len(training))

if __name__ == "__main__":
    main()