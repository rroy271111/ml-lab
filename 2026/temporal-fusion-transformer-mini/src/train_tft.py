import pandas as pd

from pytorch_forecasting import (
    TimeSeriesDataSet,
    TemporalFusionTransformer
)

DATA_PATH = "data/aapl_tft.csv" 

MAX_ENCODER_LENGTH = 30
MAX_PREDICTION_LENGTH = 7

def create_dataset(df):
    return TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        group_ids=["series"],

        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,

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

def main():
    
    df = pd.read_csv(DATA_PATH)

    training_cutoff = (
        df["time_idx"].max()
        - MAX_ENCODER_LENGTH
    )

    print("Training cutoff:", training_cutoff)

    training = create_dataset(
        df[df.time_idx <= training_cutoff]
    )

    print("Training samples:")
    print(len(training))

    validation = TimeSeriesDataSet.from_dataset(
        training,
        df,
        predict=True,
        stop_randomization=True,
    )

    print("Validation samples:")
    print(len(validation))

if __name__ == "__main__":
    main()

