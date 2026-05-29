import pandas as pd

from pytorch_forecasting import TimeSeriesDataSet

MAX_ENCODER_LENGTH= 30
MAX_PREDICTION_LENGTH = 7

def create_dataset(df:pd.DataFrame) -> TimeSeriesDataSet:
    return TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="Close",
        group_ids=["series"],

        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        
        static_categoricals=[
            "series",

        ],

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