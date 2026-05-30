import pandas as pd
from torch.utils.data import DataLoader

from pytorch_forecasting import TimeSeriesDataSet

MAX_ENCODER_LENGTH = 30
MAX_PREDICTION_LENGTH = 7
NUM_WORKERS = 0
DEFAULT_BATCH_SIZE = 64

def create_training_dataset(df: pd.DataFrame) -> TimeSeriesDataSet:
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

def create_validation_dataset(training: TimeSeriesDataSet, df: pd.DataFrame,) -> TimeSeriesDataSet:
    return TimeSeriesDataSet.from_dataset(
        training,
        df, 
        predict=True,
        stop_randomization=True,
    )


def create_dataloaders(
        training: TimeSeriesDataSet,
        validation: TimeSeriesDataSet,
        batch_size: int = 64,
) -> tuple[DataLoader, DataLoader]:
    train_loader = training.to_dataloader(
        train=True,
        batch_size=batch_size,
        num_workers=0,
    )

    val_loader= validation.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=0,
    )

    return train_loader, val_loader