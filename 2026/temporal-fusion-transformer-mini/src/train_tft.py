import pandas as pd

from pytorch_forecasting import (
    TimeSeriesDataSet,
    TemporalFusionTransformer
)

DATA_PATH = "data/aapl_tft.csv" 

MAX_ENCODER_LENGTH = 30
MAX_PREDICTION_LENGTH = 7

def create_dataset(df):
    return 