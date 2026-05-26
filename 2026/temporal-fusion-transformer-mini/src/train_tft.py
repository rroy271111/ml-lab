import pandas as pd
import lightning as L

from pytorch_forecasting import (
    TimeSeriesDataSet,
    TemporalFusionTransformer
)

from lightning.pytorch.callbacks import(
    EarlyStopping,
    LearningRateMonitor,
)

from lightning.pytorch.loggers import CSVLogger

from pytorch_forecasting.metrics import RMSE

DATA_PATH = "data/aapl_tft.csv" 

MAX_ENCODER_LENGTH = 30
MAX_PREDICTION_LENGTH = 7



def create_dataset(df):
    return TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="Close",
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
        - MAX_PREDICTION_LENGTH
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

    train_dataloader = training.to_dataloader(
        train=True,
        batch_size=64,
        num_workers=0,
    )

    val_dataloader = validation.to_dataloader(
        train=False,
        batch_size=64,
        num_workers=0,
    )

    print("Train batches:", len(train_dataloader))
    print("Validation batches:", len(val_dataloader))  

    tft = TemporalFusionTransformer.from_dataset(
    training, 
    learning_rate=1e-3,
    hidden_size=16,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=8,
    loss=RMSE(),
    log_interval=10,
    reduce_on_plateau_patience=4,
)
    print(tft)

    trainer = L.Trainer(
        max_epochs=10,
        accelerator="auto",
        devices=1,
        gradient_clip_val=0.1,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                patience=5,
                mode="min",
            ),
            LearningRateMonitor(),
        ],
        logger=CSVLogger("logs"),
    )

    trainer.fit(
        tft,
        train_dataloaders = train_dataloader,
        val_dataloaders = val_dataloader,
    )


if __name__ == "__main__":
    main()

