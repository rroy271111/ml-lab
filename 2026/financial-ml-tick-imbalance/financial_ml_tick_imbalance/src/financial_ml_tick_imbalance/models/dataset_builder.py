import pandas as pd


def build_training_dataset(features, labels):
    df_features = pd.DataFrame(features)
    df_labels = pd.Series(labels, name="label")

    if len(df_features) != len(df_labels):
        raise ValueError(
            f"Features length ({len(df_features)})"
            f"!= labels length ({len(df_labels)})"
        )
    df = df_features.copy()
    df["label"] = df_labels.values

    df = df[df["label"] != 0].reset_index(drop=True)

    return df
