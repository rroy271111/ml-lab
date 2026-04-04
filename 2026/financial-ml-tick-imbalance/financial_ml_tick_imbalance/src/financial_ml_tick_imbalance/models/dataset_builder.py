import pandas as pd


def build_training_dataset(features, labels):
    df_features = pd.DataFrame(features)
    df_labels = pd.Series(labels, name="label")

    df = df_features.copy()
    df["label"] = df_labels

    df = df[df["label"] != 0]

    return df
