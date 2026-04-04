def temporal_train_test_split(df, split_ratio=0.7):
    split_idx = int(len(df) * split_ratio)

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    return train_df, test_df
