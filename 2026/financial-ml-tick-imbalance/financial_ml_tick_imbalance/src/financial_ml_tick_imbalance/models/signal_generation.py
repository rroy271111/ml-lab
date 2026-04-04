def generate_trade_signals(model, test_df, threshold=0.55):
    X_test = test_df.drop(columns=["label"])

    probs = model.predict_proba(X_test)[:, 1]

    signals = []

    for prob in probs:
        if prob > threshold:
            signals.append(1)
        elif prob < (1 - threshold):
            signals.append(-1)
        else:
            signals.append(0)
    return signals
