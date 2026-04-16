def generate_trade_signals(model, test_df, threshold=0.55):
    X_test = test_df.drop(columns=["label"])

    probs = model.predict_proba(X_test)
    classes = model.classes_

    signals = []

    for prob_row in probs:
        best_idx = prob_row.argmax()
        best_class = classes[best_idx]
        confidence = prob_row[best_idx]

        if confidence > threshold:
            signals.append(int(best_class))
        else:
            signals.append(0)
    return signals
