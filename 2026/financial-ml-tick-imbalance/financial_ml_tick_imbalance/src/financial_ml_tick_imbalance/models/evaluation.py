from sklearn.metrics import classification_report


def evaluate_classifier(model, test_df):
    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"]

    preds = model.predict(X_test)

    return classification_report(y_test, preds)
