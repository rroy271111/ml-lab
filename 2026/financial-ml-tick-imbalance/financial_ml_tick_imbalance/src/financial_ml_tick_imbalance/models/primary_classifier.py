from sklearn.linear_model import LogisticRegression


def train_primary_classfier(train_df):
    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"]

    model = LogisticRegression()
    model.fit(X_train, y_train)

    return model
