import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc
#from src.features.feature_builder import build_features
import __editable___credit_card_fraud_detection_system_0_1_0_finder
# Load data

def train_and_save(df, out_path='models/xgb_model.pkl'):
    df_feat = build_features(df)
    X = df_feat
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    classifier = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05)
    classifier.fit(X_train, y_train, training_set=[(X_test, y_test)], early_stopping_rounds=20, verbose=10)

    joblib.dump(classifier,out_path)
    print("Saved model to ",out_path)


