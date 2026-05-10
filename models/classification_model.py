from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pandas as pd
import numpy as np


def build_classification_dataset(df: pd.DataFrame, features, target="result"):
    df = df.dropna(subset=features + [target])

    X = df[features].copy()
    y = df[target].copy()

    return X, y


def split_classification_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(X_train, y_train)
    return model


def evaluate_classification(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "report": classification_report(y_test, y_pred, output_dict=True),
    }

    return {
        "metrics": metrics,
        "predictions": y_pred,
        "probabilities": y_proba,
        "y_test": y_test
    }