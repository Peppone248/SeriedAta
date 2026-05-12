import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.metrics import confusion_matrix


NUM_FEATURES = [
    "xg",
    "xga",
    "poss",
    "sot",
    "shot_accuracy",
    "is_home",
    "strength_points_diff",
    "strength_xg_diff",
    "strength_xga_diff",
    "finishing_efficiency",
    "defensive_efficiency",
    "last_5_points",
    "last_5_goal_diff",
    "last_5_xg",
    "xg_trend",
    "points_trend",
    "days_rest"
]

CAT_FEATURES = [
    "team",
    "opponent",
    "venue"
]


def plot_feature_correlation(df, feature_cols):
    corr_df = df[feature_cols].copy().dropna()

    corr = corr_df.corr()

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5
    )
    plt.title("Correlation Matrix - Classification Features Only")
    plt.show()


def run_eda(df: pd.DataFrame):
    print("\n=== DATA OVERVIEW ===")
    print(df.head())
    print(df.info())

    print("\n=== MISSING VALUES ===")
    print(df.isna().mean().sort_values(ascending=False).head(10))

    print("\n=== TARGET DISTRIBUTION ===")
    sns.countplot(data=df, x="result")
    plt.title("Match Result Distribution")
    plt.show()

    print("\n=== CORRELATION HEATMAP ===")
    plot_feature_correlation(df, NUM_FEATURES)


def clean_data(df: pd.DataFrame):
    df = df.copy()

    # drop rows where target or key features are missing
    df = df.dropna(subset=NUM_FEATURES + CAT_FEATURES + ["result"])

    return df

def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ]
    )


def build_model_pipeline():
    preprocessor = build_preprocessor()

    model = LogisticRegression(
        max_iter=5000
    )

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])


def train_model(X_train, y_train, pipeline):
    param_grid = {
        "model__C": [0.01, 0.1, 1, 5, 10, 50],
        "model__solver": ["lbfgs", "saga"],
        "model__penalty": ["l2"],
        "model__class_weight": [None, "balanced"]
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    return grid


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "log_loss": log_loss(y_test, y_proba),
        "report": classification_report(y_test, y_pred, output_dict=True),
    }

    print(results)

    return {
        "metrics": results,
        "predictions": y_pred,
        "probabilities": y_proba,
        "y_test": y_test
    }


def run_classification_pipeline(df: pd.DataFrame, run_eda_flag=False):

    if run_eda_flag:
        run_eda(df)

    df = clean_data(df)

    X = df[NUM_FEATURES + CAT_FEATURES]
    print(CAT_FEATURES)
    y = df["result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    plot_feature_correlation(df, NUM_FEATURES)
    pipeline = build_model_pipeline()
    grid = train_model(X_train, y_train, pipeline)

    eval_results = evaluate_model(grid, X_test, y_test)

    y_pred = eval_results["predictions"]

    # tables (correct naming)
    prediction_table = build_prediction_table(
        grid.best_estimator_, X_test, y_test
    )

    probability_table = build_probability_table(
        grid.best_estimator_, X_test, y_test
    )

    return {
        "model": grid,
        "metrics": eval_results["metrics"],
        "predictions": y_pred,
        "probabilities": eval_results["probabilities"],
        "y_test": y_test,
        "X_test": X_test,
        "prediction_table": prediction_table,
        "probability_table": probability_table
    }


def build_prediction_table(model, X_test, y_test):
    y_pred = model.predict(X_test)
    df_out = X_test.copy()
    df_out["actual"] = y_test.values
    df_out["predicted"] = y_pred
    return df_out


def build_probability_table(model, X_test, y_test):
    probs = model.predict_proba(X_test)

    df_out = X_test.copy()
    df_out["actual"] = y_test.values

    df_out["P_loss"] = probs[:, 0]
    df_out["P_draw"] = probs[:, 1]
    df_out["P_win"] = probs[:, 2]

    return df_out


def plot_confusion_matrix(y_test, y_pred):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, y_pred, labels=["L", "D", "W"])

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["L", "D", "W"],
        yticklabels=["L", "D", "W"]
    )

    plt.title("Confusion Matrix - Match Outcome Prediction")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()
    plt.show()


def plot_prediction_distribution(y_test, y_pred):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    df_plot = pd.DataFrame({
        "actual": y_test,
        "predicted": y_pred
    })

    plt.figure(figsize=(8, 4))
    sns.countplot(data=df_plot.melt(), x="value", hue="variable")
    plt.title("Actual vs Predicted Distribution")
    plt.show()


def get_feature_importance(model, preprocessor, num_features, cat_features):
    feature_names = (
            num_features +
            list(preprocessor.named_transformers_["cat"]
                 .get_feature_names_out(cat_features))
    )

    coefs = model.coef_

    importance_df = pd.DataFrame(coefs.T, index=feature_names)
    importance_df["abs_mean"] = np.abs(coefs).mean(axis=0)

    return importance_df.sort_values("abs_mean", ascending=False)
