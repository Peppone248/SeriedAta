import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_feature_importance(pipeline, feature_names):
    """
    Global feature importance from Logistic Regression coefficients
    """

    model = pipeline.named_steps["model"]

    coefs = model.coef_

    importance = np.mean(np.abs(coefs), axis=0)

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values("importance", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="importance", y="feature")
    plt.title("Feature Importance (Logistic Regression)")
    plt.tight_layout()
    plt.show()

    return df


def plot_class_distribution(y_test, y_pred):
    df = pd.DataFrame({
        "actual": y_test,
        "predicted": y_pred
    })

    plt.figure(figsize=(8, 4))
    sns.countplot(data=df.melt(), x="value", hue="variable")
    plt.title("Actual vs Predicted Distribution (W / D / L)")
    plt.tight_layout()
    plt.show()


def plot_probability_distribution(proba):
    """
    Shows how confident the model is overall
    """

    df = pd.DataFrame(proba, columns=["Loss", "Draw", "Win"])

    plt.figure(figsize=(10, 5))
    sns.kdeplot(df["Win"], label="Win", fill=True)
    sns.kdeplot(df["Draw"], label="Draw", fill=True)
    sns.kdeplot(df["Loss"], label="Loss", fill=True)

    plt.title("Predicted Probability Distributions")
    plt.legend()
    plt.show()


def explain_single_match(model_pipeline, X_row, feature_names, top_k=5):
    """
    Explain a single match prediction
    """

    model = model_pipeline.best_estimator_.named_steps["model"]
    preprocessor = model_pipeline.best_estimator_.named_steps["preprocessor"]

    # transform row
    X_transformed = preprocessor.transform(X_row)

    # prediction + probabilities
    pred = model.predict(X_transformed)[0]
    proba = model.predict_proba(X_transformed)[0]

    class_names = model.classes_

    # coefficients
    coefs = model.coef_

    # compute contributions
    contributions = coefs * X_transformed.toarray()

    contributions_sum = contributions[0]

    feature_contrib = pd.DataFrame({
        "feature": feature_names,
        "value": X_transformed.toarray()[0],
        "contribution": contributions_sum
    })

    # keep only active / meaningful features
    feature_contrib = feature_contrib[
        feature_contrib["value"] != 0
        ]

    feature_contrib["abs"] = np.abs(feature_contrib["contribution"])
    feature_contrib = feature_contrib.sort_values("abs", ascending=False)

    return {
        "prediction": pred,
        "probabilities": dict(zip(class_names, proba)),
        "top_positive": feature_contrib.head(top_k),
        "top_negative": feature_contrib.tail(top_k)
    }


def print_match_explanation(explanation):

    print("\n================ MATCH EXPLANATION ================")

    print("\nPrediction:", explanation["prediction"])

    print("\nProbabilities:")
    for k, v in explanation["probabilities"].items():
        print(f"   {k}: {v:.3f}")

    print("\nStrongest positive drivers:")
    print(explanation["top_positive"][["feature", "contribution"]])

    print("\nWeakening factors:")
    print(explanation["top_negative"][["feature", "contribution"]])


def plot_feature_importance_per_class(pipeline):
    best_pipeline = pipeline.best_estimator_

    model = best_pipeline.named_steps["model"]
    preprocessor = best_pipeline.named_steps["preprocessor"]

    coefs = model.coef_

    class_names = list(model.classes_)

    # safety check
    if len(class_names) != coefs.shape[0]:
        raise ValueError(
            f"Class mismatch: {class_names} vs coef shape {coefs.shape}"
        )

    num_features = preprocessor.transformers_[0][2]
    cat_features = preprocessor.transformers_[1][2]

    cat_encoded = preprocessor.named_transformers_["cat"].get_feature_names_out(cat_features)

    feature_names = np.concatenate([num_features, cat_encoded])

    fig, axes = plt.subplots(1, len(class_names), figsize=(18, 6), sharey=True)

    for i, class_name in enumerate(class_names):
        class_coef = coefs[i]

        df = pd.DataFrame({
            "feature": feature_names,
            "importance": class_coef
        }).sort_values("importance", ascending=False)

        sns.barplot(
            data=df,
            x="importance",
            y="feature",
            ax=axes[i]
        )

        axes[i].set_title(f"Feature Impact → {class_name}")
        axes[i].axvline(0, color="black", linewidth=1)

    plt.tight_layout()
    plt.show()