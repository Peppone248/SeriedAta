import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shap


# ─────────────────────────────────────────────
# SHAP — IMPORTANZA GLOBALE DELLE FEATURE
# ─────────────────────────────────────────────

def get_feature_names(best_pipeline):
    """Estrae i nomi di tutte le feature dopo il preprocessor."""
    preprocessor = best_pipeline.named_steps["preprocessor"]
    num_features = list(preprocessor.transformers_[0][2])
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_features_raw = preprocessor.transformers_[1][2]
    cat_encoded = list(cat_encoder.get_feature_names_out(cat_features_raw))
    return num_features + cat_encoded


def compute_shap_values(grid, X_test):

    best_pipeline = grid.best_estimator_
    preprocessor = best_pipeline.named_steps["preprocessor"]
    model = best_pipeline.named_steps["model"]

    X_transformed = preprocessor.transform(X_test)
    feature_names = get_feature_names(best_pipeline)

    explainer = shap.KernelExplainer(
        model.predict_proba,
        shap.sample(X_transformed, 100)
    )

    shap_values = explainer.shap_values(X_transformed)

    return shap_values, X_transformed, feature_names


def plot_shap_global_importance(grid, X_test, top_k=20):
    shap_values, X_transformed, feature_names = compute_shap_values(grid, X_test)

    mean_abs = np.abs(shap_values).mean(axis=(0, 2))  # media su campioni e classi

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).head(top_k)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x="mean_abs_shap", y="feature", palette="Blues_r")
    plt.title(f"SHAP global feature importance — top {top_k}")
    plt.xlabel("Mean |SHAP value|")
    plt.tight_layout()
    plt.show()

    return importance_df


def plot_shap_per_class(grid, X_test, class_names=("L", "D", "W"), top_k=15):
    shap_values, X_transformed, feature_names = compute_shap_values(grid, X_test)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

    for i, cls in enumerate(class_names):
        sv_class = shap_values[:, :, i]  # (n_samples, n_features)
        mean_abs = np.abs(sv_class).mean(axis=0)

        df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False).head(top_k)

        sns.barplot(data=df, x="mean_abs_shap", y="feature", ax=axes[i], palette="Blues_r")
        axes[i].set_title(f"Feature → {cls}")
        axes[i].set_xlabel("Mean |SHAP|")
        axes[i].set_ylabel("")

    plt.suptitle("SHAP feature importance per classe (W / D / L)", fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_shap_beeswarm(grid, X_test, class_idx=2, class_name="W", top_k=20):
    shap_values, X_transformed, feature_names = compute_shap_values(grid, X_test)

    shap_exp = shap.Explanation(
        values=shap_values[:, :, class_idx],  # (n_samples, n_features)
        data=X_transformed,
        feature_names=feature_names,
    )

    plt.figure(figsize=(10, 7))
    shap.plots.beeswarm(shap_exp, max_display=top_k, show=False)
    plt.title(f"SHAP beeswarm — classe {class_name}")
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# SPIEGAZIONE SINGOLA PARTITA
# ─────────────────────────────────────────────

def compute_match_shap(grid, X_row, X_background):

    best_pipeline = grid.best_estimator_
    preprocessor = best_pipeline.named_steps["preprocessor"]
    model = best_pipeline.named_steps["model"]

    # transform
    X_row_t = preprocessor.transform(X_row)
    X_bg_t = preprocessor.transform(X_background)

    feature_names = get_feature_names(best_pipeline)

    # KernelExplainer (same logic as global)
    explainer = shap.KernelExplainer(
        model.predict_proba,
        shap.sample(X_bg_t, 100)
    )

    shap_values = explainer.shap_values(X_row_t)

    return shap_values, X_row_t, feature_names, model


def explain_single_match(grid, X_row, X_background, y_true_label, top_k=8):

    shap_values, X_row_t, feature_names, model = compute_match_shap(
        grid, X_row, X_background
    )

    pred = model.predict(X_row_t)[0]
    proba = model.predict_proba(X_row_t)[0]
    class_names = model.classes_

    pred_idx = list(class_names).index(pred)

    # =========================
    # MULTICLASS HANDLING (SAFE)
    # =========================
    if isinstance(shap_values, list):
        sv = shap_values[pred_idx][0]

    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        sv = shap_values[0, :, pred_idx]

    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
        sv = shap_values[0]

    else:
        raise ValueError(f"Unexpected SHAP shape: {type(shap_values)}")

    X_array = X_row_t.toarray() if hasattr(X_row_t, "toarray") else X_row_t

    contrib_df = pd.DataFrame({
        "feature": feature_names,
        "value": X_array[0],
        "shap": sv,
    })

    contrib_df["abs_shap"] = contrib_df["shap"].abs()
    contrib_df = contrib_df.sort_values("abs_shap", ascending=False)

    return {
        "prediction": pred,
        "actual": y_true_label,
        "correct": pred == y_true_label,
        "probabilities": dict(zip(class_names, proba)),
        "top_drivers": contrib_df.head(top_k),
        "all_contributions": contrib_df,
    }


def plot_match_shap_report(explanation, top_k=10):

    df = explanation["all_contributions"].head(top_k)

    # =========================
    # 1. TOP DRIVERS BARPLOT
    # =========================
    plt.figure(figsize=(10, 5))

    sns.barplot(
        data=df,
        x="shap",
        y="feature",
        palette="coolwarm"
    )

    plt.title("Match SHAP Explanation — Top Feature Drivers")
    plt.xlabel("SHAP value (impact on prediction)")
    plt.ylabel("Feature")

    plt.axvline(0, color="black", linewidth=1)
    plt.tight_layout()
    plt.show()

    # =========================
    # 2. POSITIVE vs NEGATIVE IMPACT
    # =========================
    pos = df[df["shap"] > 0]
    neg = df[df["shap"] < 0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.barplot(
        data=pos,
        x="shap",
        y="feature",
        ax=axes[0],
        palette="Greens_r"
    )
    axes[0].set_title("Positive Impact (pushes prediction)")

    sns.barplot(
        data=neg,
        x="shap",
        y="feature",
        ax=axes[1],
        palette="Reds_r"
    )
    axes[1].set_title("Negative Impact (pushes against prediction)")

    plt.tight_layout()
    plt.show()

    # =========================
    # 3. VALUE VS IMPACT SCATTER
    # =========================
    plt.figure(figsize=(8, 5))

    sns.scatterplot(
        data=df,
        x="value",
        y="shap",
        hue="feature",
        s=100
    )

    plt.title("Feature Value vs SHAP Impact (Match Level)")
    plt.axhline(0, color="black", linewidth=1)
    plt.tight_layout()
    plt.show()


def print_match_explanation(explanation):
    print("\n" + "=" * 50)
    print("MATCH EXPLANATION")
    print("=" * 50)

    pred = explanation["prediction"]
    actual = explanation["actual"]
    correct = explanation["correct"]

    print(f"Predizione: {pred}  |  Reale: {actual}  |  {'CORRETTA' if correct else 'SBAGLIATA'}")
    print()

    print("Probabilità:")
    for cls, p in sorted(explanation["probabilities"].items()):
        bar = "█" * int(p * 20)
        print(f"  {cls}: {bar:<20} {p:.3f}")

    print()
    print(f"Top {len(explanation['top_drivers'])} feature (SHAP):")
    df = explanation["top_drivers"][["feature", "value", "shap"]].copy()
    df["direzione"] = df["shap"].apply(lambda x: "+" if x > 0 else "-")
    print(df.to_string(index=False))


# ─────────────────────────────────────────────
# DISTRIBUZIONE CLASSI
# ─────────────────────────────────────────────

def plot_class_distribution(y_test, y_pred):
    df = pd.DataFrame({"actual": y_test, "predicted": y_pred})

    plt.figure(figsize=(8, 4))
    sns.countplot(data=df.melt(), x="value", hue="variable",
                  order=["L", "D", "W"])
    plt.title("Actual vs Predicted distribution (W / D / L)")
    plt.tight_layout()
    plt.show()
