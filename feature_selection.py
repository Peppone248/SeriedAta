"""
Workflow:
    corr   = compute_correlation_matrix(df, NUM_FEATURES)
    pairs  = find_redundant_pairs(corr, threshold=0.85)
    kept   = select_features(NUM_FEATURES, corr, threshold=0.85)
    plot_correlation_heatmap(corr, title="Logistic features")
    print_redundancy_report(pairs, NUM_FEATURES, kept)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ─── ANALISI ─────────────────────────────────────────────────────────────────

def compute_correlation_matrix(
        df: pd.DataFrame,
        features: list[str],
) -> pd.DataFrame:
    """
    Compute matrix correlation based on Pearson on numeric columns features.
    Excluded features which contain NaN.
    """
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Feature non presenti nel DataFrame: {missing}")

    return df[features].corr(method="pearson")


def find_redundant_pairs(
        corr_matrix: pd.DataFrame,
        threshold: float = 0.85,
) -> pd.DataFrame:
    """
    Return all pairwise con |r| >= threshold as DataFrame sorted by |r| desc.

    Columns: feature_a, feature_b, pearson_r, abs_r
    """
    features = corr_matrix.columns.tolist()
    rows = []

    for i, fa in enumerate(features):
        for fb in features[i + 1:]:
            r = corr_matrix.loc[fa, fb]
            if abs(r) >= threshold:
                rows.append({
                    "feature_a": fa,
                    "feature_b": fb,
                    "pearson_r": round(float(r), 4),
                    "abs_r": round(abs(float(r)), 4),
                })

    if not rows:
        return pd.DataFrame(columns=["feature_a", "feature_b", "pearson_r", "abs_r"])

    return (
        pd.DataFrame(rows)
            .sort_values("abs_r", ascending=False)
            .reset_index(drop=True)
    )


# ─── PRUNING ─────────────────────────────────────────────────────────────────

def select_features(
        features: list[str],
        corr_matrix: pd.DataFrame,
        df: pd.DataFrame,
        target: pd.Series,
        corr_threshold: float = 0.90,
        mi_threshold: float = 0.01,
) -> list[str]:
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.preprocessing import LabelEncoder

    y_enc = LabelEncoder().fit_transform(target)
    mi_scores = mutual_info_classif(
        df[features].fillna(0), y_enc, random_state=42
    )
    mi = dict(zip(features, mi_scores))

    corr_abs = corr_matrix.abs()
    mean_corr = corr_abs.mean()
    to_drop = set()

    pairs = find_redundant_pairs(corr_matrix, corr_threshold)

    for _, row in pairs.iterrows():
        fa, fb = row["feature_a"], row["feature_b"]
        if fa in to_drop or fb in to_drop:
            continue

        # rimuove solo se la feature ha anche basso segnale sul target
        # se entrambe hanno MI alta, le teniamo entrambe e lasciamo
        # la regolarizzazione L2 / tree splits gestire la ridondanza
        if mi[fa] < mi_threshold and mi[fb] < mi_threshold:
            dropped = fb if mean_corr[fa] <= mean_corr[fb] else fa
            to_drop.add(dropped)
        elif mi[fa] < mi_threshold:
            to_drop.add(fa)
        elif mi[fb] < mi_threshold:
            to_drop.add(fb)
        # altrimenti: entrambe hanno segnale → non rimuovere nessuna

    return [f for f in features if f not in to_drop]


# ─── PLOT ────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(
        corr_matrix: pd.DataFrame,
        title: str = "Feature Correlation Matrix",
        threshold: float | None = 0.85,
        figsize: tuple[int, int] = (12, 10),
        annot_fontsize: int = 7,
) -> None:
    """
    Annotated triangular heatmap of the correlation matrix.

    This follows the triangular/annotated style used in the project.
    If a threshold is provided, cells exceeding the threshold are highlighted
    with a red border using a second, overlaid heatmap.

    Args:
        corr_matrix:    output di compute_correlation_matrix()
        title:          titolo del grafico
        threshold:      se non None, evidenzia le coppie con |r| >= threshold
        figsize:        dimensioni figura
        annot_fontsize: dimensione font delle annotazioni
    """
    n = len(corr_matrix)

    # maschera triangolo superiore (inclusa diagonale)
    mask = np.triu(np.ones((n, n), dtype=bool))

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.4,
        linecolor="white",
        annot_kws={"size": annot_fontsize},
        square=True,
        cbar_kws={"shrink": 0.6, "label": "Pearson r"},
        ax=ax,
    )

    # ── evidenzia coppie oltre soglia ─────────────────────────────────────
    if threshold is not None:
        above = corr_matrix.abs() >= threshold
        highlight_mask = ~(above & ~mask)  # mostra solo le celle rilevanti

        sns.heatmap(
            corr_matrix.where(above & ~mask),  # NaN fuori dalla zona di interesse
            mask=highlight_mask,
            annot=True,
            fmt=".2f",
            cmap="Reds",
            vmin=threshold,
            vmax=1,
            linewidths=0.4,
            linecolor="white",
            annot_kws={"size": annot_fontsize, "weight": "bold", "color": "black"},
            square=True,
            cbar=False,
            ax=ax,
        )

    ax.set_title(
        f"{title}" + (f"  (rosso = |r| ≥ {threshold})" if threshold else ""),
        fontsize=12, pad=14,
    )
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)

    plt.tight_layout()
    plt.show()


# ─── REPORT TESTUALE ─────────────────────────────────────────────────────────

def print_redundancy_report(
        pairs: pd.DataFrame,
        original_features: list[str],
        kept_features: list[str],
        threshold: float = 0.85,
) -> None:
    """
    Stampa un riepilogo leggibile dell'analisi di ridondanza.
    """
    dropped = [f for f in original_features if f not in kept_features]

    print("\n" + "═" * 60)
    print(f"  FEATURE REDUNDANCY REPORT  (soglia |r| ≥ {threshold})")
    print("═" * 60)

    if pairs.empty:
        print("  Nessuna coppia ridondante trovata.")
    else:
        print(f"\n  Coppie con |r| ≥ {threshold}:\n")
        for _, row in pairs.iterrows():
            bar = "█" * int(row["abs_r"] * 20)
            print(f"  {row['feature_a']:<30} ↔  {row['feature_b']:<30} "
                  f"r={row['pearson_r']:+.3f}  {bar}")

    print(f"\n  Feature originali : {len(original_features)}")
    print(f"  Feature mantenute : {len(kept_features)}")
    print(f"  Feature rimosse   : {len(dropped)}")

    if dropped:
        print(f"\n  Da rimuovere (suggerimento):")
        for f in dropped:
            print(f"    - {f}")

    print("\n  Feature mantenute:")
    for f in kept_features:
        marker = "  " if f in kept_features else "✗ "
        print(f"    {marker}{f}")

    print("═" * 60)


# ─── ENTRY POINT (usabile standalone) ────────────────────────────────────────

def run_feature_analysis(
        df: pd.DataFrame,
        features: list[str],
        label: str = "features",
        corr_threshold: float = 0.90,
        mi_threshold: float = 0.01,
) -> list[str]:
    """
    Esegue analisi completa: heatmap + report + pruning.

    Args:
        df:        DataFrame con le feature già costruite
        features:  lista di feature numeriche da analizzare
        label:     etichetta usata nel titolo del plot e nel report
        corr_threshold: soglia |r| per considerare due feature ridondanti
        mi_threshold: soglia per considerare valida la mutual information tra le features

    Returns:
        Lista di feature da mantenere dopo il pruning.
    """
    corr = compute_correlation_matrix(df, features)
    pairs = find_redundant_pairs(corr, corr_threshold)
    kept = select_features(features, corr, df, df["result"], corr_threshold, mi_threshold)

    plot_correlation_heatmap(corr, title=f"Correlation Matrix — {label}", threshold=corr_threshold)
    print_redundancy_report(pairs, features, kept, corr_threshold)

    return kept
