"""
feature_selection.py — analisi di correlazione e pruning di feature ridondanti.

Flusso tipico:
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
    Calcola la matrice di correlazione di Pearson sulle colonne numeriche richieste.
    Righe con NaN vengono escluse solo per il calcolo (dropna pairwise).
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
    Restituisce tutti i pairwise con |r| >= threshold (triangolo superiore,
    senza la diagonale) come DataFrame ordinato per |r| decrescente.

    Colonne: feature_a, feature_b, pearson_r, abs_r
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
                    "abs_r":     round(abs(float(r)), 4),
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
    threshold: float = 0.85,
) -> list[str]:
    """
    Greedy feature selection basata su correlazione di Pearson.

    Algoritmo:
      Per ogni coppia (a, b) con |r| >= threshold, rimuove la feature
      con la media di |r| più alta verso tutte le altre — ovvero quella
      più "ridondante" rispetto all'intero set.
      Le feature già marcate come da rimuovere vengono saltate.

    Returns:
        Lista ordinata delle feature da mantenere.
    """
    corr_abs  = corr_matrix.abs()
    mean_corr = corr_abs.mean()     # media |r| di ogni feature vs le altre

    to_drop: set[str] = set()

    # itera sulle coppie più correlate prima (ordine decrescente di |r|)
    pairs = find_redundant_pairs(corr_matrix, threshold)

    for _, row in pairs.iterrows():
        fa, fb = row["feature_a"], row["feature_b"]

        if fa in to_drop or fb in to_drop:
            continue

        # rimuove quella con correlazione media più alta (più ridondante)
        dropped = fb if mean_corr[fa] <= mean_corr[fb] else fa
        to_drop.add(dropped)

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
    Heatmap triangolare annotata della matrice di correlazione.

    Segue lo stile triangolare/annotato usato nel progetto.
    Se threshold è fornito, le celle oltre soglia vengono evidenziate
    con un bordo rosso tramite un secondo heatmap sovrapposto.

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
        mask        = mask,
        annot       = True,
        fmt         = ".2f",
        cmap        = "coolwarm",
        center      = 0,
        vmin        = -1,
        vmax        = 1,
        linewidths  = 0.4,
        linecolor   = "white",
        annot_kws   = {"size": annot_fontsize},
        square      = True,
        cbar_kws    = {"shrink": 0.6, "label": "Pearson r"},
        ax          = ax,
    )

    # ── evidenzia coppie oltre soglia ─────────────────────────────────────
    if threshold is not None:
        above = corr_matrix.abs() >= threshold
        highlight_mask = ~(above & ~mask)   # mostra solo le celle rilevanti

        sns.heatmap(
            corr_matrix.where(above & ~mask),   # NaN fuori dalla zona di interesse
            mask        = highlight_mask,
            annot       = True,
            fmt         = ".2f",
            cmap        = "Reds",
            vmin        = threshold,
            vmax        = 1,
            linewidths  = 0.4,
            linecolor   = "white",
            annot_kws   = {"size": annot_fontsize, "weight": "bold", "color": "black"},
            square      = True,
            cbar        = False,
            ax          = ax,
        )

    ax.set_title(
        f"{title}" + (f"  (rosso = |r| ≥ {threshold})" if threshold else ""),
        fontsize=12, pad=14,
    )
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)

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
    threshold: float = 0.85,
) -> list[str]:
    """
    Esegue analisi completa: heatmap + report + pruning.

    Args:
        df:        DataFrame con le feature già costruite
        features:  lista di feature numeriche da analizzare
        label:     etichetta usata nel titolo del plot e nel report
        threshold: soglia |r| per considerare due feature ridondanti

    Returns:
        Lista di feature da mantenere dopo il pruning.
    """
    corr  = compute_correlation_matrix(df, features)
    pairs = find_redundant_pairs(corr, threshold)
    kept  = select_features(features, corr, threshold)

    plot_correlation_heatmap(corr, title=f"Correlation Matrix — {label}", threshold=threshold)
    print_redundancy_report(pairs, features, kept, threshold)

    return kept


# ─── LEAKAGE AUDIT ───────────────────────────────────────────────────────────

def audit_leakage(
    df:               pd.DataFrame,
    features:         list[str],
    target_col:       str   = "result",
    warn_threshold:   float = 0.25,
    danger_threshold: float = 0.45,
) -> pd.DataFrame:
    """
    Controlla la correlazione di Pearson di ogni feature con il target numerico.

    Soglie:
        |r| < warn_threshold    → OK, segnale legittimo
        |r| >= warn_threshold   → WARN, da verificare
        |r| >= danger_threshold → DANGER, quasi certamente leaky

    Feature legittime raramente superano |r| = 0.25–0.30 su questo task.
    Valori oltre 0.45 indicano quasi sempre che la feature usa dati
    della partita che stai cercando di predire (gf, ga, o loro derivate).

    Args:
        df:               DataFrame con feature e target già costruiti
        features:         lista di feature numeriche da auditare
        target_col:       colonna target (default "result")
        warn_threshold:   soglia di attenzione
        danger_threshold: soglia di pericolo (probabile leakage)

    Returns:
        pd.DataFrame con feature, pearson_r, abs_r, status ordinato per abs_r.
    """
    df = df.copy()

    # converti target in numerico — astype(str) gestisce sia object che Categorical
    df["_target_num"] = df[target_col].astype(str).map({"W": 1, "D": 0, "L": -1})

    if df["_target_num"].isna().all():
        raise ValueError(
            f"Nessun valore di '{target_col}' mappato correttamente. "
            f"Valori attesi: W / D / L. "
            f"Valori trovati: {df[target_col].unique().tolist()}"
        )

    # filtra solo le feature presenti nel DataFrame
    available = [f for f in features if f in df.columns]
    missing   = [f for f in features if f not in df.columns]
    if missing:
        print(f"  [WARN] feature non trovate nel DataFrame: {missing}")

    # filtra solo le colonne numeriche — corrwith non supporta categoriche o stringhe
    available_numeric = [
        f for f in available
        if pd.api.types.is_numeric_dtype(df[f])
    ]
    skipped = [f for f in available if f not in available_numeric]
    if skipped:
        print(f"  [INFO] feature non numeriche escluse dall'audit: {skipped}")

    correlations = (
        df[available_numeric]
        .corrwith(df["_target_num"], method="pearson")
        .dropna()
    )

    def _status(abs_r: float) -> str:
        if abs_r >= danger_threshold:
            return "DANGER ⚠"
        if abs_r >= warn_threshold:
            return "WARN   ~"
        return "OK     ✓"

    report = (
        pd.DataFrame({
            "feature":  correlations.index,
            "pearson_r": correlations.values.round(4),
            "abs_r":    correlations.abs().values.round(4),
        })
        .assign(status=lambda d: d["abs_r"].apply(_status))
        .sort_values("abs_r", ascending=False)
        .reset_index(drop=True)
    )

    # ── stampa ────────────────────────────────────────────────────────────
    n_danger = (report["abs_r"] >= danger_threshold).sum()
    n_warn   = ((report["abs_r"] >= warn_threshold) &
                (report["abs_r"] < danger_threshold)).sum()
    n_ok     = (report["abs_r"] < warn_threshold).sum()

    print("\n" + "═" * 65)
    print(f"  LEAKAGE AUDIT — correlazione con '{target_col}'")
    print(f"  soglie: WARN ≥ {warn_threshold}  |  DANGER ≥ {danger_threshold}")
    print("═" * 65)
    print(f"\n  {'feature':<32} {'pearson_r':>10}  {'abs_r':>6}  status")
    print(f"  {'─'*32}  {'─'*9}  {'─'*5}  {'─'*10}")

    for _, row in report.iterrows():
        bar = "█" * int(row["abs_r"] * 30)
        print(f"  {row['feature']:<32} {row['pearson_r']:>+10.4f}  "
              f"{row['abs_r']:>6.4f}  {row['status']}  {bar}")

    print(f"\n  Riepilogo: {n_ok} OK  |  {n_warn} WARN  |  {n_danger} DANGER")

    if n_danger > 0:
        danger_list = report[report["abs_r"] >= danger_threshold]["feature"].tolist()
        print(f"\n  Feature a rischio leakage:")
        for f in danger_list:
            print(f"    ⚠  {f}")
        print(f"\n  Verifica che queste feature non usino gf, ga, "
              f"result o derivate della partita corrente.")

    print("═" * 65)
    return report


# ─── PERMUTATION IMPORTANCE ──────────────────────────────────────────────────

def compute_permutation_importance(
    grid:          object,
    X_test:        pd.DataFrame,
    y_test:        pd.Series,
    n_repeats:     int  = 10,
    random_state:  int  = 42,
    label_encoder: object = None,
) -> pd.DataFrame:
    """
    Calcola la permutation importance sul test set usando il
    best_estimator_ del GridSearch già fittato — nessun retraining.

    Args:
        grid:          GridSearchCV fittato (output delle pipeline)
        X_test:        feature del test set
        y_test:        label reali del test set (stringhe W/D/L)
        n_repeats:     quante volte mescolare ogni feature (default=10)
        random_state:  per riproducibilità
        label_encoder: LabelEncoder usato durante il training.
                       Obbligatorio per XGBoost (predict() ritorna interi).
                       None per Logistic e LightGBM.

    Returns:
        DataFrame con colonne: feature, importance, std, ci_low, ci_high
        ordinato per importance decrescente.
    """
    from sklearn.inspection import permutation_importance

    # XGBoost predict() ritorna interi — allinea y_test al tipo delle predizioni
    y_eval = label_encoder.transform(y_test) if label_encoder is not None else y_test

    result = permutation_importance(
        grid.best_estimator_,
        X_test,
        y_eval,
        n_repeats    = n_repeats,
        scoring      = "f1_macro",
        random_state = random_state,
        n_jobs       = -1,
    )

    # intervallo di confidenza 95% sui n_repeats
    ci = 1.96 * result.importances_std / np.sqrt(n_repeats)

    return (
        pd.DataFrame({
            "feature":    X_test.columns.tolist(),
            "importance": result.importances_mean.round(5),
            "std":        result.importances_std.round(5),
            "ci_low":     (result.importances_mean - ci).round(5),
            "ci_high":    (result.importances_mean + ci).round(5),
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def plot_permutation_importance(
    importance_df: pd.DataFrame,
    threshold:     float = 0.001,
    top_k:         int   = 30,
    title:         str   = "Permutation Importance",
) -> None:
    """
    Barplot orizzontale con errorbar (± std) e linea di soglia.

    Le feature sotto soglia sono colorate in rosso per identificarle
    immediatamente. Le feature con importance negativa sono indicate
    con una nota: confondono il modello più che aiutarlo.

    Args:
        importance_df: output di compute_permutation_importance()
        threshold:     soglia sotto cui la feature è candidata alla rimozione
        top_k:         quante feature mostrare (default=30, mostra tutte
                       se il set è più piccolo)
        title:         titolo del grafico
    """
    df_plot = importance_df.head(top_k).copy()

    # colore per status
    df_plot["color"] = df_plot["importance"].apply(
        lambda x: "#d84a30" if x < threshold else "#3266ad"
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(df_plot) * 0.35)))

    # ── 1. barplot con errorbar ───────────────────────────────────────────
    ax = axes[0]
    y_pos = np.arange(len(df_plot))

    bars = ax.barh(
        y_pos,
        df_plot["importance"],
        color  = df_plot["color"],
        alpha  = 0.85,
        height = 0.65,
    )
    ax.errorbar(
        df_plot["importance"],
        y_pos,
        xerr      = df_plot["std"],
        fmt       = "none",
        color     = "black",
        capsize   = 3,
        linewidth = 1,
        alpha     = 0.6,
    )
    ax.axvline(threshold, color="#d84a30", linestyle="--",
               linewidth=1.2, label=f"soglia={threshold}")
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.4)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot["feature"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Δ F1-macro (permutando la feature)")
    ax.set_title(f"{title}\n(rosso = sotto soglia)")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    # ── 2. scatter importance vs std (stabilità) ─────────────────────────
    ax = axes[1]
    colors = ["#d84a30" if imp < threshold else "#3266ad"
              for imp in df_plot["importance"]]

    ax.scatter(
        df_plot["importance"],
        df_plot["std"],
        c     = colors,
        s     = 80,
        alpha = 0.8,
        zorder= 5,
    )
    for _, row in df_plot.iterrows():
        ax.annotate(
            row["feature"],
            (row["importance"], row["std"]),
            fontsize    = 7,
            textcoords  = "offset points",
            xytext      = (5, 2),
            alpha       = 0.8,
        )
    ax.axvline(threshold, color="#d84a30", linestyle="--",
               linewidth=1.2, alpha=0.7)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("Importance media")
    ax.set_ylabel("Std (instabilità della stima)")
    ax.set_title("Importance vs Instabilità\n"
                 "(alto std = stima poco affidabile)")
    ax.grid(alpha=0.3)

    plt.suptitle(title, fontsize=12, y=1.01)
    plt.tight_layout()
    plt.show()


def run_permutation_audit(
    grid:          object,
    X_test:        pd.DataFrame,
    y_test:        pd.Series,
    threshold:     float  = 0.001,
    n_repeats:     int    = 10,
    top_k:         int    = 30,
    model_name:    str    = "Model",
    label_encoder: object = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Pipeline completa: calcolo → plot → report testuale → suggerimento rimozione.

    Args:
        grid:          GridSearchCV fittato
        X_test:        feature del test set
        y_test:        label reali (stringhe W/D/L)
        threshold:     soglia di importanza minima (default=0.001)
        n_repeats:     ripetizioni per stima stabile (default=10)
        top_k:         feature da mostrare nel plot
        model_name:    etichetta per titoli e report
        label_encoder: LabelEncoder del modello — obbligatorio per XGBoost,
                       None per Logistic e LightGBM.

    Returns:
        (importance_df, features_to_drop)
    """
    print(f"\n  Calcolo permutation importance ({n_repeats} ripetizioni)...")
    importance_df = compute_permutation_importance(
        grid, X_test, y_test,
        n_repeats     = n_repeats,
        label_encoder = label_encoder,
    )

    plot_permutation_importance(
        importance_df,
        threshold = threshold,
        top_k     = top_k,
        title     = f"Permutation Importance — {model_name}",
    )

    # ── criteri di rimozione ──────────────────────────────────────────────
    # 1. importance media sotto soglia
    low_importance = importance_df["importance"] < threshold
    # 2. anche il limite superiore dell'IC è sotto soglia
    #    (la feature non è mai significativa nemmeno nella stima ottimistica)
    low_ci_high    = importance_df["ci_high"] < threshold
    # 3. importance negativa (la feature peggiora il modello)
    negative       = importance_df["importance"] < 0

    to_drop_mask = low_importance & low_ci_high
    to_drop      = importance_df[to_drop_mask]["feature"].tolist()
    negative_lst = importance_df[negative]["feature"].tolist()

    # ── report testuale ───────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print(f"  PERMUTATION IMPORTANCE AUDIT — {model_name}")
    print(f"  soglia: {threshold}  |  repeats: {n_repeats}")
    print("═" * 70)
    print(f"\n  {'feature':<32}  {'importance':>10}  {'±std':>7}  "
          f"{'[ci_low':>8}  {'ci_high]':>8}  status")
    print(f"  {'─'*32}  {'─'*10}  {'─'*7}  {'─'*8}  {'─'*8}  {'─'*10}")

    for _, row in importance_df.iterrows():
        is_drop = row["feature"] in to_drop
        is_neg  = row["importance"] < 0
        status  = "✗ rimuovi" if is_drop else ("~ negativa" if is_neg else "✓ ok")
        bar     = "█" * max(0, int(row["importance"] * 800))

        print(f"  {row['feature']:<32}  {row['importance']:>+10.5f}  "
              f"±{row['std']:>6.5f}  "
              f"[{row['ci_low']:>+7.5f}  {row['ci_high']:>+7.5f}]  "
              f"{status}  {bar}")

    print(f"\n  Riepilogo:")
    print(f"    Feature utili     : "
          f"{len(importance_df) - len(to_drop_mask[to_drop_mask])}")
    print(f"    Da rimuovere      : {len(to_drop)}")
    print(f"    Con impatto neg.  : {len(negative_lst)}")

    if to_drop:
        print(f"\n  Suggerimento — rimuovi da config.py:")
        for f in to_drop:
            marker = " (negativa)" if f in negative_lst else ""
            print(f"    - {f}{marker}")

    if negative_lst:
        extra_neg = [f for f in negative_lst if f not in to_drop]
        if extra_neg:
            print(f"\n  Feature con importance negativa ma IC non conclusivo")
            print(f"  (valuta manualmente con più n_repeats):")
            for f in extra_neg:
                print(f"    ~ {f}")

    print("═" * 70)
    return importance_df, to_drop