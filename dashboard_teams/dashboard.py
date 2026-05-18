"""
dashboard.py — Serie A Analytics Dashboard  (v2)
Avvia con: streamlit run dashboard.py

Usa automaticamente i file in data/processed/ se presenti.
"""

import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Serie A Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "primary": "#3266ad",
    "secondary": "#d84a30",
    "green": "#5ab27a",
    "amber": "#d48a2b",
    "purple": "#7c5cbf",
    "teal": "#2aacab",
    "gray": "#8a9bb0",
}

TEAM_PALETTE = [
    "#3266ad", "#d84a30", "#5ab27a", "#d48a2b", "#7c5cbf",
    "#2aacab", "#e06aa0", "#7b8c3e", "#a84f1c", "#1a6b5a",
    "#e5971a", "#4a90d9", "#c0392b", "#27ae60", "#8e44ad",
    "#16a085", "#d35400", "#2c3e50", "#f39c12", "#1abc9c",
    "#e74c3c", "#3498db", "#2ecc71", "#f1c40f", "#9b59b6",
    "#1abc9c", "#34495e", "#e67e22", "#95a5a6",
]

PROCESSED_DIR = Path("data/processed")


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def _find(filename):
    candidates = [
        PROCESSED_DIR / filename,
        Path(filename),
        Path("..") / "data" / "processed" / filename,
    ]
    return next((p for p in candidates if p.exists()), None)


@st.cache_data
def load_raw(csv_path):
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df["points"] = np.select(
        [df["result"] == "W", df["result"] == "D"], [3, 1], default=0
    )
    df["win_flag"] = (df["result"] == "W").astype(int)
    df["draw_flag"] = (df["result"] == "D").astype(int)
    df["loss_flag"] = (df["result"] == "L").astype(int)
    df["goal_diff"] = df["gf"] - df["ga"]
    df["shot_accuracy"] = np.where(df["sh"] > 0, df["sot"] / df["sh"], np.nan)
    df["matchweek"] = df["round"].str.extract(r"(\d+)").astype(float)
    return df


@st.cache_data
def load_processed():
    out = {}
    files = {
        "team_stats": "team_stats.csv",
        "team_season_stats": "team_season_stats.csv",
        "team_by_venue": "team_by_venue.csv",
        "venue_merged": "venue_merged.csv",
        "title_race": "title_race.csv",
        "season_champions": "season_champions.csv",
        "daily_stats": "daily_stats.csv",
        "day_stats_matches": "day_stats_matches.csv",
        "match_df": "match_df.csv",
        "top_goal_diff": "stats_summary__top_goal_diff.csv",
        "top_xg_matches": "stats_summary__top_matches_by_xg_diff.csv",
    }
    for key, fname in files.items():
        p = _find(fname)
        if p:
            out[key] = pd.read_csv(p)
    p_json = _find("stats_summary.json")
    if p_json:
        out["stats_summary"] = json.loads(p_json.read_text())
    if "team_season_stats" in out:
        tss = out["team_season_stats"]
        if "points" in tss.columns and "total_points" not in tss.columns:
            tss = tss.rename(columns={"points": "total_points"})
        out["team_season_stats"] = tss
    return out


@st.cache_data
def build_ts(df):
    ts = (
        df.groupby(["season", "team"])
            .agg(
            total_points=("points", "sum"),
            avg_xg=("xg", "mean"),
            avg_xga=("xga", "mean"),
            avg_poss=("poss", "mean"),
            avg_gf=("gf", "mean"),
            avg_ga=("ga", "mean"),
            avg_goal_diff=("goal_diff", "mean"),
            avg_sot=("sot", "mean"),
            avg_shot_acc=("shot_accuracy", "mean"),
            wins=("win_flag", "sum"),
            draws=("draw_flag", "sum"),
            losses=("loss_flag", "sum"),
            matches=("points", "count"),
        )
            .reset_index()
    )
    ts["win_rate"] = (ts["wins"] / ts["matches"] * 100).round(1)
    ts["draw_rate"] = (ts["draws"] / ts["matches"] * 100).round(1)
    ts["rank"] = ts.groupby("season")["total_points"].rank(ascending=False, method="min").astype(int)
    return ts


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

def sidebar(df):
    st.sidebar.title("⚽ Serie A Analytics")
    st.sidebar.caption("Dashboard to explode a Serie A datasets")
    st.sidebar.markdown("---")

    all_seasons = sorted(df["season"].unique().tolist())
    all_teams = sorted(df["team"].unique().tolist())
    team_colors = {t: TEAM_PALETTE[i % len(TEAM_PALETTE)] for i, t in enumerate(all_teams)}

    st.sidebar.markdown("### Filtri globali")
    sel_seasons = st.sidebar.multiselect(
        "Stagioni", all_seasons,
        default=[s for s in all_seasons if s != 2025],
    )
    sel_teams = st.sidebar.multiselect(
        "Squadre (multi-team)", all_teams,
        default=["Internazionale", "Milan", "Juventus", "Napoli", "Atalanta"],
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(f"{len(df)} righe · {df['team'].nunique()} squadre · {df['season'].nunique()} stagioni")
    return sel_seasons, sel_teams, team_colors


# ─────────────────────────────────────────────
# TAB 1 — PANORAMICA
# ─────────────────────────────────────────────

def tab_overview(ts, p, sel_seasons):
    st.header("📊 Panoramica stagioni")

    if "stats_summary" in p:
        ss = p["stats_summary"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Media spettatori", f"{ss['attendance_mean']:,.0f}")
        c2.metric("Mediana spettatori", f"{ss['attendance_median']:,.0f}")
        c3.metric("IQR possesso", f"{ss['iqr_poss']:.0f}%")
        c4.metric("IQR xG", f"{ss['iqr_xg']:.1f}")
        st.markdown("---")

    season = st.selectbox("Stagione", sorted(sel_seasons, reverse=True), key="ov_s")
    sort_k = st.selectbox("Ordina per", [
        ("total_points", "Punti totali"), ("avg_xg", "xG medio"),
        ("win_rate", "Win rate %"), ("avg_poss", "Possesso"),
        ("avg_goal_diff", "Goal diff"),
    ], format_func=lambda x: x[1], key="ov_sort")
    mk, ml = sort_k

    data = ts[ts["season"] == season].sort_values(mk, ascending=False)
    champ = data.iloc[0]
    top_xg = data.loc[data["avg_xg"].idxmax()]
    top_def = data.loc[data["avg_xga"].idxmin()]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🏆 Campione", champ["team"], f"{int(champ['total_points'])} punti")
    c2.metric("⚡ Miglior attacco", top_xg["team"], f"{top_xg['avg_xg']:.2f} xG/match")
    c3.metric("🔒 Miglior difesa", top_def["team"], f"{top_def['avg_xga']:.2f} xGA/match")
    c4.metric("⚽ Gol/match", f"{data['avg_gf'].mean():.2f}", f"stagione {season}")

    fig = go.Figure()
    fig.add_trace(go.Bar(y=data["team"], x=data["total_points"], name="Punti",
                         orientation="h", marker_color=COLORS["primary"], opacity=0.9))
    fig.add_trace(go.Bar(y=data["team"], x=data["avg_xg"] * 10, name="xG ×10",
                         orientation="h", marker_color=COLORS["green"], opacity=0.7))
    fig.add_trace(go.Bar(y=data["team"], x=data["avg_poss"], name="Possesso",
                         orientation="h", marker_color=COLORS["amber"], opacity=0.7))
    fig.update_layout(barmode="group", height=max(500, len(data) * 32),
                      title=f"Classifica {season}",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02),
                      margin=dict(l=130, r=20, t=60, b=40))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Tabella completa"):
        st.dataframe(
            data[["team", "rank", "total_points", "wins", "draws", "losses",
                  "win_rate", "avg_xg", "avg_xga", "avg_poss"]].round(2),
            use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# TAB 2 — TITLE RACE
# ─────────────────────────────────────────────

def tab_title_race(p):
    st.header("🏆 Title Race")

    if "title_race" not in p:
        st.info("title_race.csv non trovato in data/processed/");
        return

    tr = p["title_race"]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=tr["season"].astype(str), y=tr["champion_points"],
                         name="Campione", marker_color=COLORS["amber"],
                         text=tr["champion_team"], textposition="inside"))
    fig.add_trace(go.Bar(x=tr["season"].astype(str), y=tr["second_place_points"],
                         name="2° posto", marker_color=COLORS["gray"],
                         text=tr["second_place_team"], textposition="inside"))
    fig.add_trace(go.Scatter(x=tr["season"].astype(str), y=tr["title_margin"],
                             name="Margine", mode="lines+markers+text",
                             line=dict(color=COLORS["secondary"], width=2, dash="dot"),
                             text=tr["title_margin"].astype(str), textposition="top center",
                             yaxis="y2"))
    fig.update_layout(
        barmode="group", height=400,
        title="Punti campione vs 2° posto · linea = margine titolo",
        yaxis=dict(title="Punti"),
        yaxis2=dict(title="Margine", overlaying="y", side="right", range=[0, 25]),
        legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    # tabella
    tr_d = tr[["season", "champion_team", "champion_points", "second_place_team",
               "second_place_points", "title_margin", "champion_goal_diff"]].copy()
    tr_d.columns = ["Stagione", "Campione", "Punti", "2° posto", "Punti 2°", "Margine", "GD camp."]

    def hl(val):
        if isinstance(val, (int, float)):
            if val <= 2:  return "background-color:#ffeeba"
            if val >= 15: return "background-color:#d4edda"
        return ""

    st.dataframe(tr_d.style.applymap(hl, subset=["Margine"]),
                 use_container_width=True, hide_index=True)

    # titoli per squadra
    cc = tr["champion_team"].value_counts().reset_index()
    cc.columns = ["team", "titles"]
    fig2 = px.bar(cc, x="team", y="titles", color="titles",
                  color_continuous_scale="YlOrRd",
                  title="Titoli vinti nel dataset",
                  labels={"team": "", "titles": "Titoli"})
    fig2.update_layout(height=280, coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────
# TAB 3 — HOME / AWAY
# ─────────────────────────────────────────────

def tab_home_away(p):
    st.header("🏟️ Home vs Away")

    if "venue_merged" not in p:
        st.info("venue_merged.csv non trovato.");
        return

    vm = p["venue_merged"].copy().sort_values("win_rate_diff", ascending=False)

    c1, c2 = st.columns(2)
    with c1:
        colors = [COLORS["green"] if x > 0 else COLORS["secondary"] for x in vm["win_rate_diff"]]
        fig = go.Figure(go.Bar(
            x=vm["win_rate_diff"], y=vm["team"], orientation="h",
            marker_color=colors,
            text=[f"{v:+.3f}" for v in vm["win_rate_diff"]], textposition="outside",
        ))
        fig.update_layout(
            title="Δ Win rate (casa − trasferta)",
            height=560, xaxis_title="Δ Win rate",
            xaxis=dict(zeroline=True, zerolinecolor="black", zerolinewidth=1),
            margin=dict(l=120, r=60))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = px.scatter(
            vm, x="home_avg_points", y="away_avg_points", text="team",
            color="win_rate_diff", color_continuous_scale="RdYlGn",
            labels={"home_avg_points": "Punti/match casa",
                    "away_avg_points": "Punti/match trasferta"},
            title="Casa vs trasferta — sopra la diagonale = meglio in trasferta",
        )
        mx = max(vm["home_avg_points"].max(), vm["away_avg_points"].max()) + 0.1
        fig2.add_shape(type="line", x0=0, y0=0, x1=mx, y1=mx,
                       line=dict(color="gray", dash="dot", width=1))
        fig2.update_traces(textposition="top center", textfont_size=9)
        fig2.update_layout(height=560, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Tabella dettaglio")
    vm_s = vm[["team", "home_avg_points", "away_avg_points", "home_win_rate",
               "away_win_rate", "win_rate_diff", "home_avg_goals_for", "away_avg_goals_for"]].round(3)
    vm_s.columns = ["Squadra", "Punti/match Casa", "Punti/match Trasferta",
                    "Win rate Casa", "Win rate Trasferta", "Δ Win rate",
                    "Gol/match Casa", "Gol/match Trasferta"]
    st.dataframe(vm_s, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# TAB 4 — STATISTICHE STORICHE SQUADRE
# ─────────────────────────────────────────────

def tab_team_stats(p):
    st.header("📋 Statistiche storiche squadre")

    if "team_stats" not in p:
        st.info("team_stats.csv non trovato.");
        return

    ts_g = p["team_stats"].copy().sort_values("avg_points", ascending=False)

    best = ts_g.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Più forte (pts/match)", best["team"], f"{best['avg_points']:.2f}")
    c2.metric("Miglior win rate", ts_g.loc[ts_g["win_rate"].idxmax(), "team"],
              f"{ts_g['win_rate'].max() * 100:.1f}%")
    c3.metric("Miglior clean sheet", ts_g.loc[ts_g["clean_sheet_rate"].idxmax(), "team"],
              f"{ts_g['clean_sheet_rate'].max() * 100:.1f}%")
    c4.metric("Miglior shot accuracy", ts_g.loc[ts_g["avg_shot_accuracy"].idxmax(), "team"],
              f"{ts_g['avg_shot_accuracy'].max() * 100:.1f}%")

    st.markdown("---")
    metric = st.selectbox("Metrica", [
        ("avg_points", "Punti medi/match"),
        ("win_rate", "Win rate"),
        ("avg_xg", "xG medio"),
        ("avg_xga", "xGA medio"),
        ("clean_sheet_rate", "Clean sheet rate"),
        ("avg_shot_accuracy", "Shot accuracy"),
        ("avg_goals_for", "Gol segnati medi"),
        ("avg_goals_against", "Gol subiti medi"),
    ], format_func=lambda x: x[1], key="ts_m")
    mk, ml = metric

    asc = mk in ("avg_xga", "avg_goals_against")
    data = ts_g.sort_values(mk, ascending=asc)
    col = COLORS["secondary"] if asc else COLORS["primary"]

    fig = go.Figure(go.Bar(
        x=data[mk], y=data["team"], orientation="h",
        marker_color=col, text=data[mk].round(3), textposition="outside",
    ))
    fig.update_layout(title=f"{ml} — storico 2020–2025",
                      height=max(500, len(data) * 28), margin=dict(l=130, r=70))
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.scatter(
        ts_g, x="avg_xg", y="clean_sheet_rate",
        size="win_rate", text="team",
        color="avg_points", color_continuous_scale="Blues",
        labels={"avg_xg": "xG medio", "clean_sheet_rate": "Clean sheet rate",
                "win_rate": "Win rate", "avg_points": "Punti/match"},
        title="Attacco vs difesa — bubble size = win rate",
    )
    fig2.update_traces(textposition="top center", textfont_size=9)
    fig2.update_layout(height=420, coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────
# TAB 5 — PROGRESSIONE PUNTI
# ─────────────────────────────────────────────

def tab_progression(df, ts, sel_seasons, sel_teams):
    st.header("📈 Progressione punti")

    c1, c2 = st.columns([2, 1])
    with c1:
        teams = st.multiselect("Squadre", sorted(df["team"].unique()),
                               default=sel_teams[:5], key="pr_t")
    with c2:
        seasons = st.multiselect("Stagioni", sorted(sel_seasons, reverse=True),
                                 default=[max(s for s in sel_seasons if s != 2025)],
                                 key="pr_s")

    if not teams or not seasons:
        st.info("Seleziona almeno una squadra e una stagione.");
        return

    df_f = df[df["team"].isin(teams) & df["season"].isin(seasons)].copy()
    df_f = df_f.sort_values(["team", "season", "matchweek"])
    df_f["cum_points"] = df_f.groupby(["team", "season"])["points"].cumsum()
    df_f["label"] = df_f["team"] + " " + df_f["season"].astype(str)

    fig = px.line(df_f, x="matchweek", y="cum_points", color="label",
                  color_discrete_sequence=TEAM_PALETTE,
                  labels={"matchweek": "Giornata", "cum_points": "Punti cumulativi", "label": ""},
                  title="Progressione punti cumulativi")
    fig.update_layout(height=420, legend=dict(orientation="h", yanchor="bottom", y=1.02))
    fig.update_traces(line_width=2)
    st.plotly_chart(fig, use_container_width=True)

    final = (ts[ts["team"].isin(teams) & ts["season"].isin(seasons)]
             [["team", "season", "total_points", "rank", "wins", "draws", "losses"]]
             .sort_values(["season", "total_points"], ascending=[True, False]))
    st.dataframe(final, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# TAB 6 — xG E STILE
# ─────────────────────────────────────────────

def tab_xg(ts, sel_seasons, team_colors):
    st.header("⚡ xG e stile di gioco")

    season = st.selectbox("Stagione", sorted(sel_seasons, reverse=True), key="xg_s")
    data = ts[ts["season"] == season].copy()

    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(data, x="avg_xg", y="avg_xga", text="team",
                         color="total_points", color_continuous_scale="Blues",
                         size="total_points", size_max=20,
                         labels={"avg_xg": "xG creato", "avg_xga": "xGA concesso", "total_points": "Punti"},
                         title=f"Attacco vs Difesa — {season}")
        fig.add_hline(y=data["avg_xga"].mean(), line_dash="dot", line_color="gray", opacity=0.4)
        fig.add_vline(x=data["avg_xg"].mean(), line_dash="dot", line_color="gray", opacity=0.4)
        fig.update_traces(textposition="top center", textfont_size=9)
        fig.update_layout(height=400, yaxis_autorange="reversed", coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = px.scatter(data, x="avg_poss", y="win_rate", text="team",
                          color="avg_xg", color_continuous_scale="Oranges",
                          labels={"avg_poss": "Possesso %", "win_rate": "Win rate %", "avg_xg": "xG"},
                          title=f"Possesso vs Win rate — {season}")
        fig2.update_traces(textposition="top center", textfont_size=9)
        fig2.update_layout(height=400, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    team = st.selectbox("Profilo tattico", sorted(data["team"].unique()), key="xg_r")
    row = data[data["team"] == team].iloc[0]

    def norm(col):
        mn = ts[ts["season"] == season][col].min()
        mx = ts[ts["season"] == season][col].max()
        return float((row[col] - mn) / (mx - mn + 1e-9))

    cats = ["xG creato", "Difesa", "Possesso", "Shot accuracy", "Win rate"]
    vals = [norm("avg_xg"), 1 - norm("avg_xga"), norm("avg_poss"), norm("avg_shot_acc"), norm("win_rate")]
    fig3 = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]], theta=cats + [cats[0]],
        fill="toself", fillcolor=COLORS["primary"], opacity=0.35,
        line_color=COLORS["primary"], line_width=2,
    ))
    fig3.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                       title=f"Profilo tattico — {team} {season}", height=360, showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)


# ─────────────────────────────────────────────
# TAB 7 — CONFRONTO
# ─────────────────────────────────────────────

def tab_compare(ts, df, sel_seasons):
    st.header("🆚 Confronto squadre")

    all_t = sorted(ts["team"].unique())
    c1, c2 = st.columns(2)
    with c1:
        ta = st.selectbox("Squadra A", all_t,
                          index=all_t.index("Milan") if "Milan" in all_t else 0, key="ca")
    with c2:
        tb = st.selectbox("Squadra B", all_t,
                          index=all_t.index("Internazionale") if "Internazionale" in all_t else 1, key="cb")

    metric = st.selectbox("Metrica", [
        ("total_points", "Punti totali"), ("avg_xg", "xG medio"),
        ("win_rate", "Win rate %"), ("avg_poss", "Possesso %"), ("avg_goal_diff", "Goal diff"),
    ], format_func=lambda x: x[1], key="cm")
    mk, ml = metric

    sf = sorted(sel_seasons)
    da = ts[(ts["team"] == ta) & ts["season"].isin(sf)].sort_values("season")
    db = ts[(ts["team"] == tb) & ts["season"].isin(sf)].sort_values("season")
    shr = set(da["season"]) & set(db["season"])
    avg_a = da[da["season"].isin(shr)][mk].mean()
    avg_b = db[db["season"].isin(shr)][mk].mean()
    w_a = sum(da[da["season"] == s][mk].values[0] > db[db["season"] == s][mk].values[0]
              for s in shr if len(da[da["season"] == s]) > 0 and len(db[db["season"] == s]) > 0)
    w_b = sum(db[db["season"] == s][mk].values[0] > da[da["season"] == s][mk].values[0]
              for s in shr if len(da[da["season"] == s]) > 0 and len(db[db["season"] == s]) > 0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Media {ml} — {ta}", f"{avg_a:.1f}")
    c2.metric(f"Media {ml} — {tb}", f"{avg_b:.1f}")
    c3.metric(f"Stagioni avanti — {ta}", w_a)
    c4.metric(f"Stagioni avanti — {tb}", w_b)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=da["season"], y=da[mk], mode="lines+markers", name=ta,
                             line=dict(color=COLORS["primary"], width=3), marker=dict(size=9)))
    fig.add_trace(go.Scatter(x=db["season"], y=db[mk], mode="lines+markers", name=tb,
                             line=dict(color=COLORS["secondary"], width=3), marker=dict(size=9)))
    fig.update_layout(title=f"{ml} — {ta} vs {tb}", height=340,
                      legend=dict(orientation="h"), xaxis_title="Stagione")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Head-to-head")
    h2h = df[(df["team"] == ta) & (df["opponent"] == tb)]
    if len(h2h):
        s = h2h.groupby("result").size().reindex(["W", "D", "L"], fill_value=0)
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Vittorie {ta}", int(s["W"]))
        c2.metric("Pareggi", int(s["D"]))
        c3.metric(f"Sconfitte {ta}", int(s["L"]))
        with st.expander("Partite H2H"):
            st.dataframe(h2h[["date", "season", "venue", "result", "gf", "ga", "xg", "xga"]]
                         .sort_values("date", ascending=False), use_container_width=True, hide_index=True)
    else:
        st.info("Nessuna partita diretta trovata.")


# ─────────────────────────────────────────────
# TAB 8 — SINGOLA SQUADRA
# ─────────────────────────────────────────────

def tab_team(df, ts, p):
    st.header("🔍 Analisi singola squadra")

    all_t = sorted(df["team"].unique())
    team = st.selectbox("Squadra", all_t,
                        index=all_t.index("Napoli") if "Napoli" in all_t else 0, key="ts_sel")

    df_t = df[df["team"] == team].sort_values("date")
    ts_t = ts[ts["team"] == team].sort_values("season")
    vm = p.get("venue_merged", pd.DataFrame())
    vm_t = vm[vm["team"] == team] if len(vm) else pd.DataFrame()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stagioni", len(ts_t))
    c2.metric("Partite", len(df_t))
    c3.metric("Win rate", f"{df_t['win_flag'].mean() * 100:.1f}%")
    c4.metric("xG medio", f"{df_t['xg'].mean():.2f}")

    if len(vm_t):
        r = vm_t.iloc[0]
        st.info(f"**Fattore campo** — Casa: {r['home_avg_points']:.2f} pts/match · "
                f"Trasferta: {r['away_avg_points']:.2f} pts/match · "
                f"Δ win rate: {r['win_rate_diff']:+.3f}")

    st.markdown("---")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ts_t["season"].astype(str), y=ts_t["total_points"],
                         name="Punti", marker_color=COLORS["primary"]))
    fig.add_trace(go.Scatter(x=ts_t["season"].astype(str), y=ts_t["avg_xg"] * 20,
                             mode="lines+markers", name="xG ×20",
                             line=dict(color=COLORS["amber"], width=2), yaxis="y2"))
    fig.update_layout(title=f"{team} — Punti e xG per stagione", height=320,
                      yaxis=dict(title="Punti"),
                      yaxis2=dict(title="xG ×20", overlaying="y", side="right"),
                      legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    res = df_t.groupby(["season", "result"]).size().reset_index(name="count")
    fig2 = px.bar(res, x="season", y="count", color="result", barmode="stack",
                  color_discrete_map={"W": COLORS["green"], "D": COLORS["amber"], "L": COLORS["secondary"]},
                  title=f"{team} — W/D/L per stagione",
                  labels={"count": "Partite", "season": "Stagione", "result": "Esito"})
    fig2.update_layout(height=290)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Ultime 10 partite")
    last10 = (df_t.tail(10)
              [["date", "season", "opponent", "venue", "result", "gf", "ga", "xg", "xga", "poss"]]
              .sort_values("date", ascending=False))

    def color_result(val):
        return {"W": "background-color:#d4edda",
                "D": "background-color:#fff3cd",
                "L": "background-color:#f8d7da"}.get(val, "")

    st.dataframe(last10.style.applymap(color_result, subset=["result"]),
                 use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# TAB 9 — RECORD E CURIOSITÀ
# ─────────────────────────────────────────────

def tab_records(p, df):
    st.header("🏅 Record e curiosità")

    c1, c2 = st.columns(2)

    with c1:
        if "top_goal_diff" in p:
            st.markdown("#### 🎯 Partite con maggior goal diff")
            tgd = p["top_goal_diff"].copy()
            tgd["Partita"] = tgd["team"] + " " + tgd["gf"].astype(int).astype(str) + \
                             "–" + tgd["ga"].astype(int).astype(str) + " " + tgd["opponent"]
            st.dataframe(
                tgd[["date", "season", "Partita", "goal_diff"]].rename(
                    columns={"date": "Data", "season": "Stagione", "goal_diff": "Δ gol"}),
                use_container_width=True, hide_index=True)

        if "daily_stats" in p:
            st.markdown("#### 📅 Gol e spettatori per giorno")
            ds = p["daily_stats"].copy()
            day_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            ds["day"] = pd.Categorical(ds["day"], categories=day_order, ordered=True)
            ds = ds.sort_values("day")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=ds["day"], y=ds["avg_goals_for"],
                                 name="Gol/match", marker_color=COLORS["primary"]))
            fig.add_trace(go.Scatter(x=ds["day"], y=ds["avg_attendance"] / 1000,
                                     mode="lines+markers", name="Spettatori (k)",
                                     line=dict(color=COLORS["amber"], width=2), yaxis="y2"))
            fig.update_layout(
                title="Gol medi e affluenza per giorno", height=320,
                yaxis=dict(title="Gol medi/match"),
                yaxis2=dict(title="Spettatori (k)", overlaying="y", side="right"),
                legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        if "top_xg_matches" in p:
            st.markdown("#### ⚡ Partite con maggior xG diff")
            txg = p["top_xg_matches"].copy()
            txg["Partita"] = txg["team"] + " " + txg["gf"].astype(int).astype(str) + \
                             "–" + txg["ga"].astype(int).astype(str) + " " + txg["opponent"]
            st.dataframe(
                txg[["date", "season", "Partita", "goal_diff", "result"]].rename(
                    columns={"date": "Data", "season": "Stagione",
                             "goal_diff": "Δ gol", "result": "Esito"}),
                use_container_width=True, hide_index=True)

        if "match_df" in p:
            st.markdown("#### 🟨 Top 10 arbitri per presenze")
            mdf = p["match_df"].copy()
            ref = (mdf.groupby("referee")
                   .agg(partite=("result", "count"),
                        gol_medi=("gf", "mean"),
                        home_win=("win_flag", "mean"))
                   .sort_values("partite", ascending=False)
                   .head(10).reset_index().round(3))
            ref.columns = ["Arbitro", "Partite", "Gol medi/match", "Home win rate"]
            st.dataframe(ref, use_container_width=True, hide_index=True)

        if "day_stats_matches" in p:
            st.markdown("#### ⚽ Gol totali medi per giorno")
            dsm = p["day_stats_matches"].copy()
            day_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            dsm["day"] = pd.Categorical(dsm["day"], categories=day_order, ordered=True)
            dsm = dsm.sort_values("day")
            fig2 = px.bar(dsm, x="day", y="avg_total_goals",
                          color="avg_total_goals", color_continuous_scale="Blues",
                          labels={"day": "Giorno", "avg_total_goals": "Gol totali medi"},
                          title="Gol totali medi per giorno")
            fig2.update_layout(height=270, coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    raw_paths = ["data/raw/matches_seriea.csv", "matches_seriea.csv",
                 "../data/raw/matches_seriea.csv"]
    csv_path = next((p for p in raw_paths if Path(p).exists()), None)

    if csv_path is None:
        st.error("Dataset non trovato. Assicurati che `matches_seriea.csv` sia in `data/raw/`.")
        st.stop()

    df = load_raw(csv_path)
    ts = build_ts(df)
    p = load_processed()

    sel_seasons, sel_teams, team_colors = sidebar(df)
    if not sel_seasons:
        st.warning("Seleziona almeno una stagione dal pannello laterale.");
        st.stop()

    tabs = st.tabs([
        "📊 Panoramica",
        "🏆 Title Race",
        "🏟️ Home/Away",
        "📋 Stat. storiche",
        "📈 Progressione",
        "⚡ xG e stile",
        "🆚 Confronto",
        "🔍 Squadra",
        "🏅 Record",
    ])

    with tabs[0]:
        tab_overview(ts, p, sel_seasons)
    with tabs[1]:
        tab_title_race(p)
    with tabs[2]:
        tab_home_away(p)
    with tabs[3]:
        tab_team_stats(p)
    with tabs[4]:
        tab_progression(df, ts, sel_seasons, sel_teams)
    with tabs[5]:
        tab_xg(ts, sel_seasons, team_colors)
    with tabs[6]:
        tab_compare(ts, df, sel_seasons)
    with tabs[7]:
        tab_team(df, ts, p)
    with tabs[8]:
        tab_records(p, df)


if __name__ == "__main__":
    main()
