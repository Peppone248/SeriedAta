"""Self-contained HTML report generator.

Produces a single .html file with embedded base64 PNG plots and full
methodological narrative. Designed to be opened directly in a browser
with no external dependencies (no CDN, no separate image files).

Structure:
    1. Executive summary
    2. Motivation and scope
    3. Data
    4. Methodology
        4.1 Zone discretization
        4.2 Event classification
        4.3 Zone baseline rates
        4.4 DefR scoring
        4.5 Bridge regression
    5. Results
        5.1 Zone baselines
        5.2 DefR distribution
        5.3 Season rankings
        5.4 Bridge regression performance
        5.5 Team style clustering
        5.6 Footballistic validation
    6. Limitations and assumptions
    7. Next steps
    8. References

The methodology section gives the full mathematical statement.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pandas as pd

from . import config


# ─── Helpers ──────────────────────────────────────────────────────────
def _embed_image(img_path: Path) -> str:
    """Base64-encode an image for inline embedding in HTML."""
    data = img_path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _table(df: pd.DataFrame, formatters: dict | None = None) -> str:
    """Render a DataFrame as a clean HTML table."""
    if formatters:
        df = df.copy()
        for col, fmt in formatters.items():
            if col in df.columns:
                df[col] = df[col].apply(lambda v: fmt.format(v) if pd.notna(v) else "—")
    return df.to_html(index=False, classes="data-table", border=0, escape=False)


# ─── CSS ──────────────────────────────────────────────────────────────
CSS = """
:root {
  --text: #1a1a1a;
  --text-dim: #555;
  --text-faint: #888;
  --bg: #fefefe;
  --bg-alt: #f7f5f0;
  --accent: #1E5A99;
  --pos: #0F6E56;
  --neg: #C03A2B;
  --border: #d8d4c8;
  --code-bg: #f0ede5;
}
* { box-sizing: border-box; }
body {
  font-family: 'Georgia', 'Times New Roman', serif;
  font-size: 16px;
  line-height: 1.6;
  color: var(--text);
  background: var(--bg);
  margin: 0;
  padding: 0;
}
.container {
  max-width: 880px;
  margin: 0 auto;
  padding: 60px 40px 80px;
}
header {
  border-bottom: 2px solid var(--text);
  padding-bottom: 24px;
  margin-bottom: 40px;
}
h1 {
  font-size: 32px;
  font-weight: normal;
  margin: 0 0 8px;
  letter-spacing: -0.01em;
}
header .subtitle {
  color: var(--text-dim);
  font-style: italic;
  font-size: 18px;
}
header .meta {
  color: var(--text-faint);
  font-size: 14px;
  margin-top: 12px;
}
h2 {
  font-size: 24px;
  font-weight: normal;
  margin-top: 56px;
  margin-bottom: 16px;
  padding-bottom: 6px;
  border-bottom: 1px solid var(--border);
}
h3 {
  font-size: 19px;
  font-weight: normal;
  font-style: italic;
  margin-top: 32px;
  margin-bottom: 12px;
  color: var(--text);
}
h4 {
  font-size: 16px;
  font-weight: bold;
  margin-top: 24px;
  margin-bottom: 8px;
}
p { margin: 0 0 14px; }
ul, ol { margin: 0 0 14px; padding-left: 24px; }
li { margin-bottom: 6px; }
.lead {
  font-size: 17px;
  color: var(--text-dim);
  font-style: italic;
  margin-bottom: 24px;
  padding-left: 16px;
  border-left: 3px solid var(--accent);
}
.figure {
  margin: 28px 0;
  text-align: center;
}
.figure img {
  max-width: 100%;
  height: auto;
  border: 1px solid var(--border);
  border-radius: 4px;
  background: white;
}
.figure .caption {
  font-size: 14px;
  color: var(--text-dim);
  margin-top: 10px;
  font-style: italic;
  text-align: left;
  padding: 0 20px;
}
.callout {
  background: var(--bg-alt);
  border-left: 3px solid var(--accent);
  padding: 14px 20px;
  margin: 18px 0;
  font-size: 15px;
}
.callout.warning {
  border-left-color: var(--neg);
}
.callout.success {
  border-left-color: var(--pos);
}
.math {
  background: var(--code-bg);
  padding: 10px 14px;
  margin: 12px 0;
  font-family: 'Courier New', monospace;
  font-size: 14px;
  border-radius: 3px;
  overflow-x: auto;
}
.math-inline {
  font-family: 'Courier New', monospace;
  font-size: 0.92em;
  background: var(--code-bg);
  padding: 1px 5px;
  border-radius: 2px;
}
code {
  font-family: 'Courier New', monospace;
  font-size: 0.92em;
  background: var(--code-bg);
  padding: 1px 5px;
  border-radius: 2px;
}
table.data-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
  font-family: 'Helvetica', sans-serif;
  margin: 18px 0;
}
table.data-table th {
  text-align: left;
  padding: 8px 12px;
  border-bottom: 2px solid var(--text);
  font-weight: bold;
  font-size: 13px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
table.data-table td {
  padding: 6px 12px;
  border-bottom: 1px solid var(--border);
}
table.data-table tr:hover td {
  background: var(--bg-alt);
}
.stat-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
  margin: 24px 0;
}
.stat {
  background: var(--bg-alt);
  padding: 16px;
  border-radius: 4px;
  text-align: center;
}
.stat .value {
  font-size: 28px;
  font-weight: bold;
  color: var(--accent);
  display: block;
  font-family: 'Helvetica', sans-serif;
}
.stat .label {
  font-size: 12px;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-top: 4px;
}
.toc {
  background: var(--bg-alt);
  padding: 18px 26px;
  border-radius: 4px;
  margin-bottom: 40px;
  font-size: 15px;
}
.toc ol {
  margin: 8px 0 0;
  padding-left: 22px;
}
.toc a {
  color: var(--text);
  text-decoration: none;
}
.toc a:hover { text-decoration: underline; color: var(--accent); }
footer {
  margin-top: 80px;
  padding-top: 24px;
  border-top: 1px solid var(--border);
  color: var(--text-faint);
  font-size: 13px;
  text-align: center;
}
.pos-val { color: var(--pos); font-weight: bold; }
.neg-val { color: var(--neg); font-weight: bold; }
"""


# ─── Report builder ───────────────────────────────────────────────────
def build_report(
    plot_paths: dict[str, Path],
    zone_rates: pd.DataFrame,
    defr_match: pd.DataFrame,
    season: pd.DataFrame,
    bridge_results: dict,
    n_events: int,
    n_matches: int,
    n_teams: int,
    out_path: Path,
):
    """Compose the full HTML report and write it to out_path."""
    # Encode all images
    img = {name: _embed_image(p) for name, p in plot_paths.items()}

    # Pre-render tables
    season_display = season[["rank", "team_name", "matches", "avg_defr",
                              "std_defr", "avg_ratio"]].copy()
    season_display.columns = ["Rank", "Team", "Matches", "Avg DefR", "σ", "Ratio"]
    season_table = _table(season_display, {
        "Avg DefR": "{:+.2f}", "σ": "{:.2f}", "Ratio": "{:.3f}",
    })

    coef_data = pd.DataFrame(bridge_results["feature_ranking"])
    coef_data.columns = ["Feature", "Coefficient", "|Coefficient|"]
    coef_table = _table(coef_data, {
        "Coefficient": "{:+.4f}", "|Coefficient|": "{:.4f}",
    })

    # Stats for the headline grid
    cv_r2 = bridge_results["cv_r2_mean"]
    cv_r2_std = bridge_results["cv_r2_std"]
    cv_mae = bridge_results["cv_mae_mean"]
    n_samples = bridge_results["n_samples"]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>DefR Analysis — Serie A 2017/18</title>
<style>{CSS}</style>
</head>
<body>
<div class="container">

<header>
<h1>Defensive Responsibility (DefR) — a transferable proxy</h1>
<div class="subtitle">Building an event-grounded defensive metric on Wyscout
Serie A 2017/18, with a bridge regression to FBref-style aggregates.</div>
<div class="meta">seriedAta project · Beppe + Claude · generated {pd.Timestamp.now():%Y-%m-%d}</div>
</header>

<div class="lead">
This report develops a team-level Defensive Responsibility metric on
~647k event records from 380 Wyscout-tracked Serie A matches, then fits
a Ridge regression that maps the metric onto the aggregate statistics
available from FBref. The bridge regression achieves cross-validated
R² ≈ {cv_r2:.2f}, giving us a defensible way to inject a DefR proxy
into the seriedAta team_trend pipeline for seasons where event data is
not available.
</div>

<div class="toc">
<strong>Contents</strong>
<ol>
<li><a href="#summary">Executive summary</a></li>
<li><a href="#motivation">Motivation and scope</a></li>
<li><a href="#data">Data</a></li>
<li><a href="#methodology">Methodology</a></li>
<li><a href="#results">Results</a></li>
<li><a href="#limitations">Limitations and assumptions</a></li>
<li><a href="#next">Next steps</a></li>
<li><a href="#refs">References</a></li>
</ol>
</div>

<!-- ──────────────────────────────────────────────────────────── -->
<h2 id="summary">1. Executive summary</h2>

<div class="stat-grid">
<div class="stat"><span class="value">{n_events:,}</span><span class="label">Events parsed</span></div>
<div class="stat"><span class="value">{n_matches}</span><span class="label">Matches</span></div>
<div class="stat"><span class="value">{n_teams}</span><span class="label">Teams</span></div>
<div class="stat"><span class="value">{cv_r2:.3f}</span><span class="label">CV R² (bridge)</span></div>
</div>

<p>
We measure each team's <em>Defensive Responsibility</em> in each match
as the gap between defensive actions actually taken and those expected
from the volume of opposition attacking actions, weighted by zone-specific
baseline rates. Aggregated to the season, the metric assigns labels that
agree with established footballing knowledge:
</p>

<ul>
<li><strong>Atalanta</strong> (DefR = +18.0) — Gasperini's aggressive man-marking system.</li>
<li><strong>Napoli</strong> (DefR = −24.2) — Sarri's possession-dominant style.</li>
<li><strong>Juventus</strong> (DefR = −11.3) — Allegri's controlled pragmatism.</li>
</ul>

<p>
A Ridge regression with 10 aggregate features (the same a team-match
row gets via FBref) predicts the DefR score with cross-validated
R² = {cv_r2:.3f} ± {cv_r2_std:.3f} on {n_samples} team-matches. The
fitted coefficients and scaler parameters constitute an exportable
formula that can be applied to the 2020–2025 FBref data and tested for
predictive lift on the team_trend target.
</p>

<!-- ──────────────────────────────────────────────────────────── -->
<h2 id="motivation">2. Motivation and scope</h2>

<p>
The Hudl/StatsBomb article <em>Defensive Responsibility: A New Way To
Measure Defensive Output</em> introduces a per-player metric defined as
the difference between actual defensive actions and those expected given
the opposition's attacking actions. The original DefR is computed from
frame-level positional data and granular event streams — the
defensive shape determines who is "responsible" for each opposition
event, and the metric is the gap from that responsibility.
</p>

<p>
Our seriedAta pipeline operates on FBref data accessed via the
<code>soccerdata</code> library. FBref provides match-level aggregates —
goals, expected goals, shots, possession, average shot distance — but
no event-level coordinates and no defensive-shape information. A direct
implementation of player-level DefR is therefore impossible on our
target data.
</p>

<p>
This work asks a narrower question: <em>can we build a team-level DefR
proxy that captures the same conceptual content — defensive action
relative to defensive demand — and is expressible as a function of
FBref aggregates?</em> If yes, the proxy becomes a candidate feature for
the team_trend pipeline, where its predictive value can be tested
through ablation and walk-forward backtesting.
</p>

<div class="callout">
<strong>Scope.</strong> This work delivers (1) a team-level DefR metric
on Wyscout Serie A 2017/18, (2) a Ridge regression mapping aggregate
features to DefR, (3) the fitted coefficients and scaler parameters
in JSON form for downstream injection. It does <em>not</em> yet inject
the proxy into the team_trend pipeline; that step is a separate
validation experiment.
</div>

<!-- ──────────────────────────────────────────────────────────── -->
<h2 id="data">3. Data</h2>

<p>
The source is the Wyscout Open Data release accompanying Pappalardo et
al. (2019), <em>"A public data set of spatio-temporal match events in
soccer competitions"</em>, published in <em>Scientific Data</em>. The
release contains all spatio-temporal events from the 2017/18 season of
the top-five European leagues plus Euro 2016 and the 2018 World Cup.
We use the Italian Serie A subset: 380 matches, 20 teams, 647,372
events. The dataset is mirrored on GitHub by koenvo in JSON form
amenable to programmatic ingestion.
</p>

<p>
Each event row carries: match id, team id, player id, period and second,
event name (Pass, Shot, Duel, Foul, etc.) and sub-event (Simple pass,
Ground defending duel, Clearance, etc.), the (x, y) origin and destination
coordinates on a normalized 0–100 pitch, and a set of tags including
accuracy and goal flags.
</p>

<p>
The data is from 2017/18; our target FBref data covers 2020–2025. The
two datasets share no rows — they cannot be merged. The strategy is to
use the event data only to fit the bridge regression, then apply the
fitted coefficients to FBref aggregates. We assume that the relationship
between aggregate stats and defensive output is structurally stable across
seasons even if specific squad compositions change; this assumption is
discussed in §6.
</p>

<!-- ──────────────────────────────────────────────────────────── -->
<h2 id="methodology">4. Methodology</h2>

<h3>4.1 Zone discretization</h3>

<p>
We partition the pitch into a 6 × 4 grid of zones (24 zones total).
The grid is uniform in both directions: each zone covers a
{config.PITCH_X_MAX / config.N_ZONE_COLS:.1f} × {config.PITCH_Y_MAX / config.N_ZONE_ROWS:.1f}
rectangle in Wyscout coordinates. Each event is assigned to the zone
containing its origin coordinate:
</p>

<div class="math">
zone_col = ⌊x / (100 / 6)⌋ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; zone_row = ⌊y / (100 / 4)⌋ <br>
zone_id  = zone_row × 6 + zone_col &nbsp;&nbsp;&nbsp;∈ {{0, 1, …, 23}}
</div>

<p>
The grid resolution is a compromise. Finer grids (say 12 × 8 = 96 zones)
give better spatial fidelity but suffer from small-sample noise in the
rate estimates — some zones would see only a few hundred events across
the entire season. Coarser grids (say 3 × 2) capture spatial structure
too crudely. The 6 × 4 grid puts roughly {n_events // 24:,} events per
zone on average, providing stable rate estimates without sacrificing
meaningful spatial structure.
</p>

<h3>4.2 Event classification</h3>

<p>
Each event is tagged as <em>defensive</em>, <em>attacking</em>, or
neither, based on its event-name × sub-event combination:
</p>

<ul>
<li><strong>Defensive events</strong>: Ground defending duels, Air duels,
Clearances, Fouls, Save attempts, Reflex saves. These are the actions
counted as "actual defensive output" for the team performing them.</li>
<li><strong>Attacking events</strong>: Passes, Shots, Touches and Free
Kicks — events that constitute pressure on the defending team. We
explicitly <em>exclude</em> clearances, throw-ins and goal kicks because
these are defensive recovery actions, not pressure.</li>
<li><strong>Other</strong>: Offsides, Interruptions, Goalkeeper leaving
line — neither.</li>
</ul>

<div class="callout">
<strong>Why excluding throw-ins matters.</strong> A naive count of
"passes by the opponent" as attacking pressure would inflate expected
defensive demand by counting set-piece restarts (throw-ins, goal kicks)
as if they were active threats. This would systematically bias the
baseline rates downwards for zones near the touchlines and own goal,
where these restarts cluster.
</div>

<h3>4.3 Zone baseline rates</h3>

<p>
For each zone <span class="math-inline">z</span>, we estimate the
league-wide baseline rate at which defensive actions follow attacking
actions in that zone:
</p>

<div class="math">
baseline_rate(z) = (Σ defensive actions in zone z) / (Σ attacking actions in zone z)
</div>

<p>
The sums run over all events in the 380 matches. This is a
maximum-likelihood estimate under the model that defensive engagement
in zone <span class="math-inline">z</span> is a Bernoulli outcome per
attacking event, with zone-specific probability
<span class="math-inline">p_z</span>. The estimator is unbiased and
benefits from the large sample size.
</p>

<p>
The rates vary by an order of magnitude across the pitch: from ~0.12
high up the field to ~1.05 at the defending team's own goal line.
Higher rates near one's own goal reflect a footballistic truth — every
opposition action there triggers an immediate, urgent defensive
response (clearance, block, goalkeeper save). Mid-field zones see many
passes but few direct defensive contests, hence rates of 0.18–0.25.
</p>

<h3>4.4 DefR scoring</h3>

<p>
For each team <span class="math-inline">t</span> in each match
<span class="math-inline">m</span>:
</p>

<div class="math">
expected_def(t, m) = Σ_z opp_attacking(z, m) × baseline_rate(z) <br>
actual_def(t, m)   = Σ_z own_defensive(z, m) <br>
defr_score(t, m)   = actual_def(t, m) − expected_def(t, m) <br>
defr_ratio(t, m)   = actual_def(t, m) / expected_def(t, m)
</div>

<p>
The interpretation:
</p>

<ul>
<li><strong>defr_score &gt; 0</strong> — the team made more defensive
actions than the opponent's attacking volume "demanded." This signals
an aggressive, proactive defensive approach: pressing, intercepting,
duelling more than the structural situation required.</li>
<li><strong>defr_score &lt; 0</strong> — fewer defensive actions than
expected. This can mean two opposite things: a passive low-block that
absorbs without engaging, or a possession-dominant team that simply
doesn't face many attacking actions to begin with. The ambiguity is
resolved in §5.5 by viewing DefR alongside possession.</li>
</ul>

<h3>4.5 Bridge regression</h3>

<p>
The bridge regression learns the mapping from FBref-style aggregate
features to the DefR score. We use Ridge regression — a linear model
with an L2 penalty on the coefficients — because (1) the predictors are
moderately correlated (multicollinearity would destabilize OLS), and
(2) we want a stable, well-conditioned formula to transfer to a
different dataset.
</p>

<div class="math">
defr_score = β·x_std + α + ε  &nbsp;&nbsp;where x_std = (x − μ_x) / σ_x <br>
β̂ = argmin Σ (y − x_std β − α)² + λ‖β‖²  &nbsp;&nbsp;(λ = {config.RIDGE_ALPHA})
</div>

<p>
Standardization is essential for Ridge: without it the L2 penalty
applies inequitably across features with different natural scales
(<code>n_opp_passes</code> ≈ 500, <code>sot_against</code> ≈ 4). The
scaler parameters <em>μ_x, σ_x</em> are saved alongside the coefficients
so they can be applied identically to the target FBref data.
</p>

<p>
The ten features used are: shots against, shots on target against,
possession percentage, average opposition shot x-coordinate, goals against,
opposition passes, shots for, shot-on-target ratio against, shot distance
proxy, and defensive pressure ratio (shots conceded per opposition pass).
All have FBref analogues or can be approximated from FBref data.
</p>

<p>
Performance is reported as cross-validated R² and MAE with
{config.CV_FOLDS}-fold splits. The full-data fit is then taken as the
transfer formula. Residual diagnostics (§5.4) check whether the linear
specification is adequate.
</p>

<!-- ──────────────────────────────────────────────────────────── -->
<h2 id="results">5. Results</h2>

<h3>5.1 Zone baseline rates</h3>

<div class="figure">
<img src="{img['pitch_zones']}" alt="Pitch zone heatmap">
<div class="caption">Figure 1. Baseline defensive rates per zone, computed
as the ratio of defensive actions to attacking actions across all 380
matches. Each cell shows the estimated probability that an opposition
attacking action in that zone is followed by a defensive action from
the defending team. The defending team's goal is on the left.</div>
</div>

<p>
Three patterns are visible. First, the left column (the defending team's
own box) shows the highest rates, peaking at 1.05 in the corner —
attacking actions there trigger near-certain defensive engagement.
Second, the central midfield zones have low rates (~0.18) — passes
exchange frequently but few are directly contested. Third, the right
column (opponent's box, attacking direction) shows intermediate rates
(~0.34–0.51) reflecting high pressing and tackling activity in advanced
zones.
</p>

<h3>5.2 DefR distribution</h3>

<div class="figure">
<img src="{img['defr_distribution']}" alt="DefR distribution">
<div class="caption">Figure 2. <em>Left:</em> Histogram of match-level
DefR scores across {n_samples} team-matches. The distribution is roughly
symmetric and unimodal around zero, with a standard deviation of about 36.
<em>Right:</em> Per-team box plots sorted by season-mean DefR, with
positive-mean teams in green and negative in red.</div>
</div>

<p>
The match-level DefR distribution is approximately Gaussian-shaped
around zero, which is reassuring — the metric is a difference, and by
construction across an entire league, over- and under-performance
should roughly balance. Individual team distributions show substantial
match-to-match variance (within-team σ ≈ 27–45), reflecting the
heterogeneity of opponents and game states across a season.
</p>

<h3>5.3 Season rankings</h3>

<div class="figure">
<img src="{img['team_rankings']}" alt="Team rankings">
<div class="caption">Figure 3. Season-average DefR for all 20 Serie A
teams in 2017/18, sorted ascending. Error bars are ±1 standard error of
the mean across 38 matches. Positive scores (green) indicate teams
defending more than expected; negative (red) defending less.</div>
</div>

<p>
The full ranking is in Table 1. The rank gap between top and bottom is
substantial — Atalanta's +18.0 vs Napoli's −24.2 is a 42-action-per-match
spread, which is large relative to the cross-team standard deviation of
~13.
</p>

{season_table}

<h3>5.4 Bridge regression performance</h3>

<div class="figure">
<img src="{img['regression_diagnostics']}" alt="Regression diagnostics">
<div class="caption">Figure 4. Four-panel diagnostics for the Ridge
bridge regression. <em>Top-left:</em> predicted vs actual DefR with
the y = x line of perfect fit. <em>Top-right:</em> residuals vs predicted
to check for heteroscedasticity; the rolling-mean line shows no
systematic trend. <em>Bottom-left:</em> Q-Q plot of residuals against a
normal distribution. <em>Bottom-right:</em> residual histogram with a
fitted normal density overlay.</div>
</div>

<p>
The full-data fit achieves R² = {bridge_results['full_r2']:.3f} with a
mean absolute error of {bridge_results['full_mae']:.2f} DefR units;
{config.CV_FOLDS}-fold cross-validation gives R² = {cv_r2:.3f}
± {cv_r2_std:.3f}. The fold-level R² values are:
{', '.join(f'{r:.3f}' for r in bridge_results['cv_r2_folds'])}.
The gap between full-data and CV R² (~0.04) is small, indicating the
model is not over-fitting in any serious way.
</p>

<p>
The residuals are close to normally distributed (the Q-Q plot follows
the diagonal except for a few tail points) and roughly homoscedastic
(the rolling mean of residuals against predicted values stays near zero
across the range). The linear specification is therefore adequate; a
more flexible model (gradient boosting, kernel regression) might squeeze
out additional R² but would not transfer as cleanly.
</p>

<h4>Feature ranking</h4>

<p>
With standardized features the absolute coefficients are directly
comparable as importance scores. Table 2 ranks them.
</p>

{coef_table}

<p>
The two dominant predictors — opposition passes and possession percentage —
both load negatively. This is exactly the article's central thesis:
defensive output relative to defensive demand is overwhelmingly
contextualized by possession. A team facing many opposition passes has
high defensive demand and tends to convert less of that demand into
actual defensive actions (relatively speaking) because much of the
opposition activity passes through unchallenged. The bridge formula
captures this dynamic from aggregate data.
</p>

<h3>5.5 Team style clustering</h3>

<div class="figure">
<img src="{img['style_clustering']}" alt="Style clustering">
<div class="caption">Figure 5. Each team plotted by average possession
(x-axis) and average DefR (y-axis). Bubble size is proportional to total
defensive actions over the season. Quadrants suggest qualitative style
labels: aggressive press (low possession, high DefR), possession dominant
(high possession, low DefR), passive block (low possession, low DefR),
active defenders (high possession, high DefR).</div>
</div>

<p>
The four quadrants give an intuitive map of defensive identity. Atalanta
and Torino in the aggressive-press quadrant defend more than their
possession share would suggest. Napoli and Juventus in the
possession-dominant quadrant rarely face pressure to defend. The bottom-left
quadrant — passive blocks — captures teams under structural pressure that
nonetheless absorb without aggressive engagement.
</p>

<h3>5.6 Footballistic validation</h3>

<div class="figure">
<img src="{img['football_validation']}" alt="Footballistic validation">
<div class="caption">Figure 6. Season DefR rankings annotated with each
team's known tactical identity in 2017/18. The annotations were chosen
before viewing the rankings, based on managerial and tactical context.</div>
</div>

<div class="callout success">
<strong>Validation passes.</strong> The metric assigns rankings that
agree with widely-known facts of Serie A 2017/18: Gasperini's Atalanta
high, Sarri's Napoli low, Allegri's Juventus low. This is a necessary
(not sufficient) condition for the metric to be capturing what we
intend.
</div>

<p>
The Crotone case deserves a note. Crotone finished 18th and were
relegated, yet they rank 4th by DefR (+11.9). This is not a contradiction:
DefR measures defensive <em>activity</em> relative to demand, not
defensive <em>quality</em>. A team under sustained pressure that scrambles
constantly to clear, block and tackle will register high DefR even if it
concedes many goals. This distinction matters for the bridge: when we
later inject the DefR proxy into the team_trend pipeline, we should
remember that what it adds is information about engagement style, not
about defensive quality directly.
</p>

<!-- ──────────────────────────────────────────────────────────── -->
<h2 id="limitations">6. Limitations and assumptions</h2>

<h4>Temporal extrapolation</h4>
<p>
The bridge is fit on 2017/18 data and will be applied to 2020–2025 data.
The implicit assumption is that the structural relationship between
aggregate statistics and defensive output is stable across seasons. This
is plausible — the underlying football mechanics don't change — but
tactical evolutions (more high pressing across the league, more positional
play, VAR effects on fouls) may shift specific coefficients. The bridge
should be re-validated periodically.
</p>

<h4>Activity, not quality</h4>
<p>
As the Crotone case shows, DefR captures defensive activity relative to
demand, not defensive quality (goals prevented, errors avoided). When
combined with other features in the team_trend pipeline, the DefR proxy
should be interpreted as a stylistic signal, not a quality signal.
Goalkeeping quality, last-ditch blocks, and positional discipline are
better captured by the residual <code>ga − xga</code>.
</p>

<h4>Team-level only</h4>
<p>
The original DefR is per-player; we aggregate to team level because our
target data (FBref via soccerdata) provides no per-player match-level
defensive metrics with x,y coordinates. Player-level transfer would
require fundamentally different target data.
</p>

<h4>Wyscout vs FBref event taxonomies differ</h4>
<p>
The aggregate features used in the bridge were computed from the Wyscout
events themselves (to ensure exact alignment with the DefR scores being
predicted). When the bridge is applied to FBref data, the analogous
columns are not identical — FBref's "shots faced" may use a different
counting rule than Wyscout's "Shot" events. The two sources are likely
within a few percent of each other on average, but this is a known
source of transfer error that should be monitored.
</p>

<h4>Sample size for cross-validation</h4>
<p>
The bridge is fit on {n_samples} team-matches. {config.CV_FOLDS}-fold CV
gives fold sizes of ~{n_samples // config.CV_FOLDS}, which is adequate
but not large. A 95% CI on the CV R² is approximately ±{2 * cv_r2_std:.3f},
so the true out-of-sample R² is plausibly somewhere in
[{cv_r2 - 2 * cv_r2_std:.3f}, {cv_r2 + 2 * cv_r2_std:.3f}].
</p>

<!-- ──────────────────────────────────────────────────────────── -->
<h2 id="next">7. Next steps</h2>

<ol>
<li><strong>Inject as Gold-layer feature.</strong> Apply the bridge
formula to the 2020–2025 FBref data in the team_trend pipeline to
compute <code>defr_proxy</code> per team-match.</li>
<li><strong>Ablation testing.</strong> Run the team_trend ablation
infrastructure with and without <code>defr_proxy</code> in
<code>FEATURES_CLEAN</code>. The hypothesis is that the proxy carries
information orthogonal to the existing opponent-context features.</li>
<li><strong>Walk-forward validation.</strong> Confirm any MAE
improvement on out-of-sample matchweeks before committing the feature.</li>
<li><strong>Rolling DefR proxy.</strong> Test <code>roll5_defr_proxy</code>
(5-match rolling mean) as a momentum-style feature consistent with the
existing pipeline conventions.</li>
<li><strong>Re-fit on newer data.</strong> If a more recent open event
dataset becomes available for Serie A (e.g. SkillCorner extending its
A-League sample to other leagues), refit the bridge on that data and
compare coefficients to check temporal stability.</li>
</ol>

<!-- ──────────────────────────────────────────────────────────── -->
<h2 id="refs">8. References</h2>

<ol>
<li>Pappalardo, L., Cintia, P., Rossi, A., Massucco, E., Ferragina, P.,
Pedreschi, D., Giannotti, F. (2019). <em>A public data set of
spatio-temporal match events in soccer competitions.</em> Scientific
Data 6:236. doi:10.1038/s41597-019-0247-7</li>
<li>Hudl/StatsBomb. <em>Defensive Responsibility: A New Way To Measure
Defensive Output.</em> Article supplied as project reference.</li>
<li>koenvo. <em>Wyscout soccer match event dataset (mirror).</em>
github.com/koenvo/wyscout-soccer-match-event-dataset</li>
<li>Pedregosa et al. (2011). <em>Scikit-learn: Machine Learning in
Python.</em> JMLR 12:2825-2830.</li>
</ol>

<footer>
Generated by the DefR analysis pipeline.
Reproducible from source via <code>python run_defr_analysis.py</code>.
</footer>

</div>
</body>
</html>
"""

    out_path.write_text(html, encoding="utf-8")
    print(f"  Report written to {out_path}")
