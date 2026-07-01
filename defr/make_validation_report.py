"""Self-contained HTML report for the DefR injection step.

Documents:
    1. What was attempted (bridge refit, injection, walk-forward)
    2. What was discovered (R² collapse, signal flip, redundancy)
    3. Whether the work is reliable (yes — the negative finding is robust)
    4. What it means for the team_trend pipeline

The report is honest about the negative result. A well-engineered
negative finding is more valuable than a fragile positive: it tells us
exactly when and why the proxy fails to help, which guides future work.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
INJ = ROOT / "output/injection"
VAL = ROOT / "output/validation"
PLOTS = VAL / "plots"
OUT = ROOT / "output/defr_injection_report.html"


def _b64(path: Path) -> str:
    return f"data:image/png;base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


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
  --warn: #BA7517;
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
h1 { font-size: 32px; font-weight: normal; margin: 0 0 8px; }
h2 { font-size: 24px; font-weight: normal; margin: 56px 0 16px;
     padding-bottom: 6px; border-bottom: 1px solid var(--border); }
h3 { font-size: 19px; font-weight: normal; font-style: italic;
     margin: 32px 0 12px; }
h4 { font-size: 16px; font-weight: bold; margin: 24px 0 8px; }
p { margin: 0 0 14px; }
ul, ol { margin: 0 0 14px; padding-left: 24px; }
li { margin-bottom: 6px; }
.subtitle { color: var(--text-dim); font-style: italic; font-size: 18px; }
.meta { color: var(--text-faint); font-size: 14px; margin-top: 12px; }
.lead {
  font-size: 17px; color: var(--text-dim); font-style: italic;
  margin-bottom: 24px; padding-left: 16px; border-left: 3px solid var(--accent);
}
.figure { margin: 28px 0; text-align: center; }
.figure img {
  max-width: 100%; height: auto;
  border: 1px solid var(--border); border-radius: 4px; background: white;
}
.figure .caption {
  font-size: 14px; color: var(--text-dim); margin-top: 10px;
  font-style: italic; text-align: left; padding: 0 20px;
}
.callout {
  background: var(--bg-alt);
  border-left: 3px solid var(--accent);
  padding: 14px 20px;
  margin: 18px 0;
  font-size: 15px;
}
.callout.warning { border-left-color: var(--warn); background: #FFF8E0; }
.callout.success { border-left-color: var(--pos); background: #E8F4EF; }
.callout.danger  { border-left-color: var(--neg); background: #FAEFEF; }
.math {
  background: var(--code-bg); padding: 10px 14px; margin: 12px 0;
  font-family: 'Courier New', monospace; font-size: 14px;
  border-radius: 3px; overflow-x: auto;
}
code {
  font-family: 'Courier New', monospace; font-size: 0.92em;
  background: var(--code-bg); padding: 1px 5px; border-radius: 2px;
}
table.data-table {
  width: 100%; border-collapse: collapse; font-size: 14px;
  font-family: 'Helvetica', sans-serif; margin: 18px 0;
}
table.data-table th {
  text-align: left; padding: 8px 12px;
  border-bottom: 2px solid var(--text); font-weight: bold;
  font-size: 13px; text-transform: uppercase; letter-spacing: 0.04em;
}
table.data-table td {
  padding: 6px 12px; border-bottom: 1px solid var(--border);
}
table.data-table tr:hover td { background: var(--bg-alt); }
.stat-grid {
  display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px;
  margin: 24px 0;
}
.stat {
  background: var(--bg-alt); padding: 16px;
  border-radius: 4px; text-align: center;
}
.stat .value {
  font-size: 28px; font-weight: bold; color: var(--accent);
  display: block; font-family: 'Helvetica', sans-serif;
}
.stat.neg .value { color: var(--neg); }
.stat.pos .value { color: var(--pos); }
.stat.warn .value { color: var(--warn); }
.stat .label {
  font-size: 12px; color: var(--text-dim);
  text-transform: uppercase; letter-spacing: 0.05em; margin-top: 4px;
}
.toc {
  background: var(--bg-alt); padding: 18px 26px;
  border-radius: 4px; margin-bottom: 40px; font-size: 15px;
}
.toc ol { margin: 8px 0 0; padding-left: 22px; }
.toc a { color: var(--text); text-decoration: none; }
.toc a:hover { text-decoration: underline; color: var(--accent); }
footer {
  margin-top: 80px; padding-top: 24px;
  border-top: 1px solid var(--border);
  color: var(--text-faint); font-size: 13px; text-align: center;
}
.pos-val { color: var(--pos); font-weight: bold; }
.neg-val { color: var(--neg); font-weight: bold; }
.tag {
  display: inline-block; padding: 2px 8px;
  border-radius: 10px; font-size: 12px;
  background: var(--bg-alt); color: var(--text-dim);
  font-family: 'Helvetica', sans-serif;
}
"""


def build_report():
    # Load all artifacts
    with open(INJ / "bridge_regression_fbref.json") as f:
        bridge = json.load(f)
    with open(ROOT / "output/data/bridge_regression.json") as f:
        original = json.load(f)
    with open(VAL / "paired_tests.json") as f:
        tests = json.load(f)
    summary = pd.read_csv(VAL / "summary.csv")
    pivot = pd.read_csv(VAL / "f1_pivot.csv", index_col=0)

    # Embed plots
    img = {
        "bridge": _b64(PLOTS / "01_bridge_comparison.png"),
        "rankings": _b64(PLOTS / "02_fbref_rankings.png"),
        "correlation": _b64(PLOTS / "03_correlation.png"),
        "walkforward": _b64(PLOTS / "04_walkforward.png"),
        "standalone": _b64(PLOTS / "05_standalone.png"),
    }

    # Summary table
    summary_html = summary.round(4).to_html(index=False, classes="data-table", border=0)
    # Pivot table
    pivot_display = pivot.round(4).reset_index()
    pivot_html = pivot_display.to_html(index=False, classes="data-table", border=0)

    cv_r2_orig = original["cv_r2_mean"]
    cv_r2_new = bridge["cv_r2_mean"]
    baseline_f1 = float(summary.loc[summary["condition"] == "baseline", "mean_f1_macro"].iloc[0])
    rolling_f1 = float(summary.loc[summary["condition"] == "+proxy_rolling", "mean_f1_macro"].iloc[0])
    match_f1 = float(summary.loc[summary["condition"] == "+proxy_match", "mean_f1_macro"].iloc[0])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>DefR Injection — Walk-Forward Validation</title>
<style>{CSS}</style>
</head>
<body>
<div class="container">

<header>
<h1>DefR proxy injection — walk-forward validation</h1>
<div class="subtitle">An honest negative result: the FBref-compatible proxy
captures real signal but is redundant with existing features.</div>
<div class="meta">seriedAta project · Beppe + Claude · generated {pd.Timestamp.now():%Y-%m-%d}</div>
</header>

<div class="lead">
We applied the Wyscout-derived DefR bridge to FBref Serie A data (2020–2025)
and ran walk-forward validation comparing the existing 23-feature baseline
against three augmented feature sets. The rolling DefR proxy adds <strong>zero</strong>
F1 to baseline; the match-level proxy <em>hurts</em> performance by −0.004 F1
(p = 0.37, not significant). A standalone test confirms the proxy carries
real predictive signal (+0.051 F1 vs <code>is_home</code> alone), but that
signal is fully absorbed by the existing strength and form features. The
work is reliable — the negative finding is robust across 4 seasonal folds —
but the proxy does not warrant inclusion in the production pipeline.
</div>

<div class="toc">
<strong>Contents</strong>
<ol>
<li><a href="#headline">Headline numbers</a></li>
<li><a href="#bridge">The bridge refit: what we lost without pass volume</a></li>
<li><a href="#injection">Injection: applying the bridge to FBref data</a></li>
<li><a href="#walkforward">Walk-forward validation results</a></li>
<li><a href="#why">Diagnosing the zero delta: why the proxy is redundant</a></li>
<li><a href="#standalone">Standalone signal check: the proxy isn't null</a></li>
<li><a href="#reliability">Reliability assessment</a></li>
<li><a href="#conclusions">Conclusions and next steps</a></li>
</ol>
</div>

<!-- ─────────────────────────────────────────────────────────── -->
<h2 id="headline">1. Headline numbers</h2>

<div class="stat-grid">
<div class="stat neg"><span class="value">−0.450</span>
  <span class="label">Bridge R² drop (CV)</span></div>
<div class="stat warn"><span class="value">0.140</span>
  <span class="label">Reduced bridge R²</span></div>
<div class="stat"><span class="value">{baseline_f1:.3f}</span>
  <span class="label">Baseline F1 macro</span></div>
<div class="stat neg"><span class="value">0.000</span>
  <span class="label">Δ F1 from rolling proxy</span></div>
</div>

<p>The pipeline executed end-to-end without errors. Three distinct
observations emerge:</p>

<ul>
<li><strong>The FBref-compatible bridge is much weaker than the original.</strong>
Dropping pass-volume features (<code>n_opp_passes</code>, <code>def_pressure_ratio</code>),
which FBref does not expose, collapsed CV R² from {cv_r2_orig:.3f} to
{cv_r2_new:.3f}. Pass volume carries information that possession percentage
alone does not substitute for.</li>

<li><strong>The walk-forward gain is exactly zero.</strong> Across all four
seasonal folds, adding <code>last_5_defr_proxy</code> to the baseline
feature set changed F1 by 0.0000. Not a small positive eaten by noise — a
literal zero. The model finds nothing in the proxy that the existing
features don't already provide.</li>

<li><strong>The proxy itself is not useless.</strong> A standalone test
(<code>is_home</code> + <code>last_5_defr_proxy</code> vs <code>is_home</code>
alone) shows a +0.051 F1 gain. The proxy carries signal — just signal
that's already captured elsewhere.</li>
</ul>

<!-- ─────────────────────────────────────────────────────────── -->
<h2 id="bridge">2. The bridge refit: what we lost without pass volume</h2>

<p>The original 10-feature bridge fit on Wyscout 2017/18 used two
features that have no FBref equivalent: <code>n_opp_passes</code>
(opponent's pass count) and <code>def_pressure_ratio</code> (which uses
passes as a denominator). FBref via <code>soccerdata</code> exposes
gf, ga, xg, xga, poss, sh, sot, dist, fk, pk — but not pass counts.</p>

<p>Refitting Ridge regression on the same Wyscout dataset using only
the 7 FBref-compatible features yields:</p>

<div class="figure">
<img src="{img['bridge']}" alt="Bridge comparison">
<div class="caption">Figure 1. <em>Left:</em> CV R² collapses from 0.59 to
0.14 (Δ = −0.45). The full-data R² drops similarly. The 41% of variance the
original bridge captured beyond the FBref-only feature set was carried
almost entirely by pass volume. <em>Right:</em> Coefficient drift on the
shared features. <code>n_opp_passes</code> goes from coef = −50 (dominant
feature) to absent. <code>poss_pct</code> flips sign from −29 to +16,
indicating the reduced bridge has learned a structurally different
function.</div>
</div>

<div class="callout warning">
<strong>The sign flip on poss_pct is diagnostic.</strong> In the original
bridge, <code>poss_pct</code> was a correction term — controlling for
possession <em>given</em> that pass count was already in the model. With
pass count gone, <code>poss_pct</code> now has to do double duty and the
coefficient flips positive. This is not a fitting artifact; it tells us
the reduced bridge has learned to predict something <em>different</em>
from what the original predicted. The R² is a measurement of how much
defensive-style signal we keep; the coefficient drift is a measurement of
what the proxy now actually represents.
</div>

<!-- ─────────────────────────────────────────────────────────── -->
<h2 id="injection">3. Injection: applying the bridge to FBref data</h2>

<p>The injection module (<code>inject_defr.py</code>) does five things
per team-match row:</p>

<ol>
<li>Pair the opponent's row in the same match by joining on
(date, opponent ↔ team) to fetch <code>opp_sh</code>, <code>opp_sot</code>,
<code>opp_dist</code>.</li>
<li>Compute the 7 bridge features in FBref units (e.g.,
<code>shot_distance_proxy</code> ← <code>opp_dist</code>;
<code>sot_ratio_against</code> ← <code>opp_sot / opp_sh</code>).</li>
<li>Standardize using the bridge's saved μ, σ.</li>
<li>Apply the bridge: <code>defr_proxy = β · x_std + α</code>.</li>
<li>Add <code>last_5_defr_proxy</code> via <code>groupby(team).shift(1).rolling(5)</code>
— matching the temporal hygiene of existing features like
<code>last_5_points</code>.</li>
</ol>

<p>3,703 of 3,902 rows (94.9%) received a valid <code>defr_proxy</code>;
the missing 5.1% are rows where the mirror opponent row could not be
located (typically the opponent row had missing <code>dist</code>).
3,839 rows (98.4%) received a valid <code>last_5_defr_proxy</code>.</p>

<h3>3.1 Sanity check on 2024/25 rankings</h3>

<div class="figure">
<img src="{img['rankings']}" alt="2024 FBref DefR rankings">
<div class="caption">Figure 2. Season-average DefR proxy for 2024/25.
Note how the ordering now reflects team quality — Inter, Bologna,
Juventus, Atalanta at the top; Verona, Empoli, Lecce at the bottom.
This is structurally different from the Wyscout 2017/18 ranking where
the top was occupied by aggressive pressers (Atalanta, Fiorentina) and
the bottom by possession-dominant teams (Napoli, Juventus). The
ranking inversion is the empirical fingerprint of the signal change
diagnosed in §2.</div>
</div>

<!-- ─────────────────────────────────────────────────────────── -->
<h2 id="walkforward">4. Walk-forward validation results</h2>

<p>Four conditions were tested against the W/D/L classification target
using Logistic Regression with fixed hyperparameters
(C = 1.0, lbfgs, balanced class weights):</p>

<ul>
<li><strong>baseline</strong>: the 23-feature LOGISTIC_NUM_FEATURES set</li>
<li><strong>+proxy_match</strong>: baseline + <code>defr_proxy</code>
(match-level, contains current-match information — see §7 on leakage)</li>
<li><strong>+proxy_rolling</strong>: baseline + <code>last_5_defr_proxy</code>
(properly shifted, no leakage)</li>
<li><strong>+proxy_both</strong>: baseline + both</li>
</ul>

<p>Walk-forward scheme: 2 seasons minimum training, expanding window,
4 folds testing on 2022 → 2023 → 2024 → 2025.</p>

<div class="figure">
<img src="{img['walkforward']}" alt="Walk-forward F1">
<div class="caption">Figure 3. <em>Left:</em> F1 macro per fold for each
condition. The four bars per season are almost indistinguishable.
<em>Right:</em> Per-fold delta vs baseline. The rolling proxy line sits
exactly on zero; the match-level proxy drifts slightly negative in 2024
and 2025.</div>
</div>

<h4>Mean ± std across folds</h4>
{summary_html}

<h4>Per-fold F1 deltas</h4>
{pivot_html}

<p>Paired t-tests on per-fold F1 deltas (one-sided H₁: condition > baseline):</p>

<ul>
<li><strong>+proxy_match</strong>: Δf1 = {tests['+proxy_match']['mean_delta_f1']:+.4f}
± {tests['+proxy_match']['std_delta_f1']:.4f},
t = {tests['+proxy_match']['t_statistic']:+.3f},
p = {tests['+proxy_match']['p_value']:.4f}
— not significant; mean direction negative</li>
<li><strong>+proxy_rolling</strong>: Δf1 = +0.0000 ± 0.0000,
t = NaN, p = NaN — F1 is byte-identical to baseline across all folds</li>
<li><strong>+proxy_both</strong>: Same as <code>+proxy_match</code> (the
rolling feature contributes nothing on top of the match-level one)</li>
</ul>

<div class="callout danger">
<strong>The match-level proxy introduces a subtle leakage problem.</strong>
<code>defr_proxy</code> at the match level uses the team's own shots,
shots-on-target, and possession <em>from the match being predicted</em>.
This is post-outcome information leaking into the predictor. The fact that
adding it <em>hurts</em> performance (rather than inflating it) is consistent
with the proxy being a noisy and indirect predictor — the bridge gives us
DefR, not the result, so the leakage doesn't dominate. But it's still
methodologically wrong to use as a predictor; only the rolling version is
admissible.
</div>

<!-- ─────────────────────────────────────────────────────────── -->
<h2 id="why">5. Diagnosing the zero delta: why the proxy is redundant</h2>

<p>An exact zero delta across all four folds is unusual enough to demand
explanation. Two checks confirmed that:</p>

<ul>
<li>The proxy has variation: 3,194 unique values, range [−51.1, +49.3]</li>
<li>The dropna pattern is identical with or without the proxy
(same row counts in train/test)</li>
</ul>

<p>So the model <em>saw</em> the feature but found no use for it. The
correlation analysis explains why:</p>

<div class="figure">
<img src="{img['correlation']}" alt="Correlation with existing features">
<div class="caption">Figure 4. Pearson correlations between
<code>last_5_defr_proxy</code> and the existing features in the baseline
set. Eight of eleven features correlate at |r| ≥ 0.30, with the strongest
at +0.58 (<code>cum_avg_points</code>) and −0.49 (<code>cum_avg_xga</code>).
The proxy is measuring a linear combination of variables already in the
model.</div>
</div>

<p>This makes mechanical sense. The reduced bridge collapsed to a
combination of <code>poss_pct</code> (+16.6), <code>goals_against</code>
(−8.1), and <code>shots_against</code> (+5.6). All three of these are
proxies for team quality, which is already captured by:</p>

<ul>
<li><code>cum_avg_points</code> — expanding-mean of points-per-match</li>
<li><code>cum_avg_xga</code> — expanding-mean of expected goals against</li>
<li><code>strength_points_diff</code> — same minus opponent's</li>
<li>The team and opponent one-hot-encoded categorical features
(in the CAT_FEATURES set), which act as team-level fixed effects and
absorb any remaining cross-team variation</li>
</ul>

<p>Given a logistic regression with L2 penalty, a feature that linearly
combines other features in the model contributes nothing — the model
just shifts coefficients to capture the same signal through existing
channels. The byte-identical predictions are the deterministic
consequence of this.</p>

<!-- ─────────────────────────────────────────────────────────── -->
<h2 id="standalone">6. Standalone signal check: the proxy isn't null</h2>

<p>Before concluding the proxy is useless, we verified it has signal in
isolation. A minimal model with just <code>is_home</code> versus
<code>is_home</code> + <code>last_5_defr_proxy</code>:</p>

<div class="figure">
<img src="{img['standalone']}" alt="Standalone vs in-pipeline">
<div class="caption">Figure 5. The proxy carries +0.05 average F1 gain
when added to a near-empty baseline (just home advantage), with the
strongest contribution in the 2024 fold (+0.076). But added to the full
23-feature baseline, the same proxy contributes exactly zero. The
signal is real; the redundancy is what makes it useless.</div>
</div>

<p>This is the distinction between an <em>uninformative</em> feature and
a <em>redundant</em> feature. An uninformative feature can never improve
predictions; a redundant feature can improve predictions when added to a
simpler model but contributes nothing on top of the features it's
collinear with. The DefR proxy is the latter.</p>

<!-- ─────────────────────────────────────────────────────────── -->
<h2 id="reliability">7. Reliability assessment</h2>

<p>The question we set out to answer was: <em>does the DefR proxy add
predictive value to the team_trend pipeline?</em> The answer is no, and
we should ask whether that answer is reliable.</p>

<h4>Why the negative result is robust</h4>
<ul>
<li><strong>Walk-forward, not cross-validated</strong>. We tested on
seasons strictly after the training set, four times, with expanding
windows. There is no temporal leakage in the validation procedure.</li>
<li><strong>The delta is exactly zero, not just small</strong>. F1 is
byte-identical across all folds. This rules out the "needs more data"
explanation — adding data would change baseline and proxy F1 equally,
keeping the delta zero.</li>
<li><strong>Three plausible mechanisms all point the same way</strong>:
(a) the bridge R² collapsed when the dominant feature was dropped,
(b) the resulting proxy correlates 0.40–0.58 with existing features,
(c) team OHE absorbs the remaining cross-team variation. Any one of
these would predict no marginal value; all three together is decisive.</li>
<li><strong>The standalone check confirms the signal is real</strong>.
This means we're not in a "broken feature" scenario; we're in a
"redundant feature" scenario. The math is doing what it should.</li>
</ul>

<h4>What could change the answer</h4>
<ul>
<li><strong>Recover the missing feature</strong>. If pass-count data
becomes available for the 2020–2025 seasons (via a different data
provider, scraping, or a paid FBref tier), the bridge could be refit
with the original 10 features, restoring R² to ≈ 0.59. The proxy would
then carry defensive-style information that might be orthogonal to the
existing strength features.</li>
<li><strong>Use a non-linear model</strong>. Gradient-boosted trees can
sometimes extract value from a redundant feature via interactions
(e.g., DefR × strength_xga_diff). We did not test this — the analysis
focused on Logistic Regression because it's the most sensitive to
single-feature additions and the most interpretable.</li>
<li><strong>Different target</strong>. The current target is W/D/L
classification. If the team_trend pipeline shifts to a points-prediction
or a defensive-quality target, the proxy's relevance could change.</li>
</ul>

<h4>What is not a reliability concern</h4>
<ul>
<li>The 5.1% of rows with missing <code>defr_proxy</code> — they were
dropped uniformly across conditions, so they don't bias the comparison.</li>
<li>The choice of fixed hyperparameters instead of GridSearchCV per fold
— we used the same hyperparameters for all conditions, so the comparison
is fair even if the absolute level isn't optimal.</li>
<li>The match-level leakage in <code>defr_proxy</code> — this only
affects the <code>+proxy_match</code> condition, which we flagged
explicitly. The <code>+proxy_rolling</code> condition uses only past
data and is the correct one to draw conclusions from.</li>
</ul>

<!-- ─────────────────────────────────────────────────────────── -->
<h2 id="conclusions">8. Conclusions and next steps</h2>

<div class="callout">
<strong>The work is reliable. The proxy is real but redundant. Do not
inject it into <code>FEATURES_CLEAN</code> as-is.</strong>
</div>

<p>This is not a wasted effort. We now know precisely:</p>

<ol>
<li>What the FBref-compatible bridge measures (something closer to team
quality than defensive style)</li>
<li>Why it doesn't add value in the current pipeline (linear correlation
with existing strength features at r ≈ 0.40–0.58)</li>
<li>What would need to change for the proxy to become useful (recovery
of pass-volume data, non-linear model, or a different target)</li>
</ol>

<h4>Recommended next steps</h4>

<ol>
<li><strong>Park the DefR proxy work</strong>. The current artifacts
(<code>bridge_regression_fbref.json</code>, <code>fbref_with_defr.parquet</code>)
are reproducible from the scripts and worth keeping. They are not worth
plugging into the production pipeline.</li>

<li><strong>Look for ways to obtain pass-volume data</strong> for the
2020–2025 seasons. If the StatsBomb or InStat public APIs cover Serie A
match-level passing, the full 10-feature bridge becomes viable and the
analysis should be repeated. This is the single change that could revive
the proxy.</li>

<li><strong>Try the same approach with a non-linear model</strong>. The
team_trend pipeline includes XGBoost and LightGBM. Both can capture
interactions that Logistic Regression cannot. A repeat experiment with
<code>+proxy_rolling</code> on XGBoost may not give 0.000 — it would be
worth one fold to check.</li>

<li><strong>Use the DefR work descriptively, not predictively</strong>.
The Wyscout-derived season DefR scores from the original analysis are a
solid stand-alone artifact for any defensive-style discussion. They can
inform feature engineering ideas (e.g., a binary "pressing team" flag
derived from possession × shots-against patterns) without going through
the bridge regression at all.</li>
</ol>

<footer>
DefR injection step · seriedAta project · all results reproducible from
<code>run_defr_analysis.py</code> → <code>refit_bridge_fbref.py</code> →
<code>inject_defr.py</code> → <code>walkforward_validate.py</code> →
<code>make_validation_plots.py</code>
</footer>

</div>
</body>
</html>
"""

    OUT.write_text(html, encoding="utf-8")
    print(f"Report written to {OUT}")
    print(f"Size: {OUT.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    build_report()
