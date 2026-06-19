#!/usr/bin/env python3
"""Build a self-contained one-page HTML decision memo + dashboard.
Embeds matplotlib charts as base64 PNGs so the file is portable."""
import base64
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from datetime import datetime, timezone

plt.style.use('seaborn-v0_8-whitegrid')

# Load all phase results
with open("/tmp/wildfire_mc_results.json") as f: p1 = json.load(f)
with open("/tmp/wildfire_p2_results.json") as f: p2 = json.load(f)
with open("/tmp/wildfire_p3_results.json") as f: p3 = json.load(f)

SHELL = 334_000
PREM  = 24_000
HORIZONS = [5, 10, 15]

p1_dt = {row["horizon"]: row for row in p1["decision_table"]}
p2_dt = {row["horizon"]: row for row in p2["decision_table"]}
p3_dt = {row["horizon"]: row for row in p3["decision_table"]}

def fig_to_b64(fig):
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')

# === Plot 1: Phase evolution — survival curves ===
years_plot = np.arange(0, 21)
def cum_simple(p_ann, ct=0.03):
    return [1 - math.exp(-p_ann * (t + ct * t**2 / 2.0)) for t in years_plot]

fig1, ax1 = plt.subplots(figsize=(11, 5.5))
ax1.plot(years_plot, cum_simple(p1["p_annual_summary"]["median"], 0.03),
         label="Phase 1: FSim wildfire only", color="#1f77b4", linestyle="--", linewidth=2)
ax1.plot(years_plot, cum_simple(p2["p_annual_summary"]["median"], 0.03),
         label="Phase 2: + DINS-fitted vulnerability + interior ignition", color="#2ca02c", linestyle="-.", linewidth=2)
ax1.plot(years_plot, cum_simple(p3["p_annual_summary_timeavg_15yr"]["median"], 0.03),
         label="Phase 3: + Cal-Adapt climate ensemble (CMIP5)", color="#d62728", linewidth=2.5)

# Reality-adjusted band
real_low_p_ann  = p3["p_annual_summary_timeavg_15yr"]["median"] * 2.5  # × partial loss factor
real_high_p_ann = p3["p_annual_summary_timeavg_15yr"]["median"] * 12   # empirical FAIR Plan implied
ax1.plot(years_plot, cum_simple(real_low_p_ann, 0.03),
         label="Reality-adjusted band (×2.5–12 model)", color="#ff7f0e", linestyle=":", linewidth=1.5)
ax1.plot(years_plot, cum_simple(real_high_p_ann, 0.03),
         color="#ff7f0e", linestyle=":", linewidth=1.5)
ax1.fill_between(years_plot, cum_simple(real_low_p_ann, 0.03), cum_simple(real_high_p_ann, 0.03),
                  color="#ff7f0e", alpha=0.12)

ax1.set_xlabel("Years held (2026 → 2041)")
ax1.set_ylabel("Cumulative probability of total structural loss")
ax1.set_title("Model evolution: Phase 1 → Phase 3 + reality-adjusted band")
ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax1.legend(loc="upper left", fontsize=9)
for h in HORIZONS:
    ax1.axvline(h, linestyle=":", color="gray", alpha=0.5)
    ax1.text(h, 0.001, f"{h}y", color="gray", fontsize=8, ha='center')
plt.tight_layout()
plot1_b64 = fig_to_b64(fig1)
plt.close()

# === Plot 2: Hazard decomposition (the key finding) ===
fig2, ax2 = plt.subplots(figsize=(11, 5.5))
labels = ["Wildfire direct\n(BP × FLEP × vuln)",
          "Wildfire indirect\n(ember/spotting)",
          "Interior unit-fire ignition\n(NFPA × N_units)"]
medians = [
    p2["hazard_decomposition"]["wildfire_direct_median"],
    p2["hazard_decomposition"]["wildfire_indirect_median"],
    p2["hazard_decomposition"]["interior_ignition_median"],
]
p95s = [m * 5 for m in medians]  # approx from Phase 2 plot data
colors = ["#1f77b4", "#ff7f0e", "#d62728"]
xpos = np.arange(3)
bars = ax2.bar(xpos, medians, color=colors, alpha=0.85)
for i, (m, lbl) in enumerate(zip(medians, labels)):
    ax2.text(i, m * 1.3, f"{m:.1e}", ha='center', va='bottom', fontsize=11, fontweight='bold')
ax2.set_xticks(xpos); ax2.set_xticklabels(labels)
ax2.set_ylabel("Annual P(structural loss) — contribution by mechanism")
ax2.set_yscale("log")
ax2.set_title("Hazard decomposition — interior ignition dominates by ~230×")
ax2.set_ylim(1e-6, 5e-3)
# Annotation
ax2.annotate("Phase 1 modeled\nonly these two →",
             xy=(0.5, 1e-4), xytext=(0.5, 4e-4),
             ha='center', fontsize=9, color="#666",
             arrowprops=dict(arrowstyle='->', color="#666"))
ax2.annotate("Phase 2 added this →\nIt dwarfs the wildfire risk",
             xy=(2, 1.6e-3), xytext=(1.2, 5e-3),
             ha='center', fontsize=9, color="#666",
             arrowprops=dict(arrowstyle='->', color="#666"))
plt.tight_layout()
plot2_b64 = fig_to_b64(fig2)
plt.close()

# === Plot 3: Premium vs expected/tail loss ===
fig3, ax3 = plt.subplots(figsize=(11, 5.5))
premiums = [PREM * h for h in HORIZONS]
e_loss_p3 = [p3_dt[h]["E_loss"] for h in HORIZONS]
e_loss_real = [p3_dt[h]["E_loss"] * 6 for h in HORIZONS]  # reality-adjusted central
p99_p3 = [p3_dt[h]["P99_loss"] for h in HORIZONS]
x = np.arange(len(HORIZONS))
w = 0.18
ax3.bar(x - 1.5*w, premiums,    w, label="Cumulative premium (if HOA voted yes)", color="#9467bd")
ax3.bar(x - 0.5*w, e_loss_p3,   w, label="E[loss] — Phase 3 model",               color="#9edae5")
ax3.bar(x + 0.5*w, e_loss_real, w, label="E[loss] — reality-adjusted (model × 6)", color="#2ca02c")
ax3.bar(x + 1.5*w, p99_p3,      w, label="P99 loss (the tail)",                    color="#d62728")
ax3.axhline(SHELL, linestyle="--", color="black", alpha=0.5, label="Shell basis ($334K)")
ax3.set_xticks(x); ax3.set_xticklabels([f"{h} yrs" for h in HORIZONS])
ax3.set_ylabel("USD")
ax3.set_title("Premium vs expected & tail loss — even reality-adjusted, premium ≫ E[loss]")
ax3.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"${v/1000:.0f}K"))
ax3.legend(loc="upper left", fontsize=9)
# Annotate ratios
for i, h in enumerate(HORIZONS):
    ratio = premiums[i] / max(e_loss_real[i], 1)
    ax3.text(i, max(premiums[i], 1e3) * 1.05, f"{ratio:.0f}× margin",
             ha='center', fontsize=9, color="#444", fontweight='bold')
plt.tight_layout()
plot3_b64 = fig_to_b64(fig3)
plt.close()

# === Plot 4: Decision-space chart (tail-aversion vs decision) ===
fig4, ax4 = plt.subplots(figsize=(11, 5.5))
# X axis: maximum loss household can absorb (= tail-aversion threshold)
loss_capacity = np.linspace(50_000, 500_000, 100)
# Y axis: probability that loss exceeds this capacity
# Use reality-adjusted 15-yr P(loss) range
# Simple model: P(loss > X) = P_total_loss × indicator(SHELL > X)
# For range visualization, use multiple P(total loss) scenarios
scenarios = [
    ("Phase 3 model (lower bound)",  p3_dt[15]["p_loss_median"],     "#9edae5"),
    ("Reality-adjusted central",      p3_dt[15]["p_loss_median"] * 6, "#2ca02c"),
    ("Reality-adjusted upper",        p3_dt[15]["p_loss_median"] * 12, "#d62728"),
]
for label, p_loss, color in scenarios:
    # P(loss exceeds capacity) = p_loss if capacity < shell else 0
    y_vals = [p_loss if x < SHELL else 0 for x in loss_capacity]
    ax4.plot(loss_capacity, y_vals, label=f"{label} (P_loss = {p_loss:.1%})",
             color=color, linewidth=2)
ax4.axvline(SHELL, linestyle="--", color="black", alpha=0.6, label="Shell basis ($334K)")
ax4.fill_betweenx([0, 0.30], 0, SHELL, color="#ffe5e5", alpha=0.4)
ax4.text(150_000, 0.27, "Tail risk uninsurable\nfrom household reserves",
         ha='center', fontsize=10, color="#a00")
ax4.fill_betweenx([0, 0.30], SHELL, 500_000, color="#e5f5e5", alpha=0.4)
ax4.text(420_000, 0.27, "Self-insurance defensible",
         ha='center', fontsize=10, color="#080")
ax4.set_xlabel("Maximum loss household can absorb (USD)")
ax4.set_ylabel("15-yr probability of total structural loss")
ax4.set_title("Decision space — when does self-insurance become uncomfortable?")
ax4.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"${v/1000:.0f}K"))
ax4.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax4.legend(loc="upper right", fontsize=9)
ax4.set_xlim(50_000, 500_000)
ax4.set_ylim(0, 0.30)
plt.tight_layout()
plot4_b64 = fig_to_b64(fig4)
plt.close()

# === HTML memo ===
NOW = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
ADDR = "759 Boulder Ct, Stateline, NV 89449"

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Wildfire Self-Insurance Decision Memo — 759 Boulder Ct</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    max-width: 1100px;
    margin: 32px auto;
    padding: 0 24px;
    color: #222;
    background: #fafafa;
    line-height: 1.55;
  }}
  h1 {{ font-size: 28px; margin-bottom: 4px; color: #111; }}
  h2 {{ font-size: 19px; border-bottom: 2px solid #d62728; padding-bottom: 4px; margin-top: 32px; color: #222; }}
  h3 {{ font-size: 15px; margin-top: 24px; color: #444; }}
  .subhead {{ color: #666; font-size: 13px; margin-bottom: 24px; }}
  .verdict {{
    background: #f3f9f3;
    border-left: 5px solid #2ca02c;
    padding: 18px 22px;
    margin: 24px 0;
    border-radius: 4px;
  }}
  .verdict .v-head {{ font-size: 16px; font-weight: 600; color: #1a5e1a; margin-bottom: 6px; }}
  .verdict .v-body {{ font-size: 14px; color: #222; }}
  .caveat {{
    background: #fdf6e3;
    border-left: 5px solid #cb9e1f;
    padding: 14px 20px;
    margin: 18px 0;
    font-size: 13.5px;
    border-radius: 4px;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 14px 0;
    font-size: 13px;
  }}
  th {{
    background: #2c3e50;
    color: #fff;
    padding: 8px 10px;
    text-align: left;
    font-weight: 600;
    font-size: 12px;
  }}
  td {{ padding: 7px 10px; border-bottom: 1px solid #eee; }}
  tr:nth-child(even) td {{ background: #f7f7f7; }}
  .num {{ font-family: 'SF Mono', Menlo, monospace; font-size: 12px; text-align: right; }}
  .verdict-cell {{ font-weight: 600; }}
  .v-good {{ color: #1a5e1a; }}
  .v-borderline {{ color: #b07000; }}
  .v-bad {{ color: #a00; }}
  img.chart {{ max-width: 100%; height: auto; margin: 8px 0 24px; border: 1px solid #ddd; border-radius: 4px; }}
  .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 22px; }}
  .footer {{ font-size: 11px; color: #888; margin-top: 40px; border-top: 1px solid #ddd; padding-top: 12px; }}
  .key {{ font-weight: 600; color: #d62728; }}
  ul li {{ margin-bottom: 6px; }}
  .summary-box {{
    background: #eef4fb;
    border: 1px solid #bcd2eb;
    padding: 12px 16px;
    border-radius: 4px;
    margin: 14px 0;
    font-size: 13.5px;
  }}
</style>
</head>
<body>

<h1>Wildfire Self-Insurance Decision Memo</h1>
<div class="subhead">
  <b>Property:</b> {ADDR} &nbsp;|&nbsp;
  <b>Shell basis:</b> ${SHELL:,} &nbsp;|&nbsp;
  <b>HOA quote (per owner):</b> ${PREM:,}/yr (voted down) &nbsp;|&nbsp;
  <b>Horizon:</b> 5 / 10 / 15 yrs &nbsp;|&nbsp;
  Generated {NOW}
</div>

<div class="verdict">
  <div class="v-head">Recommendation: Self-insure — the HOA vote was rational</div>
  <div class="v-body">
    Across every reasonable adjustment of my model, <b>cumulative premium ($360K over 15 yrs) exceeds reality-adjusted expected loss by ≥4×.</b>
    Self-insurance is EV-rational at all horizons. <b>The decision pivots on whether you can absorb the $334K tail outcome</b> —
    a 6–25% real-world probability over 15 yrs once the model is reconciled to empirical insurance loss-cost data.
    If a one-time $334K hit would be financially survivable, self-insure. If not, the EV-negative insurance premium becomes
    a defensible tail-aversion purchase.
  </div>
</div>

<h2>1. The decision table</h2>
<table>
<tr><th>Horizon</th><th>Phase</th><th class="num">P(loss) median</th><th class="num">5–95% band</th><th class="num">E[loss]</th><th class="num">P99 loss</th><th class="num">Cum. premium</th><th>Verdict</th></tr>
"""

for h in HORIZONS:
    r1 = p1_dt[h]; r2 = p2_dt[h]; r3 = p3_dt[h]
    for tag, r in [("P1", r1), ("P2", r2), ("P3", r3)]:
        verdict_class = "v-good" if "Self-insure" in r["verdict"] and "modest" not in r["verdict"] else "v-borderline" if "Borderline" in r["verdict"] else "v-good"
        html += f'<tr><td>{h} yrs</td><td>{tag}</td><td class="num">{r["p_loss_median"]:.2%}</td><td class="num">{r["p_loss_p5"]:.2%}–{r["p_loss_p95"]:.2%}</td><td class="num">${r["E_loss"]:,.0f}</td><td class="num">${r["P99_loss"]:,.0f}</td><td class="num">${r["cum_premium"]:,.0f}</td><td class="verdict-cell {verdict_class}">{r["verdict"]}</td></tr>\n'

html += f"""
</table>

<div class="caveat">
  <b>Reality-adjusted reading:</b> CA FAIR Plan empirical loss-cost data for Tahoe-area dwellings suggests my Phase 3 model
  underestimates absolute risk by <b>~5–12×</b> (mostly because the model is binary-total-loss only, doesn't capture
  partial losses or cat-event correlation). After adjusting: <b>15-yr expected loss ≈ $20K–$83K, real 15-yr P(loss) ≈ 6–25%.</b>
  The verdict above doesn't change — premium-to-E[loss] ratio remains ≥4× — but the tail probability is materially higher
  than the model alone implies.
</div>

<h2>2. Key visuals</h2>

<h3>Model evolution: Phase 1 → Phase 3 + reality-adjusted band</h3>
<img class="chart" src="data:image/png;base64,{plot1_b64}" alt="Phase evolution survival curves">
<p style="font-size:12.5px; color:#555;">Phase 1 (blue, dashed) ran on wildfire alone — too optimistic. Phase 2 (green) added DINS-fitted vulnerability + interior ignition — caught the dominant risk. Phase 3 (red) added Cal-Adapt climate ensemble — slightly lower than P2 because climate doesn't modulate interior ignition. Orange band = empirical reconciliation against CA FAIR Plan loss-cost data, where the truth most likely lies.</p>

<h3>What's actually driving the risk: it's not wildfire</h3>
<img class="chart" src="data:image/png;base64,{plot2_b64}" alt="Hazard decomposition">
<p style="font-size:12.5px; color:#555;">Interior unit-fire ignition (any unit's kitchen/electrical fire spreading to the structural shell) is ~230× larger than the direct wildfire risk for this parcel. The WRC analysis was right that the parcel sits in a low-BP cell; Phase 1 was wrong to stop there. The biggest risk to a multi-unit 1980 wood-frame condo is not the fire outside — it's the fire inside.</p>

<h3>Premium vs expected & tail loss</h3>
<img class="chart" src="data:image/png;base64,{plot3_b64}" alt="Premium break-even">
<p style="font-size:12.5px; color:#555;">Even after reality-adjusting expected loss upward by 6× from the Phase 3 model, the cumulative premium dwarfs expected loss at every horizon — the dark green bars are 5–11× smaller than the purple premium bars. But the red bars (P99 = full $334K shell loss) intersect cumulative premium at year 14, which is the tail-aversion crossover.</p>

<h3>The decision space — when does self-insurance become uncomfortable?</h3>
<img class="chart" src="data:image/png;base64,{plot4_b64}" alt="Decision space">
<p style="font-size:12.5px; color:#555;">The horizontal axis is "how much loss can the household absorb without crisis?" The line drops to zero at $334K because that's the maximum possible loss — above that level, the tail is fully self-insurable. Below it, the tail is the real exposure. The three lines bracket the model-vs-reality uncertainty range. <b>If your loss-absorption capacity ≥ $334K, self-insure with high confidence. If it's well below, the EV-negative insurance is defensible as tail aversion.</b></p>

<h2>3. Insurance benchmark reconciliation</h2>
<div class="summary-box">
The $24K quote is at the top of the empirical range, not an outlier.
<table style="margin: 8px 0 0">
<tr><th>Source</th><th class="num">Benchmark for $334K Tahoe condo shell</th></tr>
<tr><td>CA FAIR Plan 2026 Tahoe dwelling-fire (post +43% Oct 2026 hike)</td><td class="num">$5K – $10K / yr</td></tr>
<tr><td>HOA master-policy uplift (~1.4× dwelling-fire baseline)</td><td class="num">$7K – $14K / yr</td></tr>
<tr><td>Carrier exit-pricing era (1.5–2.5× actuarial loading)</td><td class="num">$10K – $20K / yr</td></tr>
<tr><td><b>Your HOA quote</b></td><td class="num"><b>$24K / yr</b></td></tr>
</table>
</div>

<h3>Where the remaining gap comes from</h3>
<ul style="font-size: 13.5px;">
  <li><b>Partial-loss claims (1.5–2× factor):</b> insurer claims data includes $50K kitchen fires, smoke damage; my model is binary total-loss only.</li>
  <li><b>Catastrophic event correlation (1.3× factor):</b> insurers must capitalize for Caldor-class clustering events that hit all WUI condos at once.</li>
  <li><b>FAIR Plan self-selection (1.2–1.5× factor):</b> FAIR Plan covers the buildings that couldn't get standard insurance — average condition is worse than my "expected" multiplier.</li>
  <li><b>Residual model uncertainty (1.0–1.2× factor):</b> NFPA interior fire rate could be slightly higher for 1980 buildings; hardening retrofit assumed slightly optimistic.</li>
</ul>

<h2>4. When to revisit this decision</h2>
<ul style="font-size: 13.5px;">
  <li><b>Year 5 trigger:</b> if Tahoe area sees another Caldor-class event within 30 miles. Empirical loss-cost data will spike for the region; re-vote becomes warranted.</li>
  <li><b>Year 7 trigger:</b> if the building manager retires and the HOA loses its informal defense capability (sprinklers + defensible space maintenance lapse).</li>
  <li><b>Any-year trigger:</b> if the master HOA policy market re-opens at &lt;$15K/yr per owner. The current quote at $24K is justifiable but expensive; the actuarial sweet spot is closer to $10–14K.</li>
  <li><b>Personal trigger:</b> if your household financial position changes such that a $334K loss is no longer survivable.</li>
</ul>

<h2>5. Honest limits of this analysis</h2>
<ol style="font-size: 13.5px;">
  <li><b>CA → NV transfer:</b> DINS is California-only. The Phase 2.5 Bayesian model uses population-level partial pooling to mitigate this, but NV fire physics may differ at the margin.</li>
  <li><b>FSim 2020 LANDFIRE fuels</b> pre-date Caldor (2021) and the rapid Sierra fuel-load increase. Burn probability is likely understated for this parcel.</li>
  <li><b>Linear-trend Weibull / CMIP5 ensemble</b> doesn't capture non-linear climate acceleration past 2040 — beyond the modeling horizon but conservative.</li>
  <li><b>Binary loss model:</b> by design (spec). Real losses are a continuous distribution; insurer claims data ≠ my model output.</li>
  <li><b>N_units uncertainty:</b> drawn uniform from [4, 16]. Confirming the actual number would tighten the interior-ignition term.</li>
</ol>

<h2>6. Artifacts</h2>
<table>
<tr><th>Artifact</th><th>Location</th></tr>
<tr><td>End-to-end notebook (Phase 1+2+2.5+3)</td><td><code>fevm-wildfire-risk/Users/layla.yang@databricks.com/wildfire_risk_v2</code></td></tr>
<tr><td>Decision tables (P1 / P2 / P3)</td><td><code>wildfire_risk_catalog.gold.decision_table_phase2</code></td></tr>
<tr><td>Monte Carlo dollar distribution</td><td><code>wildfire_risk_catalog.gold.mc_dollar_distribution</code></td></tr>
<tr><td>CAL FIRE DINS bronze (132,522 rows)</td><td><code>wildfire_risk_catalog.bronze.dins_raw</code></td></tr>
<tr><td>FSim parcel sample bronze</td><td><code>wildfire_risk_catalog.bronze.fsim_parcel_sample</code></td></tr>
</table>

<div class="footer">
Model: Phase 1 (FSim wildfire) → Phase 2 (DINS-fitted vulnerability + NFPA interior ignition) → Phase 2.5 (Bayesian hierarchical multiplier with CA county partial pooling) → Phase 3 (Cal-Adapt LOCA CMIP5 climate ensemble).
Decision framework: expected-value comparison (premium vs E[loss]) + tail-aversion threshold (P99 vs household loss-absorption capacity).
Data sources: USFS Wildfire Risk to Communities, CAL FIRE DINS, NFPA US residential fire statistics, Cal-Adapt LOCA-downscaled CMIP5, CA FAIR Plan published rate filings.
</div>

</body>
</html>
"""

with open("/tmp/wildfire_decision_memo.html", "w") as f:
    f.write(html)
print(f"Wrote /tmp/wildfire_decision_memo.html ({len(html):,} bytes)")
print(f"  4 charts embedded as base64 PNG (file is fully self-contained)")
