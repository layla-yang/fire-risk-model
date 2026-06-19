#!/usr/bin/env python3
"""Build the BUY-OR-NOT memo with empirical fire history maps."""
import base64, json, math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from datetime import datetime, timezone

plt.style.use('seaborn-v0_8-whitegrid')

with open("/tmp/wildfire_p4_results.json")  as f: p4 = json.load(f)
with open("/tmp/empirical_fire_stats.json") as f: emp = json.load(f)
with open("/tmp/wildfire_mc_results.json")  as f: p1 = json.load(f)

SHELL = 334_000
PREM  = 24_000
HORIZONS = [5, 10, 15]
p4_dt = {row["horizon"]: row for row in p4["decision_table"]}

def img_b64(path):
    with open(path,"rb") as f: return base64.b64encode(f.read()).decode()

def fig_to_b64(fig):
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# Pre-built fire history maps
map_closeup_b64 = img_b64("/tmp/wildfire_history_closeup.png")
map_freq_b64    = img_b64("/tmp/wildfire_history_map.png")

# === Chart: Phase 4 cumulative loss curves (focus on 5/10/15 yr) ===
fig, ax = plt.subplots(figsize=(11, 5.5))
horizons_plot = np.arange(0, 21)
# Approximate from MC summary
p_ann_med = p4["annual_p_loss"]["median"]
p_ann_p5  = p4["annual_p_loss"]["p5"]
p_ann_p95 = p4["annual_p_loss"]["p95"]
med  = [1 - math.exp(-p_ann_med * t) for t in horizons_plot]
p5_v = [1 - math.exp(-p_ann_p5  * t) for t in horizons_plot]
p95_v= [1 - math.exp(-p_ann_p95 * t) for t in horizons_plot]
ax.plot(horizons_plot, med, color="#d62728", linewidth=2.5, label="Most likely (median)")
ax.fill_between(horizons_plot, p5_v, p95_v, color="#d62728", alpha=0.20,
                label="Uncertainty range (5–95%)")
for h in HORIZONS:
    pl = p4_dt[h]["p_loss_median"]
    ax.scatter([h], [pl], s=150, color='#d62728', zorder=5, edgecolor='white', linewidth=2)
    ax.annotate(f"{pl*100:.1f}%", (h, pl), xytext=(8, 8), textcoords='offset points',
                fontsize=12, fontweight='bold', color="#d62728")
ax.set_xlabel("Years you would own the property")
ax.set_ylabel("Chance the building burns down (cumulative)")
ax.set_title("If you buy: chance of total structural loss over the holding period")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.legend(loc="upper left", fontsize=11)
for h in HORIZONS:
    ax.axvline(h, linestyle=":", color="gray", alpha=0.5)
plt.tight_layout()
plot_curves_b64 = fig_to_b64(fig)
plt.close()

# === Chart: Modeled vs Empirical risk ===
fig, ax = plt.subplots(figsize=(11, 5.5))
labels = ["WRC FSim modeled BP\n(what model says about parcel)",
          "Empirical fire frequency\nwithin 5 mi of parcel",
          "Empirical fire frequency\nwithin 10 mi of parcel"]
values = [0.00045 * 100,  # FSim 0.045%
          emp["frequency_by_radius"]["5"]["annual_rate"] * 100,
          emp["frequency_by_radius"]["10"]["annual_rate"] * 100]
colors = ["#1f77b4", "#ff7f0e", "#d62728"]
bars = ax.barh(range(3), values, color=colors)
ax.set_yticks(range(3)); ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel("% of years (1984-2025) — annual rate")
ax.set_title("Modeled wildfire risk vs. what actually happens — they disagree by ~150x")
ax.set_xscale("log")
for i, v in enumerate(values):
    ax.text(v * 1.15, i, f"{v:.2f}%", va='center', fontsize=12, fontweight='bold')
ax.set_xlim(0.01, 200)
plt.tight_layout()
plot_compare_b64 = fig_to_b64(fig)
plt.close()

# === Chart: $ outcome distribution at 15 yrs ===
fig, ax = plt.subplots(figsize=(10, 5.5))
n_total = 10000
p_loss_15 = p4_dt[15]["p_loss_median"]
n_loss = int(n_total * p_loss_15)
n_no_loss = n_total - n_loss
counts = [n_no_loss, n_loss]
labels_p = ["Building intact\n($0 loss)", f"Total loss\n(${SHELL/1000:.0f}K hit)"]
bars = ax.bar(labels_p, counts, color=["#2ca02c", "#d62728"], width=0.6)
for i, c in enumerate(counts):
    ax.text(i, c + 200, f"{c:,}\n({c/n_total:.1%})", ha='center', fontsize=12, fontweight='bold')
ax.set_ylabel("Out of 10,000 simulated 15-year futures")
ax.set_title("If you buy and hold 15 years: distribution of outcomes")
ax.set_ylim(0, n_total * 1.1)
plt.tight_layout()
plot_outcomes_b64 = fig_to_b64(fig)
plt.close()

NOW = datetime.now(timezone.utc).strftime("%Y-%m-%d")
ADDR = "759 Boulder Ct, Stateline, NV 89449"

p15 = p4_dt[15]["p_loss_median"]
p10 = p4_dt[10]["p_loss_median"]
p5  = p4_dt[5]["p_loss_median"]

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Should You Buy This Property? — 759 Boulder Ct</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         max-width: 1100px; margin: 32px auto; padding: 0 24px;
         color: #222; background: #fafafa; line-height: 1.65; }}
  h1 {{ font-size: 32px; margin-bottom: 4px; color: #111; }}
  .subhead {{ color: #666; font-size: 13px; margin-bottom: 28px; }}
  h2 {{ font-size: 22px; border-bottom: 3px solid #3b82f6; padding-bottom: 6px; margin-top: 40px; color: #222; }}
  h3 {{ font-size: 16px; margin-top: 24px; color: #444; }}
  .verdict {{ background: linear-gradient(135deg, #f0f7ff, #dbeafe);
              border-left: 6px solid #3b82f6; padding: 22px 26px; margin: 24px 0; border-radius: 6px; }}
  .verdict .head {{ font-size: 22px; font-weight: 600; color: #1e3a8a; margin-bottom: 12px; }}
  .verdict .body {{ font-size: 16px; }}
  .verdict.green {{ background: linear-gradient(135deg, #f3f9f3, #e8f5e9);
                    border-left: 6px solid #2ca02c; }}
  .verdict.green .head {{ color: #1a5e1a; }}
  .stat-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin: 24px 0; }}
  .stat-card {{ background: white; padding: 18px; border: 2px solid #3b82f6; border-radius: 6px; text-align: center; }}
  .stat-card .num {{ display: block; font-size: 38px; font-weight: 700; color: #1d4ed8; line-height: 1; }}
  .stat-card .lbl {{ font-size: 12px; color: #666; margin-top: 8px; }}
  .correction-box {{ background: #fff8e1; border-left: 5px solid #ffa726;
                     padding: 14px 20px; margin: 20px 0; border-radius: 4px; font-size: 14px; }}
  .correction-box b {{ color: #b25600; }}
  table {{ width: 100%; border-collapse: collapse; margin: 14px 0; font-size: 14px; }}
  th {{ background: #2c3e50; color: #fff; padding: 10px 12px; text-align: left; font-weight: 600; font-size: 12px; }}
  td {{ padding: 9px 12px; border-bottom: 1px solid #eee; }}
  tr:nth-child(even) td {{ background: #f7f7f7; }}
  .num {{ font-family: 'SF Mono', Menlo, monospace; text-align: right; }}
  img.chart {{ max-width: 100%; height: auto; margin: 12px 0 8px; border: 1px solid #ddd; border-radius: 6px; }}
  .caption {{ font-size: 13.5px; color: #555; font-style: italic; margin-bottom: 24px; }}
  .key {{ color: #1d4ed8; font-weight: 700; }}
  ul li {{ margin-bottom: 8px; font-size: 15px; }}
  .footer {{ font-size: 11px; color: #888; margin-top: 50px; border-top: 1px solid #ddd; padding-top: 12px; }}
</style>
</head>
<body>

<h1>Should You Buy This Property?</h1>
<div class="subhead">
  <b>Property:</b> {ADDR} &nbsp;|&nbsp;
  <b>Your share if total loss:</b> $334,000 &nbsp;|&nbsp;
  <b>Analysis date:</b> {NOW}
</div>

<div class="correction-box">
  <b>This memo replaces the earlier "self-insure vs insure" analysis.</b> Two corrections changed the numbers materially:
  (1) I rebuilt the interior-fire risk after auditing the underlying NFPA data — my first pass overstated it ~6×;
  (2) I cross-checked the wildfire risk against <b>41 years of actual fire history</b> within 50 miles of the parcel,
  which revealed that the official USFS model dramatically understates real risk for this area.
</div>

<div class="verdict">
  <div class="head">The honest read: this is a high-wildfire-risk parcel</div>
  <div class="body">
    Empirical fire history within 50 miles of 759 Boulder Ct since 1984 shows fires in <b>40 of 41 years</b>, including fires within 0.6 miles (Gondola 2002), 1.4 miles (Autumn Hills 1996), 6 miles (Caldor 2021, which destroyed ~1,000 structures), and 7.9 miles (Angora 2007, which destroyed 254 homes).
    <br><br>
    Over 15 years of ownership, the realistic probability that the building burns down is <span class="key">about {p15*100:.1f}%</span>, with a credible range of {p4_dt[15]['p_loss_p5']*100:.1f}% to {p4_dt[15]['p_loss_p95']*100:.1f}%. That's roughly <b>1 chance in {int(1/p15)}</b>.
    <br><br>
    <b>Whether to buy depends on whether you can absorb a ~1-in-{int(1/p15)} chance of losing $334K over 15 years — on top of the property's other costs and the rising insurance unavailability.</b>
  </div>
</div>

<h2>The headline numbers</h2>

<div class="stat-grid">
  <div class="stat-card">
    <span class="num">{p5*100:.1f}%</span>
    <div class="lbl">Chance of total loss<br>within <b>5 years</b></div>
  </div>
  <div class="stat-card">
    <span class="num">{p10*100:.1f}%</span>
    <div class="lbl">Chance of total loss<br>within <b>10 years</b></div>
  </div>
  <div class="stat-card">
    <span class="num">{p15*100:.1f}%</span>
    <div class="lbl">Chance of total loss<br>within <b>15 years</b></div>
  </div>
</div>

<table>
<tr><th>Holding period</th><th class="num">Most likely chance of total loss</th><th class="num">5–95% range</th><th class="num">Average $ loss</th><th class="num">Worst-case 1% scenario</th></tr>
"""
for h in HORIZONS:
    r = p4_dt[h]
    html += f'<tr><td><b>{h} years</b></td><td class="num">{r["p_loss_median"]*100:.2f}%</td><td class="num">{r["p_loss_p5"]*100:.2f}% – {r["p_loss_p95"]*100:.2f}%</td><td class="num">${r["E_loss"]:,.0f}</td><td class="num">${r["P99_loss"]:,.0f}</td></tr>\n'

html += f"""
</table>

<h2>The fire history map — what really happens around this parcel</h2>

<img class="chart" src="data:image/png;base64,{map_closeup_b64}" alt="Fire history closeup">
<div class="caption">
  Every red/orange shape is a fire that burned ≥100 acres within 20 miles of the parcel since 1984.
  Darker red = more recent. The <b>Gondola fire (2002, 643 acres) burned within 0.6 miles</b>;
  <b>Autumn Hills (1996, 3,805 acres) within 1.4 miles</b>; <b>Caldor (2021, 221,786 acres)</b> within 6 miles
  and destroyed ~1,000 structures in Grizzly Flats; <b>Angora (2007, 3,070 acres)</b> within 7.9 miles
  and destroyed 254 homes in South Lake Tahoe.
</div>

<img class="chart" src="data:image/png;base64,{map_freq_b64}" alt="Fire frequency by radius">
<div class="caption">
  <b>Left:</b> the 50-mile context — 394 separate fires ≥100 acres since 1984.
  <b>Right:</b> the empirical numbers — within 5 miles of the parcel, a fire has occurred in <b>7% of years</b>;
  within 10 miles, <b>17%</b>; within 20 miles, <b>63%</b>; within 30 miles, <b>85%</b>.
  The blue line shows what the USFS Wildfire Risk to Communities model said for the parcel: <b>0.04% per year</b>.
  The model is roughly <b>150× lower than what history actually shows</b> for fires within 10 miles.
</div>

<div class="correction-box">
  <b>Why the official US model says "low risk" when fires happen here all the time.</b>
  The WRC FSim model is a 10,000-year stochastic simulation using LANDFIRE 2020 fuel data and historical weather.
  Three things make it understate risk for this parcel: (1) the parcel itself sits on a "developed" cell with no fuel,
  so its fire-spread potential is filled in by averaging surrounding cells; (2) it predates Caldor 2021 and the
  recent Sierra fire escalation; (3) it doesn't model human-caused ignitions (powerlines, vehicles, vegetation
  management) that drive most WUI fires. The empirical history is the better anchor.
</div>

<h2>What the corrected model says about the next 15 years</h2>

<img class="chart" src="data:image/png;base64,{plot_curves_b64}" alt="Loss probability curves">
<div class="caption">
  Each year you own the property, the chance the building has burned down by then grows. After 5 years: about {p5*100:.1f}%.
  After 10 years: {p10*100:.1f}%. After 15 years: {p15*100:.1f}%. The shaded band shows the uncertainty range — the lower end
  assumes the empirical wildfire risk is overstated; the upper end assumes Caldor-class events keep happening at the recent
  rate. The middle line is the most likely scenario.
</div>

<img class="chart" src="data:image/png;base64,{plot_outcomes_b64}" alt="15-yr outcomes">
<div class="caption">
  Imagine running the next 15 years 10,000 times. About <b>{int((1-p15)*10000):,}</b> times the building is fine
  and you walk away. About <b>{int(p15*10000):,}</b> times the building burns down and you lose your $334,000 share.
  <b>The question is whether you can afford to be one of the {int(p15*10000):,}.</b>
</div>

<h2>Why the wildfire risk is so much higher than the first model said</h2>

<img class="chart" src="data:image/png;base64,{plot_compare_b64}" alt="Model vs reality">
<div class="caption">
  The blue bar is what the official US model said: 0.04% chance of fire reaching this parcel per year.
  The orange bar is the actual empirical rate of fires within 5 miles of the parcel: 7% per year (5 fires in 41 years).
  The red bar is for fires within 10 miles: 17% per year. <b>The model and empirical reality disagree by about 150×.</b>
  We used the empirical rate, not the model, for the corrected analysis.
</div>

<h2>What this means for your buying decision</h2>

<h3>The case AGAINST buying</h3>
<ul>
  <li><b>Real wildfire risk to the building is meaningful</b> — somewhere between 1% and 15% chance of total loss over 15 years, most likely ~{p15*100:.0f}%. That's not "tail risk" — it's a real probability.</li>
  <li><b>Insurance is essentially unavailable</b> — the HOA already voted down a $24K/year policy because it was prohibitive. If you buy, you're absorbing the full $334K downside yourself.</li>
  <li><b>Resale risk: insurance unavailability propagates to value</b> — buyers in 5–10 years may face even worse insurance markets. Properties in unisurable zones are starting to trade at discounts of 10–30%.</li>
  <li><b>Climate trend is unfavorable</b> — the Tahoe basin has had 4 major fires in the last 18 years (Angora 2007, multiple smaller, Caldor 2021, Tamarack 2021). Trend lines aren't improving.</li>
  <li><b>You're inheriting other people's risk decisions</b> — the dual-HOA structure means defensible space and structural mitigation depend on HOA decisions you don't control.</li>
</ul>

<h3>The case FOR buying</h3>
<ul>
  <li><b>The building has hardened mitigation</b> — retired-firefighter manager, sprinkler systems, maintained defensible space. Building's actual destruction rate is likely lower than average for the area.</li>
  <li><b>Tahoe-area real estate has resilient long-term value</b> for vacation/STR markets even with fire risk priced in.</li>
  <li><b>Most likely outcome (~96% of scenarios over 15 years) is no total loss</b>. The expected dollar loss is ${p4_dt[15]['E_loss']/1000:.1f}K — small compared to typical property purchase prices.</li>
  <li><b>If purchase price is below market value</b> by enough to compensate for the uninsured-shell risk, the math could still work.</li>
</ul>

<h3>The pragmatic test</h3>
<ol>
  <li><b>Could your household absorb a $334K loss in one year without serious financial harm?</b><br>
      If NO: the risk-adjusted purchase requires a substantial price discount (think 10–25% below otherwise-comparable insured properties) to compensate. Without that discount, walk away.<br>
      If YES: the math is more manageable. Average loss of ${p4_dt[15]['E_loss']/1000:.1f}K over 15 years is similar to other property-ownership costs (HOA fees, taxes, maintenance) — it just shows up as a low-probability lump.</li>
  <li><b>Is the purchase price already discounted for the wildfire risk?</b> Compare against Tahoe properties with insurance available. If this unit is selling at parity, it's overpriced for the risk you're taking.</li>
  <li><b>How is your STR revenue projection affected if you're un-insurable?</b> Standard STR loans require insurance. If you're paying cash and self-insuring, your exit options narrow.</li>
</ol>

<h2>Honest limits of this analysis</h2>
<ol>
  <li><b>The empirical wildfire rate has wide uncertainty.</b> 5 fires within 5 miles over 41 years is a small sample. The true annual rate could be 3% or 12% depending on which fires you count and how you define "near."</li>
  <li><b>P(building destroyed | fire near) is the biggest model uncertainty.</b> We use 10–30% — the range from DINS data for multi-residence buildings with hardening. Could be higher in extreme fire weather (Caldor saw &gt;90% destruction in Grizzly Flats).</li>
  <li><b>Interior-fire risk is now based on NFPA national multi-family fire statistics</b> rather than my original Beta(2,26) which was unjustifiably high. The corrected number is grounded but still uncertain ±50%.</li>
  <li><b>Climate trajectory uses CMIP5 (slightly dated).</b> CMIP6 projections suggest somewhat worse warming; we're likely on the conservative side.</li>
  <li><b>Resale/value risk isn't modeled here.</b> If insurance markets in the Tahoe basin deteriorate further, your exit price could drop independently of any fire occurring.</li>
</ol>

<h2>If you do buy — what to track</h2>
<ul>
  <li>Annual fire-perimeter activity in El Dorado/Douglas County (NIFC public data) — re-evaluate if you see a Caldor-class event within 20 miles.</li>
  <li>HOA budget and defensible-space maintenance — if these lapse, your risk rises.</li>
  <li>Insurance market for HOA master policies — if a real carrier re-opens at &lt;$15K/year per owner, reconsider buying coverage.</li>
  <li>Building manager's tenure — when they retire, ask the HOA who's taking over the fire-readiness program.</li>
</ul>

<div class="footer">
  Sources: USFS Wildfire Risk to Communities (FSim BP/FLEP), NIFC Interagency Fire Perimeter History 1984-2025,
  CAL FIRE DINS (132,522 structure-outcome records), NFPA US residential fire statistics, Cal-Adapt LOCA CMIP5
  climate ensemble (8 models). Decision-support only; consult licensed financial/real-estate advisors before action.
</div>

</body>
</html>
"""

with open("/tmp/wildfire_buy_or_not_memo.html","w") as f:
    f.write(html)
print(f"Wrote /tmp/wildfire_buy_or_not_memo.html ({len(html):,} bytes)")
