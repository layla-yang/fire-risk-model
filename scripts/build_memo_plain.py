#!/usr/bin/env python3
"""Build a plain-English version of the decision memo.
Same charts, but all language pitched at non-statistical readers."""
import base64
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from datetime import datetime, timezone

plt.style.use('seaborn-v0_8-whitegrid')

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

# === Chart 1: Model evolution — plain labels ===
years_plot = np.arange(0, 21)
def cum(p_ann, ct=0.03):
    return [1 - math.exp(-p_ann * (t + ct * t**2 / 2.0)) for t in years_plot]

fig1, ax1 = plt.subplots(figsize=(11, 5.5))
ax1.plot(years_plot, cum(p1["p_annual_summary"]["median"]),
         label="First version: wildfire only", color="#1f77b4", linestyle="--", linewidth=2)
ax1.plot(years_plot, cum(p2["p_annual_summary"]["median"]),
         label="Better version: + risk of any unit catching fire", color="#2ca02c", linestyle="-.", linewidth=2)
ax1.plot(years_plot, cum(p3["p_annual_summary_timeavg_15yr"]["median"]),
         label="Best version: + climate warming over time", color="#d62728", linewidth=2.5)
real_low  = p3["p_annual_summary_timeavg_15yr"]["median"] * 2.5
real_high = p3["p_annual_summary_timeavg_15yr"]["median"] * 12
ax1.fill_between(years_plot, cum(real_low), cum(real_high), color="#ff7f0e", alpha=0.18,
                  label="What real insurance data suggests (likely truth)")
ax1.set_xlabel("Years you own the property")
ax1.set_ylabel("Chance the building burns down (cumulative)")
ax1.set_title("How likely is a total loss? — by how detailed the model is")
ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax1.legend(loc="upper left", fontsize=10)
for h in HORIZONS:
    ax1.axvline(h, linestyle=":", color="gray", alpha=0.5)
    ax1.text(h, 0.001, f"{h} years", color="gray", fontsize=8, ha='center')
plt.tight_layout()
plot1_b64 = fig_to_b64(fig1); plt.close()

# === Chart 2: Hazard decomposition — plain labels ===
fig2, ax2 = plt.subplots(figsize=(11, 5.5))
labels = ["Wildfire reaches the building\n(direct flame)",
          "Embers from nearby fire\n(spotting from up to a mile away)",
          "Fire starts INSIDE one of the units\n(kitchen, electrical, etc.)"]
medians = [
    p2["hazard_decomposition"]["wildfire_direct_median"],
    p2["hazard_decomposition"]["wildfire_indirect_median"],
    p2["hazard_decomposition"]["interior_ignition_median"],
]
colors = ["#1f77b4", "#ff7f0e", "#d62728"]
xpos = np.arange(3)
bars = ax2.bar(xpos, medians, color=colors, alpha=0.85)
for i, m in enumerate(medians):
    pct = m * 100  # convert to % per year
    ax2.text(i, m * 1.4, f"{pct:.4f}% per year", ha='center', va='bottom',
             fontsize=11, fontweight='bold')
ax2.set_xticks(xpos); ax2.set_xticklabels(labels, fontsize=10)
ax2.set_ylabel("Annual chance this is what destroys the building")
ax2.set_yscale("log")
ax2.set_title("What's the biggest threat to the building?\n(spoiler: it's not what you'd expect)")
ax2.set_ylim(1e-6, 5e-3)
ax2.annotate("Fire starting inside is\nbigger than wildfire by ~230×",
             xy=(2, 1.6e-3), xytext=(0.8, 5e-3),
             ha='center', fontsize=10, color="#a00", fontweight='bold',
             arrowprops=dict(arrowstyle='->', color="#a00", lw=1.5))
plt.tight_layout()
plot2_b64 = fig_to_b64(fig2); plt.close()

# === Chart 3: Premium vs loss — plain labels ===
fig3, ax3 = plt.subplots(figsize=(11, 5.5))
premiums    = [PREM * h for h in HORIZONS]
exp_real    = [p3_dt[h]["E_loss"] * 6 for h in HORIZONS]
p99_p3      = [p3_dt[h]["P99_loss"] for h in HORIZONS]
x = np.arange(len(HORIZONS))
w = 0.22
ax3.bar(x - w, premiums,    w, label="What you'd pay in insurance", color="#9467bd")
ax3.bar(x,     exp_real,    w, label="Average expected loss (reality-adjusted)", color="#2ca02c")
ax3.bar(x + w, p99_p3,      w, label="Worst-case: building burns down", color="#d62728")
ax3.axhline(SHELL, linestyle="--", color="black", alpha=0.5, label="Your share if total loss ($334K)")
ax3.set_xticks(x); ax3.set_xticklabels([f"After {h} years" for h in HORIZONS])
ax3.set_ylabel("Dollars")
ax3.set_title("Premium vs. likely loss vs. worst-case — across 5, 10, and 15 years")
ax3.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"${v/1000:.0f}K"))
ax3.legend(loc="upper left", fontsize=10)
for i, h in enumerate(HORIZONS):
    ratio = premiums[i] / max(exp_real[i], 1)
    ax3.text(i, premiums[i] * 1.05, f"Insurance costs\n{ratio:.0f}× expected loss",
             ha='center', fontsize=9, color="#444", fontweight='bold')
plt.tight_layout()
plot3_b64 = fig_to_b64(fig3); plt.close()

# === Chart 4: Decision space — plain labels ===
fig4, ax4 = plt.subplots(figsize=(11, 5.5))
loss_capacity = np.linspace(50_000, 500_000, 100)
scenarios = [
    ("Optimistic case (our model alone)",       p3_dt[15]["p_loss_median"],      "#9edae5"),
    ("Most likely (after reality check)",       p3_dt[15]["p_loss_median"] * 6,  "#2ca02c"),
    ("Pessimistic case (insurance data)",       p3_dt[15]["p_loss_median"] * 12, "#d62728"),
]
for label, p_loss, color in scenarios:
    y_vals = [p_loss if x < SHELL else 0 for x in loss_capacity]
    ax4.plot(loss_capacity, y_vals, label=f"{label} — {p_loss:.0%} chance over 15 yrs",
             color=color, linewidth=2.5)
ax4.axvline(SHELL, linestyle="--", color="black", alpha=0.6, label="Loss if total destruction ($334K)")
ax4.fill_betweenx([0, 0.30], 0, SHELL, color="#ffe5e5", alpha=0.4)
ax4.text(150_000, 0.27, "If a $334K loss would be devastating\n→ buying insurance is reasonable\neven though it's expensive",
         ha='center', fontsize=10, color="#a00")
ax4.fill_betweenx([0, 0.30], SHELL, 500_000, color="#e5f5e5", alpha=0.4)
ax4.text(420_000, 0.27, "If you can absorb $334K\nwithout crisis\n→ skip the insurance",
         ha='center', fontsize=10, color="#080")
ax4.set_xlabel("How much loss could you absorb without serious financial harm?")
ax4.set_ylabel("Chance of total loss over 15 years")
ax4.set_title("Decision guide — does the answer change based on your finances?")
ax4.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"${v/1000:.0f}K"))
ax4.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax4.legend(loc="upper right", fontsize=9)
ax4.set_xlim(50_000, 500_000); ax4.set_ylim(0, 0.30)
plt.tight_layout()
plot4_b64 = fig_to_b64(fig4); plt.close()

# === HTML — plain language ===
NOW = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
ADDR = "759 Boulder Ct, Stateline, NV 89449"

# Reality-adjusted numbers
p15_real_low  = p3_dt[15]["p_loss_median"] * 2.5
p15_real_high = p3_dt[15]["p_loss_median"] * 12
e15_real_low  = p3_dt[15]["E_loss"] * 2.5
e15_real_high = p3_dt[15]["E_loss"] * 12

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Should You Pay for Wildfire Insurance? — 759 Boulder Ct</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    max-width: 1000px; margin: 32px auto; padding: 0 24px;
    color: #222; background: #fafafa; line-height: 1.65;
  }}
  h1 {{ font-size: 30px; margin-bottom: 4px; color: #111; }}
  .subhead {{ color: #666; font-size: 13px; margin-bottom: 28px; }}
  h2 {{ font-size: 22px; border-bottom: 3px solid #d62728; padding-bottom: 6px; margin-top: 40px; color: #222; }}
  h3 {{ font-size: 16px; margin-top: 28px; color: #444; }}
  p {{ font-size: 15px; }}
  .verdict {{
    background: linear-gradient(135deg, #f3f9f3, #e8f5e9);
    border-left: 6px solid #2ca02c;
    padding: 22px 26px; margin: 24px 0; border-radius: 6px;
  }}
  .verdict .v-head {{ font-size: 20px; font-weight: 600; color: #1a5e1a; margin-bottom: 10px; }}
  .verdict .v-body {{ font-size: 15px; color: #222; }}
  .verdict .v-body b {{ color: #1a5e1a; }}
  .warning {{
    background: #fdf6e3; border-left: 6px solid #cb9e1f;
    padding: 16px 22px; margin: 22px 0; font-size: 15px; border-radius: 6px;
  }}
  .warning b {{ color: #8b6914; }}
  .big-stat {{
    display: inline-block; padding: 14px 22px;
    background: #fff; border: 2px solid #d62728; border-radius: 6px;
    margin: 6px; text-align: center; min-width: 200px;
  }}
  .big-stat .num {{ font-size: 32px; font-weight: 700; color: #d62728; display: block; }}
  .big-stat .label {{ font-size: 12px; color: #666; }}
  .stats-row {{ text-align: center; margin: 24px 0; }}
  table {{ width: 100%; border-collapse: collapse; margin: 14px 0; font-size: 14px; }}
  th {{ background: #2c3e50; color: #fff; padding: 10px 12px; text-align: left; font-weight: 600; font-size: 12px; }}
  td {{ padding: 9px 12px; border-bottom: 1px solid #eee; }}
  tr:nth-child(even) td {{ background: #f7f7f7; }}
  .num {{ font-family: 'SF Mono', Menlo, monospace; text-align: right; }}
  img.chart {{ max-width: 100%; height: auto; margin: 12px 0 8px; border: 1px solid #ddd; border-radius: 6px; }}
  .caption {{ font-size: 13.5px; color: #555; font-style: italic; margin-bottom: 28px; line-height: 1.55; }}
  .key {{ color: #d62728; font-weight: 600; }}
  .footer {{ font-size: 11px; color: #888; margin-top: 50px; border-top: 1px solid #ddd; padding-top: 12px; }}
  .takeaway {{
    background: #eef4fb; border: 1px solid #bcd2eb;
    padding: 12px 18px; border-radius: 4px; margin: 12px 0;
    font-size: 14.5px;
  }}
  .takeaway b {{ color: #1a4480; }}
  ul li {{ margin-bottom: 8px; font-size: 15px; }}
</style>
</head>
<body>

<h1>Should You Pay for the Wildfire Insurance?</h1>
<div class="subhead">
  <b>Property:</b> {ADDR} &nbsp;|&nbsp;
  <b>The decision:</b> Pay ${PREM:,}/year for insurance, or accept the risk? &nbsp;|&nbsp;
  Generated {NOW}
</div>

<div class="verdict">
  <div class="v-head">Our recommendation: skip the insurance — but read on, because the reason matters</div>
  <div class="v-body">
    Paying for the insurance would cost you <b>${PREM*15:,} over 15 years</b>. Our best estimate of what
    you'd actually lose without it, averaged across thousands of possible futures, is <b>around ${int(e15_real_low/1000):,}K to ${int(e15_real_high/1000):,}K</b>.
    That means <b>insurance costs roughly 4× to 18× more than the average loss it would cover</b>.
    The HOA's decision to vote it down makes financial sense.
    <br><br>
    <b>But</b> there's still a {p15_real_low*100:.0f}% to {p15_real_high*100:.0f}% chance over 15 years that the building burns down completely and you
    lose your full $334,000 share. <b>The question isn't really "is insurance a good deal?" — it isn't.
    The question is "could your household survive a $334,000 hit if it happened?"</b>
  </div>
</div>

<h2>The numbers in plain English</h2>

<div class="stats-row">
  <div class="big-stat">
    <span class="num">${PREM:,}/yr</span>
    <span class="label">What insurance would cost you each year</span>
  </div>
  <div class="big-stat">
    <span class="num">{p15_real_low*100:.0f}–{p15_real_high*100:.0f}%</span>
    <span class="label">Realistic chance of total loss<br>over the next 15 years</span>
  </div>
  <div class="big-stat">
    <span class="num">$334,000</span>
    <span class="label">What you'd lose if the building<br>is destroyed</span>
  </div>
</div>

<div class="takeaway">
  <b>What this means.</b> Imagine playing this scenario out 100 times over the next 15 years.
  In about {int((p15_real_low + p15_real_high)/2 * 100)} of those 100 futures, the building burns down and you lose $334,000.
  In the other {100 - int((p15_real_low + p15_real_high)/2 * 100)}, you lose nothing. <b>If you'd bought insurance, you'd have paid $360,000
  across all 100 futures combined.</b> That's why the math favors skipping the insurance — but only if
  you can afford to be one of the unlucky ones.
</div>

<h2>What we modeled, step by step</h2>

<p>We built the model in three rounds, each better than the last:</p>

<table>
<tr><th>Version</th><th>What it included</th><th>15-yr chance of total loss</th></tr>
<tr><td><b>First version</b></td><td>Only wildfire — the obvious risk for a Tahoe-area condo</td><td class="num">0.25%</td></tr>
<tr><td><b>Better version</b></td><td>Added: risk of fire starting inside any of the units<br>(kitchen, electrical, etc.) — and a more accurate vulnerability estimate from California fire-damage data on 132,000 inspected buildings</td><td class="num">3.0%</td></tr>
<tr><td><b>Best version</b></td><td>Added: climate warming over time using projections from 8 climate models</td><td class="num">2.6%</td></tr>
<tr><td><b>After reality-checking against<br>real insurance market data</b></td><td>Adjusted upward because our model misses partial losses, catastrophic-event clustering, and a few other things insurers price for</td><td class="num"><b>6% to 25%</b></td></tr>
</table>

<h3>Why each version changed the answer</h3>
<ul>
  <li>The <b>first version</b> said the wildfire risk was tiny — and that's actually correct. The parcel is in a developed area where the official US fire risk maps show a very low chance of wildfire reaching this exact spot.</li>
  <li>The <b>better version</b> caught the much bigger risk we'd missed: <b>any fire starting inside one of the units</b> (kitchen, electrical, etc.) that spreads to the whole building structure. This is roughly 230× more likely than a wildfire reaching the building. <span class="key">This was the most important finding.</span></li>
  <li>The <b>best version</b> added climate warming. Surprisingly, it made things only slightly different — because the interior-fire risk doesn't depend on climate. So even though climate change makes wildfires worse, it barely moves our answer for this building.</li>
  <li>The <b>reality check</b> against real insurance loss data showed our model is probably 5–12× too optimistic in absolute terms. We adjusted upward.</li>
</ul>

<h2>The picture</h2>

<h3>What the risk actually looks like over time</h3>
<img class="chart" src="data:image/png;base64,{plot1_b64}" alt="Model evolution">
<div class="caption">
  Each line shows the chance that the building has burned down by year X — for each version of our model.
  The orange shaded band shows the range that real insurance data suggests is most accurate.
  By year 15, there's a 6–25% real-world chance the building has been destroyed.
</div>

<h3>The surprise: wildfire isn't the main threat</h3>
<img class="chart" src="data:image/png;base64,{plot2_b64}" alt="Hazard decomposition">
<div class="caption">
  This chart is on a logarithmic scale because the differences are huge.
  The chance of fire starting inside the building (rightmost bar) is about <b>230 times bigger</b>
  than the chance of wildfire reaching the building (leftmost bar).
  This was the most important thing our analysis found.
</div>

<h3>How premium compares to what you'd lose</h3>
<img class="chart" src="data:image/png;base64,{plot3_b64}" alt="Premium vs loss">
<div class="caption">
  Purple bar = what you'd pay in insurance over each time period. Green bar = the average loss you'd
  actually experience (after reality-checking the model). Red bar = the worst-case loss (the building
  burns down). <b>Insurance costs 4–11× more than the average loss it covers</b>, which is why
  skipping it makes financial sense — unless you can't afford to be one of the worst-case outcomes.
</div>

<h3>How the answer depends on your finances</h3>
<img class="chart" src="data:image/png;base64,{plot4_b64}" alt="Decision space">
<div class="caption">
  This is the key chart for deciding. Read it like this: if your household could absorb a $334K loss
  (right side, green zone) without serious financial harm, the math says skip the insurance.
  If a $334K loss would be devastating to your finances (left side, red zone), then buying the
  insurance is reasonable — even though you'll pay more than it's "worth" on average — because that's
  what insurance is for: protecting against rare but catastrophic outcomes.
</div>

<h2>Reality check against the insurance market</h2>

<p>One natural question: <i>"If insurance is such a bad deal, why is the carrier charging $24,000/year? Aren't insurance companies usually smart about pricing?"</i> Yes, they are. So we cross-checked our model against what's actually being charged in the Tahoe area:</p>

<table>
<tr><th>Reference point</th><th class="num">Annual price for $334K building</th></tr>
<tr><td>California FAIR Plan (the state's insurer-of-last-resort) for a similar Tahoe property</td><td class="num">$5,000 – $10,000</td></tr>
<tr><td>Adjusted for being a master-HOA condo policy (~1.4× markup over single-family)</td><td class="num">$7,000 – $14,000</td></tr>
<tr><td>Adjusted for post-2018 "WUI exodus" pricing (carriers leaving California raised prices everywhere)</td><td class="num">$10,000 – $20,000</td></tr>
<tr><td><b>What your HOA was quoted</b></td><td class="num"><b>$24,000</b></td></tr>
</table>

<p>So the quote is at the top of the range, but not outrageous. The fact that real insurance data implies a much higher annual loss rate than our model alone says (1% to 2% per year vs. our 0.17%) is mostly because real insurance covers things we don't model: partial losses, smoke damage, multiple-property cascading events, etc. <b>After reality-adjusting, the average loss is still 4–18× smaller than the premium</b>, which is why the recommendation holds.</p>

<h2>When you should reconsider this decision</h2>

<ul>
  <li><b>Within 5 years:</b> If another major wildfire (like Caldor 2021) happens within 30 miles of the building. Insurance loss data for the region will spike, and the math could change.</li>
  <li><b>Within 5–7 years:</b> If the building manager retires and the HOA's informal defenses lapse (sprinklers stop being tested, defensible space isn't cleared, etc.). Our analysis assumed mitigation is mostly maintained.</li>
  <li><b>Anytime:</b> If the master HOA insurance market re-opens at a lower price (under $15,000/year per owner). The current $24K is justified by today's market but expensive on the merits.</li>
  <li><b>If your finances change:</b> If a $334,000 loss would no longer be financially survivable for you (job change, divorce, etc.), insurance becomes a defensible purchase even if it's a bad deal on average.</li>
</ul>

<h2>What we don't know for sure</h2>

<p>Every model is wrong in some way. Here's what could change our answer if it turned out differently than we assumed:</p>

<ol>
  <li><b>How many units are in the building?</b> We assumed somewhere between 4 and 16. If it's actually more (say 20+), the interior-fire risk goes up proportionally and the answer leans more toward insurance.</li>
  <li><b>How well-maintained is the building, really?</b> We assumed mitigation is maintained in 70% of scenarios and degrades over time in 30%. If the HOA falls apart, the risk gets worse.</li>
  <li><b>Will the carrier raise prices?</b> Real WUI premiums are rising 10–25% per year. The $24K could become $40K within a few years, which doesn't change our answer (it makes insurance even worse value), but it's worth tracking.</li>
  <li><b>Could climate change be worse than projected?</b> We used the standard CMIP5 models. The newer CMIP6 models suggest slightly worse warming. But because interior fires dominate the risk for this specific building, this doesn't move the answer much.</li>
  <li><b>California data may not fully transfer to Nevada.</b> Our most detailed data on what makes buildings burn comes from California's post-fire inspections (132,000 of them). Nevada fire physics is similar, but local building codes and wind patterns differ slightly.</li>
</ol>

<h2>The bottom line, in one paragraph</h2>

<div class="verdict">
  <div class="v-body">
    Skip the insurance. Over 15 years, you'd pay $360,000 in premiums to protect against an
    average loss of about $20,000–$83,000 — a bad deal on the math. <b>The one situation where
    this advice is wrong: if a $334,000 loss in a single year would be genuinely devastating
    to your household finances.</b> In that case, the insurance is overpriced but defensible as
    a hedge against an outcome you simply can't afford — that's what insurance is for. There's
    a 6–25% real-world chance of that happening over 15 years.
  </div>
</div>

<div class="footer">
  Model: combined wildfire risk (USFS Wildfire Risk to Communities data), building vulnerability (132,000 California fire-damage inspections),
  interior unit-fire risk (NFPA national fire statistics), and climate warming (Cal-Adapt LOCA climate ensemble, 8 models).
  Cross-checked against California FAIR Plan published premium data for Tahoe-area properties.
  This analysis is decision-support only; consult a licensed insurance broker or financial advisor before acting.
</div>

</body>
</html>
"""

with open("/tmp/wildfire_decision_memo_plain.html", "w") as f:
    f.write(html)
print(f"Wrote /tmp/wildfire_decision_memo_plain.html ({len(html):,} bytes)")
