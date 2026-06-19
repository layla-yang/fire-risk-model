#!/usr/bin/env python3
"""Local runner — same math as the Databricks notebook, no pandas/spark dependency.
Reads the cached FSim bronze grid from /tmp/fsim_grid_parcel.json (already pulled this session)."""

import json
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from datetime import datetime, timezone

CONFIG = {
    "parcel": {
        "address": "759 Boulder Ct, Stateline, NV 89449",
        "lat":      38.967876,
        "lon":     -119.887425,
    },
    "financials": {
        "shell_basis_usd":          334_000,
        "premium_saved_annual_usd":  24_000,
    },
    "horizons_years": [5, 10, 15],
    "monte_carlo":   {"iters": 10_000, "random_seed": 42},
    "vulnerability_scenarios": {
        "best_realized": {"prob": 0.20, "mult": 0.30},
        "expected":      {"prob": 0.50, "mult": 0.55},
        "degraded":      {"prob": 0.25, "mult": 0.75},
        "worst":         {"prob": 0.05, "mult": 0.90},
    },
    "hazard_model": {
        "ember_baseline_central":     2.0e-4,
        "ember_baseline_sigma_log":      0.5,
        "climate_trend_central_per_yr": 0.030,
        "climate_trend_sigma":          0.015,
    },
}

SEED = CONFIG["monte_carlo"]["random_seed"]
SHELL = CONFIG["financials"]["shell_basis_usd"]
PREM  = CONFIG["financials"]["premium_saved_annual_usd"]

# Load FSim data from this session's bronze fetch
with open("/tmp/fsim_grid_parcel.json") as f:
    d = json.load(f)
scales = {"BP": 1/100_000, "FLEP4": 1/1_000, "FLEP8": 1/1_000}
grid_scaled = {
    layer: [[(v * scales[layer]) if v is not None else None for v in row] for row in mat]
    for layer, mat in d["raw_values"].items()
}

# Aggregate
bp_samples       = [v for row in grid_scaled["BP"]    for v in row if v is not None]
flep8_burnable   = [v for row in grid_scaled["FLEP8"] for v in row if v not in (None, 0.0)]
flep4_burnable   = [v for row in grid_scaled["FLEP4"] for v in row if v not in (None, 0.0)]

print("=" * 70)
print(f"FSim aggregation:")
print(f"  BP    n={len(bp_samples)}     min={min(bp_samples):.5f}  med={np.median(bp_samples):.5f}  max={max(bp_samples):.5f}")
print(f"  FLEP8 n={len(flep8_burnable)} (burnable only)  min={min(flep8_burnable):.4f}  med={np.median(flep8_burnable):.4f}  max={max(flep8_burnable):.4f}")
print(f"  FLEP4 n={len(flep4_burnable)} (burnable only)  min={min(flep4_burnable):.4f}  med={np.median(flep4_burnable):.4f}  max={max(flep4_burnable):.4f}")

# Vuln PMF
vuln_mults = np.array([s["mult"] for s in CONFIG["vulnerability_scenarios"].values()])
vuln_probs = np.array([s["prob"] for s in CONFIG["vulnerability_scenarios"].values()])
vuln_probs = vuln_probs / vuln_probs.sum()
print(f"\nVulnerability PMF: mults={vuln_mults.tolist()}, probs={vuln_probs.tolist()}")
print(f"  E[vuln_mult] = {(vuln_mults * vuln_probs).sum():.3f}")

# Hazard sampling
def sample_annual_hazard(rng):
    bp     = rng.choice(bp_samples)
    flep8  = rng.choice(flep8_burnable)
    flep4  = rng.choice(flep4_burnable)
    vuln_h = rng.choice(vuln_mults, p=vuln_probs)
    vuln_l = vuln_h * 0.4
    ember  = rng.lognormal(
        mean  = math.log(CONFIG["hazard_model"]["ember_baseline_central"]),
        sigma = CONFIG["hazard_model"]["ember_baseline_sigma_log"],
    ) * vuln_h
    p_destroy_given_fire = flep8 * vuln_h + max(0.0, flep4 - flep8) * vuln_l
    p_direct   = bp * p_destroy_given_fire
    p_indirect = ember
    p_annual   = p_direct + p_indirect
    ct = max(0.0, rng.normal(
        CONFIG["hazard_model"]["climate_trend_central_per_yr"],
        CONFIG["hazard_model"]["climate_trend_sigma"],
    ))
    return p_annual, ct, bp, flep8, flep4, vuln_h, p_direct, p_indirect

def p_cum_constant(p, t):
    return 1.0 - (1.0 - p) ** t

def p_cum_rising(p, t, ct):
    H = p * (t + ct * t**2 / 2.0)
    return 1.0 - math.exp(-H)

# Monte Carlo
rng = np.random.default_rng(SEED)
N   = CONFIG["monte_carlo"]["iters"]
horizons = CONFIG["horizons_years"]

p_annual_arr  = np.zeros(N)
climate_arr   = np.zeros(N)
pcum_const    = {h: np.zeros(N) for h in horizons}
pcum_rise     = {h: np.zeros(N) for h in horizons}
dollar_rise   = {h: np.zeros(N, dtype=int) for h in horizons}
breakdown     = {"bp": np.zeros(N), "flep8": np.zeros(N), "vuln_h": np.zeros(N),
                 "p_direct": np.zeros(N), "p_indirect": np.zeros(N)}

print(f"\nMonte Carlo: N={N}, seed={SEED}")
for i in range(N):
    p_ann, ct, bp, flep8, flep4, vh, pd_, pi_ = sample_annual_hazard(rng)
    p_annual_arr[i]   = p_ann
    climate_arr[i]    = ct
    breakdown["bp"][i] = bp
    breakdown["flep8"][i] = flep8
    breakdown["vuln_h"][i] = vh
    breakdown["p_direct"][i] = pd_
    breakdown["p_indirect"][i] = pi_
    for h in horizons:
        pcum_const[h][i] = p_cum_constant(p_ann, h)
        pcum_rise[h][i]  = p_cum_rising(p_ann, h, ct)
        dollar_rise[h][i] = SHELL if rng.random() < pcum_rise[h][i] else 0

print(f"  P_annual:      median={np.median(p_annual_arr):.2e}  mean={p_annual_arr.mean():.2e}  p95={np.percentile(p_annual_arr,95):.2e}")
print(f"  Climate trend: median={np.median(climate_arr):.3f}/yr")
print(f"  Hazard breakdown — median p_direct = {np.median(breakdown['p_direct']):.2e}, median p_indirect = {np.median(breakdown['p_indirect']):.2e}")

# Decision table
print("\n" + "=" * 70)
print("DECISION TABLE (rising-hazard variant)")
print("=" * 70)
header = f"{'Horizon':>8s}  {'P_loss median':>14s}  {'P_loss 5–95% band':>22s}  {'E[$ loss]':>11s}  {'P95':>11s}  {'P99':>11s}  {'CumPrem':>11s}  {'Verdict':>40s}"
print(header)
print("-" * len(header))
results_rows = []
for h in horizons:
    pl_med = np.median(pcum_rise[h])
    pl_p5  = np.percentile(pcum_rise[h], 5)
    pl_p95 = np.percentile(pcum_rise[h], 95)
    e_loss = dollar_rise[h].mean()
    p95    = np.percentile(dollar_rise[h], 95)
    p99    = np.percentile(dollar_rise[h], 99)
    cum_p  = PREM * h
    if e_loss * 3 < cum_p and p99 < cum_p * 2:
        verdict = "Self-insure (HOA vote rational)"
    elif e_loss > cum_p:
        verdict = "INSURE (E[loss] > premium)"
    elif p99 > cum_p * 3:
        verdict = "Borderline — tail dominates"
    else:
        verdict = "Self-insure (modest tail)"
    print(f"{h:>6d} yrs  {pl_med:>13.2%}  [{pl_p5:>7.2%} – {pl_p95:>7.2%}]  ${e_loss:>9,.0f}  ${p95:>9,.0f}  ${p99:>9,.0f}  ${cum_p:>9,.0f}  {verdict:>40s}")
    results_rows.append({"horizon": h, "p_loss_median": pl_med, "p_loss_p5": pl_p5, "p_loss_p95": pl_p95,
                          "E_loss": e_loss, "P95_loss": p95, "P99_loss": p99,
                          "cum_premium": cum_p, "verdict": verdict})

# Insurance-implied risk cross-check
print("\n" + "=" * 70)
print("INSURANCE-IMPLIED ANNUAL HAZARD (cross-check):")
print("=" * 70)
# At fair pricing: premium ≈ (1 + loading) × shell × p_annual
# Assume 1.5–3x risk loading + 25% overhead → effective multiplier ~2.0
for loading in [1.5, 2.0, 3.0]:
    implied_p = PREM / (loading * SHELL)
    print(f"  At {loading}x loading: implied annual P(loss) = {implied_p:.2%}  vs. modeled median P(loss/yr) = {np.median(p_annual_arr):.2%}  ({implied_p/np.median(p_annual_arr):.0f}x higher)")

# ----- Plots -----
print("\nGenerating plots...")
plt.style.use('seaborn-v0_8-whitegrid')

# Plot 1: Survival curves
years_plot = np.arange(0, 21)
mat_const = np.array([[1 - (1 - p)**t for t in years_plot] for p in p_annual_arr])
mat_rise  = np.array([[1 - math.exp(-p * (t + ct * t**2 / 2.0)) for t in years_plot] for p, ct in zip(p_annual_arr, climate_arr)])

fig, ax = plt.subplots(figsize=(11, 6))
for mat, label, color in [(mat_const, "Constant hazard", "#1f77b4"),
                           (mat_rise,  "Rising hazard (climate trend)", "#d62728")]:
    med = np.median(mat, axis=0)
    p5  = np.percentile(mat, 5, axis=0)
    p95 = np.percentile(mat, 95, axis=0)
    ax.plot(years_plot, med, label=f"{label} — median", color=color, linewidth=2)
    ax.fill_between(years_plot, p5, p95, color=color, alpha=0.15, label=f"{label} — 5–95% band")
ax.set_xlabel("Years held")
ax.set_ylabel("Cumulative P(total structural loss)")
ax.set_title("Wildfire total-loss probability vs. horizon\n759 Boulder Ct, Stateline NV — Phase 1 model (MC n=10,000)")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.legend(loc="upper left", fontsize=9)
for h in horizons:
    ax.axvline(h, linestyle=":", color="gray", alpha=0.6)
plt.tight_layout()
plt.savefig("/tmp/wildfire_plot1_survival.png", dpi=150)
print("  Saved /tmp/wildfire_plot1_survival.png")
plt.close()

# Plot 2: MC dollar distribution at 15 yrs
h_focus = 15
dollar = dollar_rise[h_focus]
fig, ax = plt.subplots(figsize=(10, 6))
counts = {"No loss ($0)": (dollar == 0).sum(),
          f"Total loss (${SHELL/1000:.0f}K)": (dollar > 0).sum()}
bars = ax.bar(list(counts.keys()), list(counts.values()), color=["#2ca02c", "#d62728"])
for i, (k, v) in enumerate(counts.items()):
    ax.text(i, v, f"{v:,}\n({v/len(dollar):.1%})", ha="center", va="bottom", fontsize=12)
ax.set_ylabel("Monte Carlo iterations (n=10,000)")
ax.set_title(f"15-yr dollar-loss outcome distribution — mostly-nothing-with-a-tail")
plt.tight_layout()
plt.savefig("/tmp/wildfire_plot2_distribution.png", dpi=150)
print("  Saved /tmp/wildfire_plot2_distribution.png")
plt.close()

# Plot 3: Premium vs loss
premiums = [PREM * h for h in horizons]
expected = [dollar_rise[h].mean()                  for h in horizons]
tail95   = [np.percentile(dollar_rise[h], 95)      for h in horizons]
tail99   = [np.percentile(dollar_rise[h], 99)      for h in horizons]
x = np.arange(len(horizons))
w = 0.18
fig, ax = plt.subplots(figsize=(11, 6))
ax.bar(x - 1.5*w, premiums, w, label="Cumulative premium (if HOA voted yes)", color="#9467bd")
ax.bar(x - 0.5*w, expected, w, label="Expected $ loss (MC mean)",              color="#2ca02c")
ax.bar(x + 0.5*w, tail95,   w, label="P95 $ loss",                              color="#ff7f0e")
ax.bar(x + 1.5*w, tail99,   w, label="P99 $ loss",                              color="#d62728")
ax.axhline(SHELL, linestyle="--", color="black", alpha=0.5, label="Shell basis ($334K)")
ax.set_xticks(x); ax.set_xticklabels([f"{h} yrs" for h in horizons])
ax.set_ylabel("USD")
ax.set_title("Premium vs. expected & tail loss — break-even visualization")
ax.legend(loc="upper left", fontsize=9)
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
plt.tight_layout()
plt.savefig("/tmp/wildfire_plot3_breakeven.png", dpi=150)
print("  Saved /tmp/wildfire_plot3_breakeven.png")
plt.close()

# Plot 4: Sensitivity tornado
def quick_run(bp_pool, flep8_pool, flep4_pool, vuln_mults_arr, vuln_probs_arr, ember_central, climate_central, n=2500):
    rng_l = np.random.default_rng(SEED + 7)
    losses = []
    for _ in range(n):
        bp = rng_l.choice(bp_pool)
        flep8 = rng_l.choice(flep8_pool) if len(flep8_pool) else 0.0
        flep4 = rng_l.choice(flep4_pool) if len(flep4_pool) else 0.0
        vh = rng_l.choice(vuln_mults_arr, p=vuln_probs_arr)
        vl = vh * 0.4
        ember = rng_l.lognormal(mean=math.log(ember_central), sigma=0.5) * vh
        pd_ = bp * (flep8 * vh + max(0.0, flep4 - flep8) * vl)
        p_ann = pd_ + ember
        ct = max(0.0, rng_l.normal(climate_central, 0.015))
        pc = 1 - math.exp(-p_ann * (15 + ct * 15**2 / 2.0))
        losses.append(SHELL if rng_l.random() < pc else 0)
    return float(np.mean(losses))

bp_low  = [float(np.percentile(bp_samples, 5))]
bp_high = [float(np.percentile(bp_samples, 95))]

base_E = quick_run(bp_samples, flep8_burnable, flep4_burnable, vuln_mults, vuln_probs, 2.0e-4, 0.03)

scenarios = [
    ("BP at parcel-cell P5 (≈0.00041)",      quick_run(bp_low,  flep8_burnable, flep4_burnable, vuln_mults, vuln_probs, 2.0e-4, 0.03)),
    ("BP at neighborhood P95 (≈0.001)",      quick_run(bp_high, flep8_burnable, flep4_burnable, vuln_mults, vuln_probs, 2.0e-4, 0.03)),
    ("FLEP8 = 0 (no burnable cells)",        quick_run(bp_samples, [0.0],            [0.0],            vuln_mults, vuln_probs, 2.0e-4, 0.03)),
    ("FLEP8 = max (0.10)",                   quick_run(bp_samples, [0.10],           flep4_burnable,    vuln_mults, vuln_probs, 2.0e-4, 0.03)),
    ("Vulnerability = best (0.30)",          quick_run(bp_samples, flep8_burnable, flep4_burnable, np.array([0.30]), np.array([1.0]), 2.0e-4, 0.03)),
    ("Vulnerability = worst (0.90)",         quick_run(bp_samples, flep8_burnable, flep4_burnable, np.array([0.90]), np.array([1.0]), 2.0e-4, 0.03)),
    ("Ember baseline low (5e-5)",            quick_run(bp_samples, flep8_burnable, flep4_burnable, vuln_mults, vuln_probs, 5.0e-5, 0.03)),
    ("Ember baseline high (1e-3)",           quick_run(bp_samples, flep8_burnable, flep4_burnable, vuln_mults, vuln_probs, 1.0e-3, 0.03)),
    ("Climate trend = 0 %/yr",               quick_run(bp_samples, flep8_burnable, flep4_burnable, vuln_mults, vuln_probs, 2.0e-4, 0.00)),
    ("Climate trend = 6 %/yr",               quick_run(bp_samples, flep8_burnable, flep4_burnable, vuln_mults, vuln_probs, 2.0e-4, 0.06)),
]
deltas = [(label, val - base_E) for label, val in scenarios]
deltas.sort(key=lambda kv: abs(kv[1]), reverse=True)

fig, ax = plt.subplots(figsize=(11, 6))
labels = [k for k, _ in deltas]
vals   = [v for _, v in deltas]
colors = ["#d62728" if v > 0 else "#1f77b4" for v in vals]
y = np.arange(len(labels))
ax.barh(y, vals, color=colors)
ax.set_yticks(y); ax.set_yticklabels(labels)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Δ 15-yr expected $ loss vs. base case")
ax.set_title(f"Sensitivity tornado — base E[loss @15yrs] = ${base_E:,.0f}")
ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x/1000:+.0f}K"))
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("/tmp/wildfire_plot4_tornado.png", dpi=150)
print("  Saved /tmp/wildfire_plot4_tornado.png")
plt.close()

print(f"\nBase 15-yr expected loss: ${base_E:,.0f}")
print("Sensitivity (largest movers):")
for k, v in deltas[:6]:
    print(f"  {k:42s}  Δ = ${v:+,.0f}")

# Save MC results to JSON for downstream
with open("/tmp/wildfire_mc_results.json", "w") as f:
    json.dump({
        "config": CONFIG,
        "fsim_summary": {
            "bp": {"min": min(bp_samples), "med": float(np.median(bp_samples)), "max": max(bp_samples)},
            "flep8_burnable": {"n": len(flep8_burnable),
                                "max": max(flep8_burnable) if flep8_burnable else 0.0},
        },
        "decision_table": results_rows,
        "p_annual_summary": {
            "median": float(np.median(p_annual_arr)),
            "mean":   float(p_annual_arr.mean()),
            "p5":     float(np.percentile(p_annual_arr, 5)),
            "p95":    float(np.percentile(p_annual_arr, 95)),
        },
        "base_E_loss_15yr":   base_E,
        "sensitivity":        [{"label": k, "delta_E_loss": v} for k, v in deltas],
        "run_timestamp_utc":  datetime.now(timezone.utc).isoformat(),
    }, f, indent=2)

print("\nResults JSON saved to /tmp/wildfire_mc_results.json")
print("Plots saved to /tmp/wildfire_plot{1..4}_*.png")
