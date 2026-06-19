#!/usr/bin/env python3
"""Phase 3 MC — non-linear, climate-driven hazard trajectory.

Replaces Phase 2's linear h(t) = h0 × (1 + 0.03t) with h(t) = h0 × f_climate(t)
where f_climate is derived from Cal-Adapt LOCA-downscaled CMIP5 projections at the parcel,
across 4 GCMs × 2 RCPs (8 ensemble members), through 2041.

Wildfire-driven hazard scales with summer/annual Tmax anomaly per published Sierra elasticities:
  - Westerling 2018, Goss et al. 2020, Abatzoglou & Williams 2016
  - Central β = 0.75 per °C of annual Tmax anomaly (log-linear relationship)
  - Range 0.5–1.0 per °C accounts for uncertainty in the climate-fire response
Interior ignition is NOT climate-modulated.
"""
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from datetime import datetime, timezone

START_YEAR = 2026
HORIZONS = [5, 10, 15]
HORIZON_YEARS = {h: list(range(START_YEAR, START_YEAR + h)) for h in HORIZONS}
SHELL = 334_000
PREM  = 24_000
SEED  = 42
N     = 10_000

# --- Load Phase 2 inputs (vulnerability + ember + interior) ---
with open("/tmp/dins_phase2_inputs.json") as f: p2_inp = json.load(f)
with open("/tmp/fsim_grid_parcel.json") as f: fsim_d = json.load(f)
with open("/tmp/caladapt_tasmax.json") as f: clim = json.load(f)
with open("/tmp/wildfire_p2_results.json") as f: p2_results = json.load(f)
with open("/tmp/wildfire_mc_results.json") as f: p1_results = json.load(f)

# FSim aggregation
scales = {"BP": 1/100_000, "FLEP4": 1/1_000, "FLEP8": 1/1_000}
grid_scaled = {
    layer: [[(v * scales[layer]) if v is not None else None for v in row] for row in mat]
    for layer, mat in fsim_d["raw_values"].items()
}
bp_samples     = [v for row in grid_scaled["BP"]    for v in row if v is not None]
flep8_burnable = [v for row in grid_scaled["FLEP8"] for v in row if v not in (None, 0.0)]
flep4_burnable = [v for row in grid_scaled["FLEP4"] for v in row if v not in (None, 0.0)]

# --- Build climate-driven hazard trajectories ---
print("=" * 75)
print("Climate-driven hazard trajectories per Cal-Adapt ensemble member")
print("=" * 75)
ensemble = {}
for name, series in clim["series"].items():
    yrs  = series["years"]
    tmax = series["tasmax_c"]
    # Baseline = mean of 2006-2025 (the 20-yr period before parcel hold begins)
    baseline_mask = [(2006 <= y <= 2025) for y in yrs]
    base = np.mean([t for t, m in zip(tmax, baseline_mask) if m])
    # Project trajectory for 2026-2041 (or as far as available)
    yr_anomaly = {}
    for y, t in zip(yrs, tmax):
        if 2026 <= y <= 2050:
            yr_anomaly[y] = t - base
    ensemble[name] = {"baseline_c": base, "anomaly_by_year": yr_anomaly}
    # Print 5/10/15 year horizon averages
    a5  = np.mean([yr_anomaly.get(y, np.nan) for y in range(2026, 2031)])
    a10 = np.mean([yr_anomaly.get(y, np.nan) for y in range(2026, 2036)])
    a15 = np.mean([yr_anomaly.get(y, np.nan) for y in range(2026, 2041)])
    print(f"  {name:30s}  base={base:.2f}°C  Δ5yr={a5:+.2f}  Δ10yr={a10:+.2f}  Δ15yr={a15:+.2f}")

# Beta elasticity prior (Westerling 2018 central, range from literature)
BETA_CENTRAL = 0.75   # log-burn-area per °C of Tmax anomaly
BETA_LOW     = 0.50
BETA_HIGH    = 1.00

# Per year hazard multiplier given ensemble member m and beta:
def f_climate(year, ensemble_member, beta):
    a = ensemble_member["anomaly_by_year"].get(year, 0.0)
    return math.exp(beta * a)

# Quick visualization of central trajectory
print("\nCentral-β median ensemble climate hazard multiplier f_climate(y):")
print(f"  {'year':>6s}  {'min':>5s}  {'med':>5s}  {'max':>5s}")
ensemble_names = list(ensemble.keys())
for y in [2026, 2030, 2035, 2040]:
    mults = [f_climate(y, ensemble[m], BETA_CENTRAL) for m in ensemble_names]
    print(f"  {y:>6d}  {min(mults):.2f}  {np.median(mults):.2f}  {max(mults):.2f}")

# --- Phase 3 hazard sampler ---
# Same as Phase 2 BUT replace the constant climate_trend with per-year multipliers
P2_SCENARIOS = p2_inp["phase2_scenarios"]
vuln_mults = np.array([s["mult"] for s in P2_SCENARIOS.values()])
vuln_probs = np.array([s["prob"] for s in P2_SCENARIOS.values()])
vuln_probs = vuln_probs / vuln_probs.sum()

EMBER_BASELINE_MED = 2.0e-4
EMBER_BASELINE_SIGMA_LOG = 0.6
PER_UNIT_FIRE_RATE_PER_YR_MED   = 3.0e-3
PER_UNIT_FIRE_RATE_PER_YR_SIGMA = 0.3
P_SHELL_LOSS_GIVEN_UNIT_FIRE    = 0.07  # mean of Beta(2, 26)

def sample_interior(rng):
    n_units = rng.integers(4, 17)
    per_unit_fire_rate = rng.lognormal(math.log(PER_UNIT_FIRE_RATE_PER_YR_MED),
                                       PER_UNIT_FIRE_RATE_PER_YR_SIGMA)
    p_shell_loss_given_fire = rng.beta(2.0, 26.0)
    p_unit_total = per_unit_fire_rate * p_shell_loss_given_fire
    p_interior_annual = 1.0 - (1.0 - p_unit_total) ** n_units
    return p_interior_annual

def sample_hazard_p3(rng):
    bp     = rng.choice(bp_samples)
    flep8  = rng.choice(flep8_burnable)
    flep4  = rng.choice(flep4_burnable)
    vuln_h = rng.choice(vuln_mults, p=vuln_probs)
    vuln_l = vuln_h * 0.4

    ember = rng.lognormal(math.log(EMBER_BASELINE_MED), EMBER_BASELINE_SIGMA_LOG) * vuln_h
    interior = sample_interior(rng)

    p_destroy_given_fire = flep8 * vuln_h + max(0.0, flep4 - flep8) * vuln_l
    h0_wildfire = bp * p_destroy_given_fire + ember
    h0_interior = interior

    # Climate ensemble draw
    member_name = rng.choice(ensemble_names)
    member = ensemble[member_name]
    # Beta: triangular over [0.5, 0.75, 1.0]
    beta = rng.triangular(BETA_LOW, BETA_CENTRAL, BETA_HIGH)

    return {
        "h0_wildfire": h0_wildfire,
        "h0_interior": h0_interior,
        "member":      member,
        "beta":        beta,
        "member_name": member_name,
    }

def cumulative_loss_p3(s, n_years):
    """Per-year hazard integration with climate-driven wildfire multiplier."""
    H = 0.0
    for k in range(n_years):
        y = START_YEAR + k
        f_c = f_climate(y, s["member"], s["beta"])
        H += s["h0_wildfire"] * f_c + s["h0_interior"]
    return 1.0 - math.exp(-H)

# --- Run MC ---
rng = np.random.default_rng(SEED)
p_annual_arr = np.zeros(N)
member_counts = {}
pcum     = {h: np.zeros(N) for h in HORIZONS}
dollar   = {h: np.zeros(N, dtype=int) for h in HORIZONS}
decomp_climate = []  # store climate multipliers used per iter
for i in range(N):
    s = sample_hazard_p3(rng)
    member_counts[s["member_name"]] = member_counts.get(s["member_name"], 0) + 1
    decomp_climate.append({
        "h0_wildfire": s["h0_wildfire"],
        "h0_interior": s["h0_interior"],
        "beta": s["beta"],
        "member": s["member_name"],
    })
    # Annual hazard at year 0 (for reference)
    p_annual_arr[i] = s["h0_wildfire"] + s["h0_interior"]
    for h in HORIZONS:
        pc = cumulative_loss_p3(s, h)
        pcum[h][i] = pc
        dollar[h][i] = SHELL if rng.random() < pc else 0

print(f"\nPhase 3 MC: N={N}")
print(f"  Ensemble members sampled: {member_counts}")
print(f"  P_annual(t=0):  median={np.median(p_annual_arr):.2e}  mean={p_annual_arr.mean():.2e}")

# --- Decision table — Phase 3 ---
p3_decision = []
for h in HORIZONS:
    pl = pcum[h]; d = dollar[h]
    e_loss = float(d.mean()); p95 = float(np.percentile(d,95)); p99 = float(np.percentile(d,99))
    cum_p = PREM * h
    if e_loss * 3 < cum_p and p99 < cum_p:
        verdict = "Self-insure"
    elif e_loss > cum_p:
        verdict = "INSURE (E[loss] > premium)"
    elif p99 >= SHELL and p99 > cum_p:
        verdict = "Borderline — tail hits"
    else:
        verdict = "Self-insure (modest tail)"
    p3_decision.append({
        "horizon": h,
        "p_loss_median": float(np.median(pl)), "p_loss_p5": float(np.percentile(pl,5)), "p_loss_p95": float(np.percentile(pl,95)),
        "E_loss": e_loss, "P95_loss": p95, "P99_loss": p99,
        "cum_premium": cum_p, "verdict": verdict,
    })

# --- Side-by-side P1/P2/P3 ---
print("\n" + "=" * 110)
print("DECISION TABLE — PHASE 1 vs PHASE 2 vs PHASE 3")
print("=" * 110)
print(f"{'h':>3s}  {'Phase':>5s}  {'P_loss_med':>11s}  {'5–95% band':>17s}  {'E[$ loss]':>11s}  {'P95':>10s}  {'P99':>10s}  {'CumPrem':>10s}  Verdict")
print("-" * 130)
p1_dt = {row["horizon"]: row for row in p1_results["decision_table"]}
p2_dt = {row["horizon"]: row for row in p2_results["decision_table"]}
for h in HORIZONS:
    r1 = p1_dt[h]
    r2 = p2_dt[h]
    r3 = next(r for r in p3_decision if r["horizon"] == h)
    for tag, r in [("P1", r1), ("P2", r2), ("P3", r3)]:
        print(f"{h:>2d}y  {tag:>5s}  {r['p_loss_median']:>11.2%}  [{r['p_loss_p5']:>5.2%}–{r['p_loss_p95']:>5.2%}]  ${r['E_loss']:>9,.0f}  ${r['P95_loss']:>8,.0f}  ${r['P99_loss']:>8,.0f}  ${r['cum_premium']:>8,.0f}  {r['verdict']}")
    print()

# Insurance-implied cross-check (Phase 3)
print("=" * 110)
print("INSURANCE-IMPLIED RISK CROSS-CHECK (Phase 3)")
print("=" * 110)
p_ann_med_p3 = float(np.median(p_annual_arr))
print(f"  Phase 3 median annual P(loss):  {p_ann_med_p3:.4%}  (at t=0; rises over time)")
# Also report time-averaged hazard
avg_h = np.zeros(N)
for i in range(N):
    s_member = ensemble[decomp_climate[i]["member"]]
    avg_f = np.mean([f_climate(y, s_member, decomp_climate[i]["beta"]) for y in range(2026, 2041)])
    avg_h[i] = decomp_climate[i]["h0_wildfire"] * avg_f + decomp_climate[i]["h0_interior"]
print(f"  Phase 3 median *time-avg* annual P(loss) over 15yr: {np.median(avg_h):.4%}")
for loading in [1.5, 2.0, 3.0]:
    implied = PREM / (loading * SHELL)
    ratio = implied / np.median(avg_h)
    print(f"  At {loading}x loading, insurer implies P(loss/yr) = {implied:.2%}, ratio to P3 time-avg = {ratio:.0f}x")

# ===== Plots =====
plt.style.use('seaborn-v0_8-whitegrid')

# Plot 1: Climate trajectory fan chart
fig, ax = plt.subplots(figsize=(11, 6))
years_plot = list(range(2026, 2041))
for name, m in ensemble.items():
    series_mult = [f_climate(y, m, BETA_CENTRAL) for y in years_plot]
    color = "#d62728" if "rcp85" in name else "#1f77b4"
    label = name if name == ensemble_names[0] else None
    ax.plot(years_plot, series_mult, color=color, alpha=0.55, linewidth=1.4, label=name)
ax.axhline(1.0, color="gray", linestyle="--", alpha=0.6, label="Baseline (no warming)")
ax.set_xlabel("Year")
ax.set_ylabel("Climate-driven hazard multiplier  f_climate(y)")
ax.set_title("Sierra fire-hazard trajectory by CMIP5 ensemble member (β=0.75/°C)\n759 Boulder Ct (Cal-Adapt LOCA-downscaled)")
ax.legend(loc="upper left", fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig("/tmp/wildfire_p3_plot1_trajectory.png", dpi=150)
print("\nSaved /tmp/wildfire_p3_plot1_trajectory.png")
plt.close()

# Plot 2: Phase 1/2/3 survival curves
fig, ax = plt.subplots(figsize=(11, 6))
years_curve = np.arange(0, 21)

# Phase 1 from saved
def cum_simple(p_ann, ct=0.03, years=years_curve):
    return [1 - math.exp(-p_ann * (t + ct * t**2 / 2.0)) for t in years]
ax.plot(years_curve, cum_simple(p1_results["p_annual_summary"]["median"]),
        label="Phase 1 median", color="#1f77b4", linestyle="--", linewidth=2)

# Phase 2 from saved
ax.plot(years_curve, cum_simple(p2_results["p_annual_summary"]["median"]),
        label="Phase 2 median", color="#2ca02c", linestyle="-.", linewidth=2)

# Phase 3 — recompute curves across all MC iters
def p3_curve(s, years):
    out = np.zeros(len(years))
    H = 0.0
    for j, t in enumerate(years):
        if t == 0:
            out[j] = 0.0; continue
        # incremental: year (t-1)+START to t+START
        y = START_YEAR + int(t) - 1
        f_c = f_climate(y, s["member"], s["beta"])
        H = s["h0_wildfire"] * f_c + s["h0_interior"]  # WRONG — needs cumulative
        # actually: cumulative hazard
    # Re-do correctly
    out = np.zeros(len(years))
    H_cum = 0.0
    yrs_seen = 0
    for j, t in enumerate(years):
        # bring H_cum up to year t
        target = int(t)
        while yrs_seen < target:
            y = START_YEAR + yrs_seen
            f_c = f_climate(y, s["member"], s["beta"])
            H_cum += s["h0_wildfire"] * f_c + s["h0_interior"]
            yrs_seen += 1
        out[j] = 1 - math.exp(-H_cum)
    return out

# Sample 1000 iters for curves (full 10K is overkill for plot)
rng2 = np.random.default_rng(SEED + 99)
idx_sample = rng2.choice(N, 1000, replace=False)
p3_mat = np.zeros((1000, len(years_curve)))
for k, i in enumerate(idx_sample):
    s = decomp_climate[i]
    s_full = {"h0_wildfire": s["h0_wildfire"], "h0_interior": s["h0_interior"],
              "member": ensemble[s["member"]], "beta": s["beta"]}
    p3_mat[k] = p3_curve(s_full, years_curve)
med = np.median(p3_mat, axis=0); p5 = np.percentile(p3_mat,5,axis=0); p95 = np.percentile(p3_mat,95,axis=0)
ax.plot(years_curve, med, label="Phase 3 median (climate-driven)", color="#d62728", linewidth=2)
ax.fill_between(years_curve, p5, p95, color="#d62728", alpha=0.20, label="Phase 3 5–95%")

ax.set_xlabel("Years held (2026 → 2041)")
ax.set_ylabel("Cumulative P(total structural loss)")
ax.set_title("Phase 1 → Phase 2 → Phase 3 — total-loss probability\n759 Boulder Ct (climate-driven hazard from Cal-Adapt LOCA ensemble)")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.legend(loc="upper left", fontsize=9)
for h in HORIZONS: ax.axvline(h, linestyle=":", color="gray", alpha=0.5)
plt.tight_layout()
plt.savefig("/tmp/wildfire_p3_plot2_survival_compare.png", dpi=150)
print("Saved /tmp/wildfire_p3_plot2_survival_compare.png")
plt.close()

# Plot 3: P3 dollar exposure
fig, ax = plt.subplots(figsize=(11, 6))
horizons_x = np.arange(len(HORIZONS))
w = 0.13
ax.bar(horizons_x - 2.5*w, [PREM*h for h in HORIZONS], w, label="Cum. premium", color="#9467bd")
for k, (label, color, src) in enumerate([
    ("E P1", "#9edae5", p1_dt), ("P99 P1", "#1f77b4", p1_dt),
    ("E P2", "#98df8a", p2_dt), ("P99 P2", "#2ca02c", p2_dt),
    ("E P3", "#ff9896", "p3"),  ("P99 P3", "#d62728", "p3")]):
    if src == "p3":
        vals = [next(r for r in p3_decision if r["horizon"]==h)["E_loss" if k%2==0 else "P99_loss"] for h in HORIZONS]
    else:
        vals = [src[h]["E_loss" if k%2==0 else "P99_loss"] for h in HORIZONS]
    offset = (k - 2.5)*w + (0 if k < 2 else (3*w if k < 4 else 6*w))
    # Hmm this is getting messy. Simpler: 3 group sets
ax.clear()
groups = ["E[loss]", "P99 loss"]
group_colors_p1 = ["#aec7e8", "#1f77b4"]
group_colors_p2 = ["#98df8a", "#2ca02c"]
group_colors_p3 = ["#ff9896", "#d62728"]
n_horiz = len(HORIZONS)
w = 0.10
x = np.arange(n_horiz)
ax.bar(x - 3.5*w, [PREM*h for h in HORIZONS], w, label="Cum. premium", color="#9467bd")
for i, (label, color) in enumerate(zip(["E P1","P99 P1"], group_colors_p1)):
    vals = [p1_dt[h]["E_loss" if i==0 else "P99_loss"] for h in HORIZONS]
    ax.bar(x + (-2.5 + i)*w, vals, w, label=label, color=color)
for i, (label, color) in enumerate(zip(["E P2","P99 P2"], group_colors_p2)):
    vals = [p2_dt[h]["E_loss" if i==0 else "P99_loss"] for h in HORIZONS]
    ax.bar(x + (-0.5 + i)*w, vals, w, label=label, color=color)
for i, (label, color) in enumerate(zip(["E P3","P99 P3"], group_colors_p3)):
    vals = [next(r for r in p3_decision if r["horizon"]==h)["E_loss" if i==0 else "P99_loss"] for h in HORIZONS]
    ax.bar(x + (1.5 + i)*w, vals, w, label=label, color=color)
ax.axhline(SHELL, linestyle="--", color="black", alpha=0.5, label="Shell ($334K)")
ax.set_xticks(x); ax.set_xticklabels([f"{h} yr" for h in HORIZONS])
ax.set_ylabel("USD")
ax.set_title("Phase 1 vs Phase 2 vs Phase 3 — $ exposure (rising-hazard / climate-driven)")
ax.legend(loc="upper left", fontsize=8, ncol=4)
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
plt.tight_layout()
plt.savefig("/tmp/wildfire_p3_plot3_compare.png", dpi=150)
print("Saved /tmp/wildfire_p3_plot3_compare.png")
plt.close()

# Save Phase 3 results
out = {
    "phase": 3,
    "config": {
        "climate_ensemble_members": list(ensemble.keys()),
        "beta_per_C": {"central": BETA_CENTRAL, "low": BETA_LOW, "high": BETA_HIGH},
        "start_year": START_YEAR,
        "horizons":   HORIZONS,
    },
    "decision_table":   p3_decision,
    "climate_summary": {
        name: {
            "baseline_c":    m["baseline_c"],
            "delta_15yr_c":  np.mean([m["anomaly_by_year"].get(y, np.nan) for y in range(2026, 2041)]),
            "fclim_15yr":    np.exp(BETA_CENTRAL * np.mean([m["anomaly_by_year"].get(y, np.nan) for y in range(2026, 2041)])),
        } for name, m in ensemble.items()
    },
    "p_annual_summary_t0": {
        "median": float(np.median(p_annual_arr)),
        "p5": float(np.percentile(p_annual_arr,5)),
        "p95": float(np.percentile(p_annual_arr,95)),
    },
    "p_annual_summary_timeavg_15yr": {
        "median": float(np.median(avg_h)),
        "p5": float(np.percentile(avg_h,5)),
        "p95": float(np.percentile(avg_h,95)),
    },
    "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
}
with open("/tmp/wildfire_p3_results.json","w") as f:
    json.dump(out, f, indent=2, default=str)
print(f"Saved /tmp/wildfire_p3_results.json")
