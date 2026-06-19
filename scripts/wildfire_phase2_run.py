#!/usr/bin/env python3
"""Phase 2 MC — fitted DINS vulnerability multiplier + interior-ignition term.

Inputs:
  Phase 2 vulnerability scenarios (logreg-fitted, 4-scenario PMF with bootstrap CIs)
  Phase 2 ember term — DINS-aggregate sanity-checked, keep 2e-4 central
  Phase 2 interior-ignition baseline — NFPA US residential fire statistics

Outputs:
  Decision table — side-by-side Phase 1 vs Phase 2
  Updated plots: survival curves, dollar distribution, break-even, sensitivity
"""
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from datetime import datetime, timezone

# Load Phase 2 inputs from DINS training
with open("/tmp/dins_phase2_inputs.json") as f:
    p2_inp = json.load(f)
with open("/tmp/fsim_grid_parcel.json") as f:
    fsim_d = json.load(f)
with open("/tmp/wildfire_mc_results.json") as f:
    p1_results = json.load(f)

SHELL = 334_000
PREM  = 24_000
SEED  = 42
N     = 10_000
HORIZONS = [5, 10, 15]

# FSim aggregation (same as Phase 1)
scales = {"BP": 1/100_000, "FLEP4": 1/1_000, "FLEP8": 1/1_000}
grid_scaled = {
    layer: [[(v * scales[layer]) if v is not None else None for v in row] for row in mat]
    for layer, mat in fsim_d["raw_values"].items()
}
bp_samples     = [v for row in grid_scaled["BP"]    for v in row if v is not None]
flep8_burnable = [v for row in grid_scaled["FLEP8"] for v in row if v not in (None, 0.0)]
flep4_burnable = [v for row in grid_scaled["FLEP4"] for v in row if v not in (None, 0.0)]

# --- Phase 2 vulnerability PMF (from DINS) ---
P2_SCENARIOS = p2_inp["phase2_scenarios"]
vuln_mults = np.array([s["mult"] for s in P2_SCENARIOS.values()])
vuln_probs = np.array([s["prob"] for s in P2_SCENARIOS.values()])
vuln_probs = vuln_probs / vuln_probs.sum()
E_vuln_p2 = (vuln_mults * vuln_probs).sum()
print(f"Phase 2 E[vuln_mult] = {E_vuln_p2:.3f}  (Phase 1 was 0.568)")

# --- Phase 2 ember baseline ---
# DINS sanity check: of 2007 multi-res in fire-perimeter inspections, 27.4% destroyed
# At an annual structure exposure rate of ~0.001 (US WUI average), implied annual P_loss ≈ 0.001 × 0.27 = 2.7e-4
# Our parcel BP × vuln ≈ 0.0005 × 0.21 ≈ 1e-4 — so ember should account for the gap to insurance pricing
# Keep 2e-4 central but widen the upper-end uncertainty
EMBER_BASELINE_MED = 2.0e-4
EMBER_BASELINE_SIGMA_LOG = 0.6  # slightly wider than Phase 1 (0.5)

# --- Phase 2 INTERIOR-IGNITION term (new in Phase 2) ---
# NFPA residential structure-fire statistics for the US (averaging 2018-2022):
#   Apartments / multi-family: ~3.0 fires per 1000 occupied units per year
#   Of those, ~5–10% result in damage extensive enough to total the structural shell
#   (most are confined to single unit; flame spreads beyond unit of origin ~12% of the time;
#    flame damages structural shell to "destroyed" level: ~5–10% conditional)
#
# For an N-unit building, the annual P(any unit fire that totals the building shell):
#   = 1 - (1 - per_unit_fire_rate × P_shell_loss_given_fire) ** N
#
# For a 1980 multi-unit condo, plausible N is 4-20 units (we don't know exactly)
# Use N=8 as central, draw uniformly from [4, 16] in MC for uncertainty
PER_UNIT_FIRE_RATE_PER_YR_MED   = 3.0e-3
PER_UNIT_FIRE_RATE_PER_YR_SIGMA = 0.3  # log-normal sigma
P_SHELL_LOSS_GIVEN_UNIT_FIRE    = 0.07  # central: 7% of unit fires reach shell-loss

def sample_interior(rng):
    n_units = rng.integers(4, 17)
    per_unit_fire_rate = rng.lognormal(math.log(PER_UNIT_FIRE_RATE_PER_YR_MED),
                                       PER_UNIT_FIRE_RATE_PER_YR_SIGMA)
    p_shell_loss_given_fire = rng.beta(2.0, 26.0)  # mean ~0.07
    # P(any unit fires AND that fire totals the building) for the year
    p_unit_total = per_unit_fire_rate * p_shell_loss_given_fire
    p_interior_annual = 1.0 - (1.0 - p_unit_total) ** n_units
    return p_interior_annual, n_units

# --- Updated hazard sampler (Phase 2) ---
def sample_hazard_p2(rng):
    bp     = rng.choice(bp_samples)
    flep8  = rng.choice(flep8_burnable)
    flep4  = rng.choice(flep4_burnable)
    vuln_h = rng.choice(vuln_mults, p=vuln_probs)
    vuln_l = vuln_h * 0.4

    ember = rng.lognormal(math.log(EMBER_BASELINE_MED), EMBER_BASELINE_SIGMA_LOG) * vuln_h
    interior, n_units = sample_interior(rng)

    p_destroy_given_fire = flep8 * vuln_h + max(0.0, flep4 - flep8) * vuln_l
    p_wildfire_direct    = bp * p_destroy_given_fire
    p_wildfire_indirect  = ember
    p_interior           = interior

    p_annual = p_wildfire_direct + p_wildfire_indirect + p_interior

    climate_trend = max(0.0, rng.normal(0.030, 0.015))
    return {
        "p_annual": p_annual, "climate_trend": climate_trend,
        "bp": bp, "flep8": flep8, "vuln_h": vuln_h,
        "p_wildfire_direct": p_wildfire_direct,
        "p_wildfire_indirect": p_wildfire_indirect,
        "p_interior": p_interior,
        "n_units": n_units,
    }

def p_cum_rising(p, t, ct):
    H = p * (t + ct * t**2 / 2.0)
    return 1.0 - math.exp(-H)

# --- Monte Carlo ---
rng = np.random.default_rng(SEED)
p_annual_arr   = np.zeros(N)
climate_arr    = np.zeros(N)
pcum_rise      = {h: np.zeros(N) for h in HORIZONS}
dollar         = {h: np.zeros(N, dtype=int) for h in HORIZONS}
decomp         = {k: np.zeros(N) for k in ("p_wildfire_direct","p_wildfire_indirect","p_interior")}

for i in range(N):
    s = sample_hazard_p2(rng)
    p_annual_arr[i] = s["p_annual"]
    climate_arr[i]  = s["climate_trend"]
    for k in decomp: decomp[k][i] = s[k]
    for h in HORIZONS:
        pc = p_cum_rising(s["p_annual"], h, s["climate_trend"])
        pcum_rise[h][i] = pc
        dollar[h][i] = SHELL if rng.random() < pc else 0

print(f"\nPhase 2 MC: N={N}")
print(f"  P_annual:   median={np.median(p_annual_arr):.2e}  mean={p_annual_arr.mean():.2e}  p95={np.percentile(p_annual_arr,95):.2e}")
print(f"  Wildfire direct:   median={np.median(decomp['p_wildfire_direct']):.2e}")
print(f"  Wildfire indirect: median={np.median(decomp['p_wildfire_indirect']):.2e}")
print(f"  Interior ignition: median={np.median(decomp['p_interior']):.2e}  ← NEW in Phase 2")

# === Decision tables: Phase 1 vs Phase 2 ===
print("\n" + "=" * 90)
print("DECISION TABLE — PHASE 1 vs PHASE 2 (rising-hazard variant)")
print("=" * 90)
print(f"{'horizon':>8s}  {'Phase':>6s}  {'P_loss med':>11s}  {'5–95% band':>17s}  {'E[$ loss]':>11s}  {'P95':>10s}  {'P99':>10s}  {'CumPrem':>10s}  {'Verdict':<40s}")
print("-" * 130)
p1_dt = {row["horizon"]: row for row in p1_results["decision_table"]}
p2_decision = []
for h in HORIZONS:
    # Phase 1 row
    r1 = p1_dt[h]
    print(f"{h:>6d} y  {'P1':>6s}  {r1['p_loss_median']:>11.2%}  [{r1['p_loss_p5']:>5.2%}–{r1['p_loss_p95']:>5.2%}]  ${r1['E_loss']:>9,.0f}  ${r1['P95_loss']:>8,.0f}  ${r1['P99_loss']:>8,.0f}  ${r1['cum_premium']:>8,.0f}  {r1['verdict']:<40s}")
    # Phase 2 row
    pl = pcum_rise[h]
    d  = dollar[h]
    pl_med = float(np.median(pl)); pl_p5 = float(np.percentile(pl,5)); pl_p95 = float(np.percentile(pl,95))
    e_loss = float(d.mean()); p95 = float(np.percentile(d,95)); p99 = float(np.percentile(d,99))
    cum_p  = PREM * h
    if e_loss * 3 < cum_p and p99 < cum_p:
        verdict = "Self-insure"
    elif e_loss > cum_p:
        verdict = "INSURE (E[loss] > premium)"
    elif p99 >= SHELL and p99 > cum_p:
        verdict = "Borderline — tail of $334K hits"
    else:
        verdict = "Self-insure (modest tail)"
    p2_decision.append({"horizon": h, "p_loss_median": pl_med, "p_loss_p5": pl_p5, "p_loss_p95": pl_p95,
                         "E_loss": e_loss, "P95_loss": p95, "P99_loss": p99,
                         "cum_premium": cum_p, "verdict": verdict})
    print(f"{h:>6d} y  {'P2':>6s}  {pl_med:>11.2%}  [{pl_p5:>5.2%}–{pl_p95:>5.2%}]  ${e_loss:>9,.0f}  ${p95:>8,.0f}  ${p99:>8,.0f}  ${cum_p:>8,.0f}  {verdict:<40s}")
    print()

# === Insurance-implied cross-check (Phase 2) ===
print("=" * 90)
print("INSURANCE-IMPLIED RISK CROSS-CHECK (Phase 2)")
print("=" * 90)
print(f"  Phase 2 median annual P(loss):  {np.median(p_annual_arr):.4%}")
for loading in [1.5, 2.0, 3.0]:
    implied = PREM / (loading * SHELL)
    ratio = implied / np.median(p_annual_arr)
    print(f"  At {loading}x loading, insurer implies P(loss/yr) = {implied:.2%}, ratio to Phase 2 median = {ratio:.0f}x")

# === Plots ===
plt.style.use('seaborn-v0_8-whitegrid')

# Plot 1: side-by-side survival curves
years_plot = np.arange(0, 21)
def cum_curve_p2(p_arr, ct_arr):
    out = np.zeros((len(p_arr), len(years_plot)))
    for i, (p, ct) in enumerate(zip(p_arr, ct_arr)):
        for j, t in enumerate(years_plot):
            out[i,j] = p_cum_rising(p, t, ct)
    return out

# Re-create Phase 1 curves from saved data
p1_p_ann_med = p1_results["p_annual_summary"]["median"]
p1_p_ann_p5  = p1_results["p_annual_summary"]["p5"]
p1_p_ann_p95 = p1_results["p_annual_summary"]["p95"]
def cum_curve_simple(p_ann, ct=0.03):
    return [1 - math.exp(-p_ann * (t + ct * t**2 / 2.0)) for t in years_plot]

mat_p2 = cum_curve_p2(p_annual_arr, climate_arr)

fig, ax = plt.subplots(figsize=(11, 6))
# Phase 1 single line for context
ax.plot(years_plot, cum_curve_simple(p1_p_ann_med), label=f"Phase 1 median (no interior term)", color="#1f77b4", linestyle="--", linewidth=2)
ax.fill_between(years_plot, cum_curve_simple(p1_p_ann_p5), cum_curve_simple(p1_p_ann_p95),
                color="#1f77b4", alpha=0.10, label="Phase 1 5–95%")
# Phase 2
med = np.median(mat_p2, axis=0); p5 = np.percentile(mat_p2,5,axis=0); p95 = np.percentile(mat_p2,95,axis=0)
ax.plot(years_plot, med, label="Phase 2 median (DINS-fitted + interior)", color="#d62728", linewidth=2)
ax.fill_between(years_plot, p5, p95, color="#d62728", alpha=0.20, label="Phase 2 5–95%")
ax.set_xlabel("Years held")
ax.set_ylabel("Cumulative P(total structural loss)")
ax.set_title("Phase 1 vs Phase 2 — total-loss probability vs horizon\n759 Boulder Ct, Stateline NV")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.legend(loc="upper left", fontsize=9)
for h in HORIZONS:
    ax.axvline(h, linestyle=":", color="gray", alpha=0.5)
plt.tight_layout()
plt.savefig("/tmp/wildfire_p2_plot1_survival.png", dpi=150)
print("\nSaved /tmp/wildfire_p2_plot1_survival.png")
plt.close()

# Plot 2: Hazard decomposition (where does the risk actually come from?)
fig, ax = plt.subplots(figsize=(10, 6))
labels = ["Wildfire direct\n(BP × FLEP × vuln)", "Wildfire indirect\n(ember/spotting)", "Interior ignition\n(unit fires)"]
medians = [float(np.median(decomp["p_wildfire_direct"])),
           float(np.median(decomp["p_wildfire_indirect"])),
           float(np.median(decomp["p_interior"]))]
p95s    = [float(np.percentile(decomp["p_wildfire_direct"],95)),
           float(np.percentile(decomp["p_wildfire_indirect"],95)),
           float(np.percentile(decomp["p_interior"],95))]
x = np.arange(3)
ax.bar(x - 0.2, medians, 0.4, label="Median", color="#1f77b4")
ax.bar(x + 0.2, p95s,    0.4, label="P95",    color="#d62728")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("Annual P(structural loss) — contribution by mechanism")
ax.set_title("Phase 2 hazard decomposition — what's actually driving the risk")
ax.legend()
ax.set_yscale("log")
for i, (m, p) in enumerate(zip(medians, p95s)):
    ax.text(i - 0.2, m, f"{m:.1e}", ha='center', va='bottom', fontsize=9)
    ax.text(i + 0.2, p, f"{p:.1e}", ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig("/tmp/wildfire_p2_plot2_decomp.png", dpi=150)
print("Saved /tmp/wildfire_p2_plot2_decomp.png")
plt.close()

# Plot 3: P1 vs P2 dollar exposure
fig, ax = plt.subplots(figsize=(11, 6))
phase = ["Phase 1", "Phase 2"]
metrics = ["E[$ loss]", "P95 $ loss", "P99 $ loss"]
p1_vals = {h: [p1_dt[h]['E_loss'], p1_dt[h]['P95_loss'], p1_dt[h]['P99_loss']] for h in HORIZONS}
p2_vals = {h: [float(np.mean(dollar[h])), float(np.percentile(dollar[h],95)), float(np.percentile(dollar[h],99))] for h in HORIZONS}
x = np.arange(len(HORIZONS))
w = 0.13
ax.bar(x - 2.5*w, [PREM*h for h in HORIZONS], w, label="Cum. premium", color="#9467bd")
for k, (label, color) in enumerate(zip(["E P1","P95 P1","P99 P1","E P2","P95 P2","P99 P2"],
                                        ["#9edae5","#aec7e8","#1f77b4","#ff9896","#d62728","#8c564b"])):
    j = k % 3; phase_i = k // 3
    arr = [p1_vals[h][j] if phase_i == 0 else p2_vals[h][j] for h in HORIZONS]
    offset = (k - 2.5) * w + (0 if phase_i == 0 else 3*w)
    ax.bar(x + offset, arr, w, label=label, color=color)
ax.axhline(SHELL, linestyle="--", color="black", alpha=0.5, label="Shell ($334K)")
ax.set_xticks(x); ax.set_xticklabels([f"{h} yr" for h in HORIZONS])
ax.set_ylabel("USD")
ax.set_title("Phase 1 vs Phase 2 — $ exposure across horizons")
ax.legend(loc="upper left", fontsize=8, ncol=4)
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
plt.tight_layout()
plt.savefig("/tmp/wildfire_p2_plot3_compare.png", dpi=150)
print("Saved /tmp/wildfire_p2_plot3_compare.png")
plt.close()

# Save Phase 2 results
out = {
    "phase":             2,
    "config": {
        "vuln_scenarios":      P2_SCENARIOS,
        "ember_baseline_med":  EMBER_BASELINE_MED,
        "interior_n_units":    "4-16 uniform",
        "per_unit_fire_rate":  PER_UNIT_FIRE_RATE_PER_YR_MED,
        "p_shell_given_fire":  P_SHELL_LOSS_GIVEN_UNIT_FIRE,
    },
    "decision_table":         p2_decision,
    "hazard_decomposition": {
        "wildfire_direct_median":   float(np.median(decomp["p_wildfire_direct"])),
        "wildfire_indirect_median": float(np.median(decomp["p_wildfire_indirect"])),
        "interior_ignition_median": float(np.median(decomp["p_interior"])),
    },
    "p_annual_summary": {
        "median": float(np.median(p_annual_arr)),
        "p5":     float(np.percentile(p_annual_arr, 5)),
        "p95":    float(np.percentile(p_annual_arr, 95)),
    },
    "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
}
with open("/tmp/wildfire_p2_results.json","w") as f:
    json.dump(out, f, indent=2)
print(f"Saved /tmp/wildfire_p2_results.json")
