#!/usr/bin/env python3
"""Phase 4 — refit with BOTH corrections:
  1. Interior-ignition: p_shell_loss_given_fire from 7.1% (Beta(2,26)) → NFPA-decomposed 0.5%
  2. Wildfire BP: replace FSim's 0.0004/yr (clearly understated vs empirical) with
     empirical anchor from NIFC fire-perimeter history (1984-2025, 41 yrs):
       - 5 fires within 5 mi of parcel  →  7% annual rate of fire within 5 mi
       - Of those, P(building actually destroyed) requires another conditional
"""
import json, math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from datetime import datetime, timezone

with open("/tmp/empirical_fire_stats.json") as f: emp = json.load(f)
with open("/tmp/fsim_grid_parcel.json") as f: fsim_d = json.load(f)
with open("/tmp/dins_phase2_inputs.json") as f: p2_inp = json.load(f)
with open("/tmp/caladapt_tasmax.json") as f: clim = json.load(f)

SHELL = 334_000
PREM  = 24_000
SEED  = 42
N     = 10_000
HORIZONS = [5, 10, 15]
START_YEAR = 2026

# ===== EMPIRICAL WILDFIRE ANCHOR =====
# From NIFC history near the parcel (1984-2025):
#   - 5 fires ≥100 ac within 5 mi  (~7% annual rate of fire within 5 mi)
#   - 11 fires within 10 mi (17% annual)
#   - Gondola 2002 burned within 0.6 mi
#   - Caldor 2021 within 6 mi, destroyed ~1000 structures
#
# Annual P(loss from wildfire) = P(fire within ember-spotting distance) × P(building destroyed | fire near)
# Decompose:
#   P(fire within 5 mi of parcel)              = 0.07/yr  (empirical)
#   P(fire spreads to parcel | fire within 5 mi) = 0.10-0.30   (depends on wind, fuels, intervention)
#   P(destroyed | fire at parcel)              = 0.15-0.45  (DINS multi-res destruction rate, hardened)
# Product: 0.07 × {0.10, 0.20, 0.30} × {0.15, 0.30, 0.45} = 0.001 to 0.0095 = 0.1% to 1%/yr
# Use lognormal centered on geometric mean.
WF_ANNUAL_MED   = 0.0035   # ~0.35%/yr median annual P(wildfire-driven total loss)
WF_ANNUAL_SIGMA = 0.6      # log-normal uncertainty (wide — empirical anchor has noise)

# ===== INTERIOR IGNITION (CORRECTED) =====
# Per NFPA decomposition:
#   per_unit_fire_rate           = 6.6e-3/yr  (apartment-specific NFPA)
#   P(spreads beyond unit)       = 12%
#   P(structural damage | spread) = 30%
#   P(complete shell loss | struct dmg) = 15%
#   → P(shell loss | unit fire) = 0.12 × 0.30 × 0.15 = 0.0054   (vs old 0.071 — 13× lower)
PER_UNIT_FIRE_RATE_MED      = 6.6e-3
PER_UNIT_FIRE_RATE_SIGMA    = 0.3
P_SHELL_LOSS_BETA_A         = 5.4
P_SHELL_LOSS_BETA_B         = 994.6     # mean = 5.4/(5.4+994.6) = 0.0054, std small enough to keep it near central
N_UNITS_LO                  = 4
N_UNITS_HI                  = 17

def sample_interior(rng):
    n_units = rng.integers(N_UNITS_LO, N_UNITS_HI)
    per_unit = rng.lognormal(math.log(PER_UNIT_FIRE_RATE_MED), PER_UNIT_FIRE_RATE_SIGMA)
    p_shell  = rng.beta(P_SHELL_LOSS_BETA_A, P_SHELL_LOSS_BETA_B)
    p_unit_total = per_unit * p_shell
    return 1.0 - (1.0 - p_unit_total) ** n_units

# ===== Vulnerability scenarios (Phase 2 DINS-fitted) — keep as-is =====
P2_SCENARIOS = p2_inp["phase2_scenarios"]
vuln_mults = np.array([s["mult"] for s in P2_SCENARIOS.values()])
vuln_probs = np.array([s["prob"] for s in P2_SCENARIOS.values()])
vuln_probs = vuln_probs / vuln_probs.sum()

# ===== Climate ensemble — keep as-is =====
ensemble = {}
for name, s in clim["series"].items():
    yrs = s["years"]; tmax = s["tasmax_c"]
    base = np.mean([t for t, y in zip(tmax, yrs) if 2006 <= y <= 2025])
    ensemble[name] = {"baseline_c": base,
                      "anomaly_by_year": {y: t - base for y, t in zip(yrs, tmax) if 2026 <= y <= 2050}}
ensemble_names = list(ensemble.keys())

def f_climate(year, m, beta):
    return math.exp(beta * m["anomaly_by_year"].get(year, 0.0))

def sample_hazard_p4(rng):
    # Wildfire annual hazard sampled from lognormal empirical anchor, scaled by vulnerability
    vuln_h     = rng.choice(vuln_mults, p=vuln_probs)
    wf_raw     = rng.lognormal(math.log(WF_ANNUAL_MED), WF_ANNUAL_SIGMA)
    # The empirical anchor INCLUDES historical vulnerability — we use the DINS multiplier
    # to ADJUST relative to baseline (assume historical avg buildings have vuln ~0.5)
    # so apply a relative scaling rather than blindly multiplying.
    wf_relative_to_avg = vuln_h / 0.5  # 0.5 = approximate avg multi-res destruction rate from DINS
    h0_wildfire = wf_raw * wf_relative_to_avg

    # Interior ignition (corrected NFPA decomposition)
    h0_interior = sample_interior(rng)

    # Climate ensemble for wildfire
    member = ensemble[rng.choice(ensemble_names)]
    beta   = rng.triangular(0.5, 0.75, 1.0)
    return {"h0_wildfire": h0_wildfire, "h0_interior": h0_interior,
            "member": member, "beta": beta, "vuln_h": vuln_h}

def cum_loss(s, n_years):
    H = 0.0
    for k in range(n_years):
        y = START_YEAR + k
        H += s["h0_wildfire"] * f_climate(y, s["member"], s["beta"]) + s["h0_interior"]
    return 1.0 - math.exp(-H)

# ===== Monte Carlo =====
rng = np.random.default_rng(SEED)
p_annual_arr = np.zeros(N)
decomp = {"h0_wildfire": np.zeros(N), "h0_interior": np.zeros(N), "vuln_h": np.zeros(N)}
pcum   = {h: np.zeros(N) for h in HORIZONS}
dollar = {h: np.zeros(N, dtype=int) for h in HORIZONS}

for i in range(N):
    s = sample_hazard_p4(rng)
    decomp["h0_wildfire"][i] = s["h0_wildfire"]
    decomp["h0_interior"][i] = s["h0_interior"]
    decomp["vuln_h"][i]      = s["vuln_h"]
    p_annual_arr[i] = s["h0_wildfire"] + s["h0_interior"]
    for h in HORIZONS:
        pc = cum_loss(s, h)
        pcum[h][i] = pc
        dollar[h][i] = SHELL if rng.random() < pc else 0

print("="*80)
print("PHASE 4 — CORRECTED MODEL (empirical wildfire + fixed interior)")
print("="*80)
print(f"\nAnnual hazard components (median):")
print(f"  Wildfire:    {np.median(decomp['h0_wildfire']):.2e}/yr  ({np.median(decomp['h0_wildfire'])*100:.3f}%)")
print(f"  Interior:    {np.median(decomp['h0_interior']):.2e}/yr  ({np.median(decomp['h0_interior'])*100:.4f}%)")
print(f"  TOTAL:       {np.median(p_annual_arr):.2e}/yr  ({np.median(p_annual_arr)*100:.3f}%)")

print(f"\nRatio wildfire / interior (median): {np.median(decomp['h0_wildfire'])/np.median(decomp['h0_interior']):.1f}x")

print(f"\n{'Horizon':>8s}  {'P(loss) median':>16s}  {'5-95% band':>17s}  {'E[$ loss]':>11s}  {'P95':>11s}  {'P99':>11s}")
p4_decision = []
for h in HORIZONS:
    pl = pcum[h]; d = dollar[h]
    e = float(d.mean()); p95 = float(np.percentile(d,95)); p99 = float(np.percentile(d,99))
    cum_p = PREM * h
    row = {"horizon": h,
           "p_loss_median": float(np.median(pl)),
           "p_loss_p5":     float(np.percentile(pl, 5)),
           "p_loss_p95":    float(np.percentile(pl, 95)),
           "E_loss": e, "P95_loss": p95, "P99_loss": p99,
           "cum_premium": cum_p}
    p4_decision.append(row)
    print(f"{h:>6d} y  {row['p_loss_median']:>15.2%}  [{row['p_loss_p5']:>5.2%}–{row['p_loss_p95']:>5.2%}]  ${e:>9,.0f}  ${p95:>9,.0f}  ${p99:>9,.0f}")

# ===== Save =====
out = {
    "phase": 4,
    "config_summary": {
        "wildfire": "empirical NIFC anchor, lognormal(med=0.0035/yr, sigma=0.6)",
        "interior": f"NFPA-decomposed: per_unit={PER_UNIT_FIRE_RATE_MED}, p_shell~Beta({P_SHELL_LOSS_BETA_A},{P_SHELL_LOSS_BETA_B}), mean=0.0054",
    },
    "annual_p_loss": {
        "median": float(np.median(p_annual_arr)),
        "p5":     float(np.percentile(p_annual_arr, 5)),
        "p95":    float(np.percentile(p_annual_arr, 95)),
        "mean":   float(p_annual_arr.mean()),
    },
    "hazard_decomposition": {
        "wildfire_median": float(np.median(decomp["h0_wildfire"])),
        "interior_median": float(np.median(decomp["h0_interior"])),
    },
    "decision_table": p4_decision,
    "empirical_anchor": emp["frequency_by_radius"],
    "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
}
with open("/tmp/wildfire_p4_results.json","w") as f:
    json.dump(out, f, indent=2, default=str)
print(f"\nSaved /tmp/wildfire_p4_results.json")
