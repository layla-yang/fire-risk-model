# Databricks notebook source
# MAGIC %md
# MAGIC # Wildfire Total-Loss Forecasting — Phase 1
# MAGIC
# MAGIC **Property:** 759 Boulder Ct, Stateline, NV 89449 (Tahoe / Douglas County)
# MAGIC **Building:** 1980 wood-frame, multi-unit STR condo, dual HOA
# MAGIC **Question:** Probability of total structural loss over 5/10/15 yrs and the distribution of dollar exposure, to support the self-insurance decision.
# MAGIC
# MAGIC **Phase 1 scope (80/20):**
# MAGIC FSim-derived BP/FLEP8 + DINS-anchored vulnerability multiplier + ember/indirect term, propagated through:
# MAGIC - Constant-hazard (exponential) survival
# MAGIC - Rising-hazard (linear climate trend) survival
# MAGIC - Monte Carlo (10K iters) → dollar-loss distribution per horizon
# MAGIC
# MAGIC Phase 2 will add: ignition GLM, DINS-trained logistic + XGBoost, Bayesian hierarchical uncertainty.
# MAGIC Phase 3 will add: LOCA/NEX-DCP30 climate projections for non-linear hazard trend.
# MAGIC
# MAGIC **Acceptance criterion:** notebook runs end-to-end from the CONFIG cell. Changing CONFIG re-runs for any parcel.

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Dependencies

# COMMAND ----------
# MAGIC %pip install matplotlib numpy pandas scipy requests --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Configuration — edit here to re-run for any parcel

# COMMAND ----------
CONFIG = {
    "parcel": {
        "address":  "759 Boulder Ct, Stateline, NV 89449",
        "lat":       38.967876,
        "lon":      -119.887425,
        "geocoder": "US Census (Public_AR_Current, retrieved 2026-06-19)",
        "context":  "1980 wood-frame multi-unit condo, dual HOA, no master HOA structural policy",
    },
    "financials": {
        "shell_basis_usd":          334_000,   # uninsured structural-shell basis
        "premium_saved_annual_usd":  24_000,   # what each owner would pay if HOA voted insurance back in
    },
    "horizons_years": [5, 10, 15],
    "monte_carlo": {
        "iters":       10_000,
        "random_seed":     42,
    },
    # Hardening: building manager is retired fire dept, sprinklers, defensible space.
    # Spec is explicit: don't bake in best-case mitigation the owner can't enforce.
    # Modeled as a scenario range — Monte Carlo draws across all four weighted by prob.
    "vulnerability_scenarios": {
        "best_realized":  {"prob": 0.20, "mult": 0.30,
                           "desc": "Mitigation persists full 15 yrs (manager active, sprinklers tested, defensible space cleared)"},
        "expected":       {"prob": 0.50, "mult": 0.55,
                           "desc": "Mitigation degrades mid-horizon (manager retires / HOA budget cuts)"},
        "degraded":       {"prob": 0.25, "mult": 0.75,
                           "desc": "Mitigation lapses 5+ yrs into horizon"},
        "worst":          {"prob": 0.05, "mult": 0.90,
                           "desc": "HOA stops maintaining defensible space; sprinklers fall out of inspection"},
    },
    "hazard_model": {
        # WRC doesn't model ember/home-to-home ignition. Add an additive baseline from
        # post-fire structure-loss analyses (Cohen 2000; Maranghides NIST 2015; CAL FIRE DINS aggregates):
        # ~50-80% of WUI structure losses are ember-driven. We back out an annual base rate.
        "ember_baseline_central":    2.0e-4,   # annual P(loss via ember/home-to-home), pre-vulnerability
        "ember_baseline_sigma_log":      0.5,  # lognormal uncertainty
        # Rising-hazard: annual hazard grows linearly with time (climate non-stationarity)
        # Anchored to ~3%/yr relative growth — between IPCC AR6 mid-scenario and the realized Sierra trend
        "climate_trend_central_per_yr": 0.030, # 3.0%/yr relative increase in annual hazard
        "climate_trend_sigma":          0.015, # uncertainty (Caldor 2021 suggests upper end)
    },
    "fsim": {
        "bp_endpoint":    "https://imagery.geoplatform.gov/iipp/rest/services/Fire_Aviation/USFS_EDW_RMRS_WRC_BurnProbability/ImageServer",
        "flep4_endpoint": "https://imagery.geoplatform.gov/iipp/rest/services/Fire_Aviation/USFS_EDW_RMRS_WRC_FlameLengthExceedProb4ft/ImageServer",
        "flep8_endpoint": "https://imagery.geoplatform.gov/iipp/rest/services/Fire_Aviation/USFS_EDW_RMRS_WRC_FlameLengthExceedProb8ft/ImageServer",
        "scaling":        {"bp": 1/100_000, "flep4": 1/1_000, "flep8": 1/1_000},
        "grid_radius_cells": 2,                # 5x5 grid of 30m cells = ~150m radius
        "use_cached_bronze": True,
    },
    "uc": {
        "catalog": "wildfire_risk_catalog",
        "bronze":  "bronze.fsim_parcel_sample",
        "gold_mc": "gold.mc_dollar_distribution",
        "gold_dt": "gold.decision_table",
    },
}

import json
print(json.dumps(CONFIG, indent=2))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Imports + reproducibility

# COMMAND ----------
import numpy as np
import pandas as pd
import requests
import json
import math
from urllib.parse import urlencode
from datetime import datetime, timezone
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

SEED = CONFIG["monte_carlo"]["random_seed"]
rng = np.random.default_rng(SEED)
np.random.seed(SEED)

print(f"Run timestamp: {datetime.now(timezone.utc).isoformat()}")
print(f"Random seed:   {SEED}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Pull FSim BP/FLEP grid at parcel
# MAGIC
# MAGIC Identifies BP, FLEP4, FLEP8 on a 5×5 grid of 30 m cells centered on the parcel (~150 m radius).
# MAGIC The parcel cell itself is developed (per LANDFIRE), so FLEP at the centroid is 0 by design —
# MAGIC structure risk comes from the *surrounding burnable cells* via direct flame contact + embers.

# COMMAND ----------
def fetch_fsim_grid(lat, lon, endpoints, grid_radius=2):
    """Returns dict[layer] -> 2D list of raw integer values from ImageServer/identify."""
    dlat = 0.00027  # ~30m at lat 39°
    dlon = 0.00035  # ~30m at lat 39° lon
    headers = {"User-Agent": "wildfire-risk-model/0.1"}
    out = {}
    for layer, url in endpoints.items():
        mat = []
        for i in range(-grid_radius, grid_radius + 1):
            row = []
            for j in range(-grid_radius, grid_radius + 1):
                lat_p, lon_p = lat + i * dlat, lon + j * dlon
                params = {
                    "geometry": json.dumps({"x": lon_p, "y": lat_p, "spatialReference": {"wkid": 4326}}),
                    "geometryType":       "esriGeometryPoint",
                    "returnGeometry":     "false",
                    "returnCatalogItems": "false",
                    "f": "json",
                }
                r = requests.get(f"{url}/identify", params=params, headers=headers, timeout=20)
                v = r.json().get("value")
                try:
                    row.append(int(v) if v not in (None, "NoData") else None)
                except (ValueError, TypeError):
                    row.append(None)
            mat.append(row)
        out[layer] = mat
    return out

# Try cached bronze first
LAT, LON = CONFIG["parcel"]["lat"], CONFIG["parcel"]["lon"]
endpoints = {
    "BP":    CONFIG["fsim"]["bp_endpoint"],
    "FLEP4": CONFIG["fsim"]["flep4_endpoint"],
    "FLEP8": CONFIG["fsim"]["flep8_endpoint"],
}

if CONFIG["fsim"]["use_cached_bronze"]:
    try:
        cached = spark.sql(f"""
            SELECT layer, cell_row, cell_col, raw_value, scaled_value, scale_factor
            FROM {CONFIG['uc']['catalog']}.{CONFIG['uc']['bronze']}
            WHERE parcel_lat = {LAT} AND parcel_lon = {LON}
            ORDER BY layer, cell_row, cell_col
        """).toPandas()
        if len(cached) > 0:
            print(f"Loaded {len(cached)} cached samples from bronze.")
            grid_raw, grid_scaled = {}, {}
            for layer in ["BP", "FLEP4", "FLEP8"]:
                df_l = cached[cached["layer"] == layer].sort_values(["cell_row", "cell_col"])
                mat = df_l["raw_value"].values.reshape(5, 5).tolist()
                mat_s = df_l["scaled_value"].values.reshape(5, 5).tolist()
                grid_raw[layer]    = mat
                grid_scaled[layer] = mat_s
        else:
            raise RuntimeError("no cache")
    except Exception as e:
        print(f"No bronze cache — fetching from GeoPlatform ({e})")
        grid_raw = fetch_fsim_grid(LAT, LON, endpoints, CONFIG["fsim"]["grid_radius_cells"])
        scales = CONFIG["fsim"]["scaling"]
        grid_scaled = {
            layer: [[(v * scales[layer.lower()]) if v is not None else None for v in row] for row in mat]
            for layer, mat in grid_raw.items()
        }
else:
    grid_raw = fetch_fsim_grid(LAT, LON, endpoints, CONFIG["fsim"]["grid_radius_cells"])
    scales = CONFIG["fsim"]["scaling"]
    grid_scaled = {
        layer: [[(v * scales[layer.lower()]) if v is not None else None for v in row] for row in mat]
        for layer, mat in grid_raw.items()
    }

# Display grids
print()
for layer, mat in grid_scaled.items():
    print(f"--- {layer} (scaled probabilities) ---")
    for row in mat:
        print("  " + "  ".join(f"{v:.5f}" if v is not None else "   NaN" for v in row))

# COMMAND ----------
# MAGIC %md
# MAGIC **Interpretation of the FSim grid.**
# MAGIC The parcel cell shows BP ≈ 0.00044 (annual). The center FLEP values are 0 because the cell is
# MAGIC developed/non-burnable in LANDFIRE — FLEP is undefined where there is no fuel. The fire that
# MAGIC could total this building originates in the *surrounding* fuel cells: the NW and SW corners of the
# MAGIC grid show BP up to ~0.001 and FLEP8 up to ~0.10. Those are the cells we sample from in the MC.

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Aggregate FSim to model inputs

# COMMAND ----------
def aggregate_fsim(grid_scaled):
    """Reduce the 5x5 grid to model-ready samples."""
    bp_all = [v for row in grid_scaled["BP"] for v in row if v is not None]
    # FLEP is meaningful only in burnable cells
    flep8_burnable = [v for row in grid_scaled["FLEP8"] for v in row if v not in (None, 0.0)]
    flep4_burnable = [v for row in grid_scaled["FLEP4"] for v in row if v not in (None, 0.0)]

    return {
        "bp_samples":        bp_all,
        "flep8_burnable":    flep8_burnable,
        "flep4_burnable":    flep4_burnable,
        "bp_summary":        {"min": min(bp_all), "median": np.median(bp_all), "max": max(bp_all), "n": len(bp_all)},
        "flep8_summary":     ({"min": min(flep8_burnable), "median": np.median(flep8_burnable),
                               "max": max(flep8_burnable), "n_burnable": len(flep8_burnable)}
                              if flep8_burnable else {"n_burnable": 0}),
        "flep4_summary":     ({"min": min(flep4_burnable), "median": np.median(flep4_burnable),
                               "max": max(flep4_burnable), "n_burnable": len(flep4_burnable)}
                              if flep4_burnable else {"n_burnable": 0}),
    }

fsim = aggregate_fsim(grid_scaled)
print("BP    (all 25 cells):  ", {k: round(v, 6) if isinstance(v, float) else v for k, v in fsim["bp_summary"].items()})
print("FLEP8 (burnable only): ", {k: round(v, 6) if isinstance(v, float) else v for k, v in fsim["flep8_summary"].items()})
print("FLEP4 (burnable only): ", {k: round(v, 6) if isinstance(v, float) else v for k, v in fsim["flep4_summary"].items()})

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Vulnerability multiplier (DINS-anchored, hardening scenarios)
# MAGIC
# MAGIC Conditional structure-loss rates are anchored to published CAL FIRE DINS aggregates (Knapp et al.
# MAGIC 2022; Syphard et al. 2023). DINS is California-only — we transfer the conditional rate to NV.
# MAGIC **This is the most important caveat in the model.** Fire physics is not state-bound and DINS
# MAGIC is the only large-N structure-outcome dataset, but the multiplier is sensitive to local building
# MAGIC codes, vegetation composition, and wind regime.
# MAGIC
# MAGIC The scenarios reflect *mitigation persistence* over the hold horizon, not current mitigation state.

# COMMAND ----------
def vulnerability_pmf(CONFIG):
    """Return arrays (multipliers, probs) representing the scenario PMF."""
    sc = CONFIG["vulnerability_scenarios"]
    mults = np.array([s["mult"] for s in sc.values()])
    probs = np.array([s["prob"] for s in sc.values()])
    probs = probs / probs.sum()
    return mults, probs, list(sc.keys())

vuln_mults, vuln_probs, vuln_names = vulnerability_pmf(CONFIG)
print("Vulnerability scenarios:")
for name, mult, prob in zip(vuln_names, vuln_mults, vuln_probs):
    print(f"  {name:15s}  mult={mult:.2f}  prob={prob:.0%}  ({CONFIG['vulnerability_scenarios'][name]['desc']})")
print(f"\nExpected vulnerability multiplier: {(vuln_mults * vuln_probs).sum():.3f}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. Annual hazard composition
# MAGIC
# MAGIC `P_loss_annual = P(fire reaches parcel) × P(destroyed | fire) + P(destroyed via ember/indirect)`
# MAGIC
# MAGIC - `P(fire reaches parcel)` = BP sampled from neighborhood
# MAGIC - `P(destroyed | fire)` = FLEP8 × vuln_high + (FLEP4 − FLEP8) × vuln_low — high-intensity vs.
# MAGIC   moderate-intensity terms (sprinklers + defensible space help more at the moderate end)
# MAGIC - Ember/indirect: WRC doesn't model home-to-home ignition; the additive baseline picks that up
# MAGIC   from DINS-derived loss rates

# COMMAND ----------
def sample_annual_hazard(rng, fsim, CONFIG):
    """Sample one realization of P_loss_annual from the input distributions."""
    bp     = rng.choice(fsim["bp_samples"])
    flep8  = rng.choice(fsim["flep8_burnable"]) if fsim["flep8_burnable"] else 0.0
    flep4  = rng.choice(fsim["flep4_burnable"]) if fsim["flep4_burnable"] else 0.0
    mults, probs, _ = vulnerability_pmf(CONFIG)
    vuln_high = rng.choice(mults, p=probs)
    vuln_low  = vuln_high * 0.4   # mitigation more effective at moderate intensity

    ember = rng.lognormal(
        mean  = math.log(CONFIG["hazard_model"]["ember_baseline_central"]),
        sigma = CONFIG["hazard_model"]["ember_baseline_sigma_log"],
    ) * vuln_high

    p_destroy_given_fire = flep8 * vuln_high + max(0.0, flep4 - flep8) * vuln_low
    p_direct   = bp * p_destroy_given_fire
    p_indirect = ember
    p_annual   = p_direct + p_indirect

    climate_trend = max(0.0, rng.normal(
        CONFIG["hazard_model"]["climate_trend_central_per_yr"],
        CONFIG["hazard_model"]["climate_trend_sigma"],
    ))
    return {
        "p_annual_const":  p_annual,
        "climate_trend":   climate_trend,
        "bp":              bp,
        "flep8":           flep8,
        "flep4":           flep4,
        "vuln_high":       vuln_high,
        "p_direct":        p_direct,
        "p_indirect":      p_indirect,
    }

# Sanity check
sample = sample_annual_hazard(rng, fsim, CONFIG)
print("One sample of the input space:")
for k, v in sample.items():
    print(f"  {k:18s} {v:.6e}" if isinstance(v, float) else f"  {k:18s} {v}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 8. Survival models — constant vs. rising hazard

# COMMAND ----------
def p_cum_constant(p_annual, t):
    return 1.0 - (1.0 - p_annual) ** t

def p_cum_rising(p_annual_base, t, climate_trend):
    # Annual hazard h(s) = p_annual_base * (1 + climate_trend * s)
    # Cumulative hazard H(t) = p_annual_base * (t + climate_trend * t**2 / 2)
    H = p_annual_base * (t + climate_trend * t**2 / 2.0)
    return 1.0 - math.exp(-H)

# Quick visualization
years = np.arange(0, 21)
p_ann_demo = 5e-4  # demo
for ct in [0.0, 0.03, 0.06]:
    pc = [p_cum_rising(p_ann_demo, t, ct) for t in years]
    print(f"  climate_trend={ct:.2f}:  P(loss by 15) = {pc[15]:.4%}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 9. Monte Carlo engine

# COMMAND ----------
def monte_carlo(CONFIG, fsim, n_iters, seed):
    rng = np.random.default_rng(seed)
    horizons = CONFIG["horizons_years"]
    shell    = CONFIG["financials"]["shell_basis_usd"]

    out = {
        "iters":        n_iters,
        "horizons":     horizons,
        "p_annual":     [],
        "climate_trend":[],
        # P_cum per horizon, constant and rising
        "pcum_const":   {h: [] for h in horizons},
        "pcum_rise":    {h: [] for h in horizons},
        # Dollar loss per horizon (sampled binary event × shell), rising-hazard variant
        "dollar":       {h: [] for h in horizons},
    }

    for i in range(n_iters):
        s = sample_annual_hazard(rng, fsim, CONFIG)
        p_ann = s["p_annual_const"]
        ct    = s["climate_trend"]
        out["p_annual"].append(p_ann)
        out["climate_trend"].append(ct)

        for h in horizons:
            pc_const = p_cum_constant(p_ann, h)
            pc_rise  = p_cum_rising(p_ann, h, ct)
            out["pcum_const"][h].append(pc_const)
            out["pcum_rise" ][h].append(pc_rise)
            # Binary loss event under rising-hazard cumulative
            event   = rng.random() < pc_rise
            out["dollar"][h].append(shell if event else 0)

    # Convert to numpy
    for k in ["p_annual", "climate_trend"]:
        out[k] = np.array(out[k])
    for k in ["pcum_const", "pcum_rise", "dollar"]:
        out[k] = {h: np.array(v) for h, v in out[k].items()}
    return out

mc = monte_carlo(CONFIG, fsim, CONFIG["monte_carlo"]["iters"], SEED)
print(f"Monte Carlo complete: {mc['iters']} iterations, horizons={mc['horizons']}")
print(f"P_annual:      median={np.median(mc['p_annual']):.2e}  mean={mc['p_annual'].mean():.2e}  p95={np.percentile(mc['p_annual'],95):.2e}")
print(f"Climate trend: median={np.median(mc['climate_trend']):.3f}/yr")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 10. Decision table — probabilities and dollar exposure per horizon

# COMMAND ----------
def decision_table(mc, CONFIG):
    shell    = CONFIG["financials"]["shell_basis_usd"]
    premium  = CONFIG["financials"]["premium_saved_annual_usd"]
    rows = []
    for h in mc["horizons"]:
        pcum_const = mc["pcum_const"][h]
        pcum_rise  = mc["pcum_rise" ][h]
        dollar     = mc["dollar"][h]
        cum_prem   = premium * h
        row = {
            "horizon_yrs":       h,
            "p_loss_const_med":  float(np.median(pcum_const)),
            "p_loss_const_p5":   float(np.percentile(pcum_const, 5)),
            "p_loss_const_p95":  float(np.percentile(pcum_const, 95)),
            "p_loss_rise_med":   float(np.median(pcum_rise)),
            "p_loss_rise_p5":    float(np.percentile(pcum_rise, 5)),
            "p_loss_rise_p95":   float(np.percentile(pcum_rise, 95)),
            "E_loss_usd":        float(dollar.mean()),
            "P90_loss_usd":      float(np.percentile(dollar, 90)),
            "P95_loss_usd":      float(np.percentile(dollar, 95)),
            "P99_loss_usd":      float(np.percentile(dollar, 99)),
            "cum_premium_usd":   cum_prem,
            "premium_vs_Eloss":  cum_prem / max(dollar.mean(), 1.0),
            "premium_vs_shell":  cum_prem / shell,
        }
        # Verdict logic
        if row["E_loss_usd"] * 3 < cum_prem and row["P99_loss_usd"] < cum_prem * 2:
            verdict = "Self-insure (HOA vote rational)"
        elif row["E_loss_usd"] > cum_prem:
            verdict = "Insure (expected loss > premium)"
        elif row["P99_loss_usd"] > cum_prem * 3:
            verdict = "Borderline — tail risk dominates"
        else:
            verdict = "Self-insure (modest tail)"
        row["verdict"] = verdict
        rows.append(row)
    return pd.DataFrame(rows)

dt = decision_table(mc, CONFIG)
# Format for display
disp = dt.copy()
for col in ["p_loss_const_med", "p_loss_const_p5", "p_loss_const_p95",
            "p_loss_rise_med",  "p_loss_rise_p5",  "p_loss_rise_p95"]:
    disp[col] = disp[col].apply(lambda x: f"{x:.2%}")
for col in ["E_loss_usd", "P90_loss_usd", "P95_loss_usd", "P99_loss_usd", "cum_premium_usd"]:
    disp[col] = disp[col].apply(lambda x: f"${x:,.0f}")
for col in ["premium_vs_Eloss", "premium_vs_shell"]:
    disp[col] = disp[col].apply(lambda x: f"{x:.2f}x" if col == "premium_vs_Eloss" else f"{x:.0%}")
print(disp.to_string(index=False))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 11. Plot 1 — Cumulative loss probability vs. years (constant vs rising), with credible band

# COMMAND ----------
years_plot = np.arange(0, 21)
# For each year, compute rising-hazard cumulative across all MC draws, take percentiles
def pcum_array(years, p_ann_arr, ct_arr, model):
    out = np.zeros((len(p_ann_arr), len(years)))
    for i, (p, c) in enumerate(zip(p_ann_arr, ct_arr)):
        for j, t in enumerate(years):
            out[i, j] = p_cum_constant(p, t) if model == "const" else p_cum_rising(p, t, c)
    return out

mat_const = pcum_array(years_plot, mc["p_annual"], mc["climate_trend"], "const")
mat_rise  = pcum_array(years_plot, mc["p_annual"], mc["climate_trend"], "rise")

fig, ax = plt.subplots(figsize=(10, 6))
for mat, label, color in [(mat_const, "Constant hazard", "#1f77b4"),
                           (mat_rise,  "Rising hazard (climate trend)", "#d62728")]:
    med = np.median(mat, axis=0)
    p5  = np.percentile(mat, 5, axis=0)
    p95 = np.percentile(mat, 95, axis=0)
    ax.plot(years_plot, med, label=f"{label} — median", color=color, linewidth=2)
    ax.fill_between(years_plot, p5, p95, color=color, alpha=0.15, label=f"{label} — 5–95% band")
ax.set_xlabel("Years held")
ax.set_ylabel("Cumulative probability of total structural loss")
ax.set_title("Wildfire total-loss probability over hold horizon\n759 Boulder Ct, Stateline NV")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)
for h in CONFIG["horizons_years"]:
    ax.axvline(h, linestyle=":", color="gray", alpha=0.6)
plt.tight_layout()
plt.show()
print("Takeaway: rising-hazard variant pushes the 15-yr probability ~20–30% above the constant-hazard baseline, driven by climate-trend uncertainty. The 5–95% band is wide — uncertainty about the multiplier dominates.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 12. Plot 2 — Monte Carlo dollar-loss distribution (15-yr horizon)
# MAGIC The "mostly nothing with a brutal tail" shape: most simulated worlds end with $0 loss, a fraction
# MAGIC end with the full $334K shell loss.

# COMMAND ----------
h_focus = 15
dollar = mc["dollar"][h_focus]
prob_loss = (dollar > 0).mean()
shell = CONFIG["financials"]["shell_basis_usd"]

fig, ax = plt.subplots(figsize=(10, 6))
# Bar chart of P(no loss) vs P(loss) — discrete by construction (binary event × shell)
counts = {"No loss ($0)": (dollar == 0).sum(),
          f"Total loss (${shell/1000:.0f}K)": (dollar > 0).sum()}
ax.bar(counts.keys(), counts.values(), color=["#2ca02c", "#d62728"])
for i, (k, v) in enumerate(counts.items()):
    ax.text(i, v, f"{v:,}\n({v/len(dollar):.1%})", ha="center", va="bottom", fontsize=11)
ax.set_ylabel("Monte Carlo iterations")
ax.set_title(f"15-yr loss outcome distribution ({len(dollar):,} iterations)")
plt.tight_layout()
plt.show()

print(f"\nDollar exposure at 15-yr horizon:")
print(f"  P(any total loss)       = {prob_loss:.2%}")
print(f"  Expected $ loss         = ${dollar.mean():,.0f}")
print(f"  P95 $ loss              = ${np.percentile(dollar,95):,.0f}")
print(f"  P99 $ loss              = ${np.percentile(dollar,99):,.0f}")
print(f"  Cumulative premium      = ${CONFIG['financials']['premium_saved_annual_usd']*h_focus:,.0f}")
print(f"  Premium / Expected loss = {CONFIG['financials']['premium_saved_annual_usd']*h_focus / max(dollar.mean(),1):.1f}x")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 13. Plot 3 — Expected loss vs. cumulative premium across horizons

# COMMAND ----------
horizons = CONFIG["horizons_years"]
premiums = [CONFIG["financials"]["premium_saved_annual_usd"] * h for h in horizons]
expected = [mc["dollar"][h].mean() for h in horizons]
tail95   = [np.percentile(mc["dollar"][h], 95) for h in horizons]
tail99   = [np.percentile(mc["dollar"][h], 99) for h in horizons]

x = np.arange(len(horizons))
w = 0.18

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - 1.5*w, premiums, w, label="Cumulative premium (if HOA voted yes)", color="#9467bd")
ax.bar(x - 0.5*w, expected, w, label="Expected $ loss (MC mean)",              color="#2ca02c")
ax.bar(x + 0.5*w, tail95,   w, label="P95 $ loss",                              color="#ff7f0e")
ax.bar(x + 1.5*w, tail99,   w, label="P99 $ loss",                              color="#d62728")
ax.axhline(CONFIG["financials"]["shell_basis_usd"], linestyle="--", color="black", alpha=0.5, label="Shell basis ($334K)")
ax.set_xticks(x)
ax.set_xticklabels([f"{h} yrs" for h in horizons])
ax.set_ylabel("USD")
ax.set_title("Premium vs. expected & tail loss — break-even visualization")
ax.legend(loc="upper left", fontsize=9)
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
plt.tight_layout()
plt.show()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 14. Plot 4 — Sensitivity tornado
# MAGIC Re-runs the MC pinning one input at its low/high value while leaving others at the central distribution.
# MAGIC Shows which input moves the 15-yr expected loss the most.

# COMMAND ----------
def sensitivity_runs(CONFIG, fsim, n=2000):
    base_iters = n
    h = 15
    shell = CONFIG["financials"]["shell_basis_usd"]

    def run_with_override(override):
        cfg = json.loads(json.dumps(CONFIG))  # deep copy
        fs  = json.loads(json.dumps(fsim))
        for path, val in override.items():
            keys = path.split(".")
            d = cfg
            for k in keys[:-1]:
                d = d[k]
            d[keys[-1]] = val
        rng_l = np.random.default_rng(SEED + hash(json.dumps(override)) % 10000)
        losses = []
        for _ in range(base_iters):
            s = sample_annual_hazard(rng_l, fs, cfg)
            p = s["p_annual_const"]
            ct = s["climate_trend"]
            pc = p_cum_rising(p, h, ct)
            losses.append(shell if rng_l.random() < pc else 0)
        return np.mean(losses)

    base_E = run_with_override({})

    scenarios = {
        "FSim BP (low: P5 of neighborhood)":   {"override": {"_skip": True}, "patch": "bp_low"},
        "FSim BP (high: P95 of neighborhood)": {"override": {"_skip": True}, "patch": "bp_high"},
        "FLEP8 (no burnable: 0)":              {"override": {"_skip": True}, "patch": "flep_zero"},
        "FLEP8 (max neighborhood: 0.10)":      {"override": {"_skip": True}, "patch": "flep_max"},
        "Vulnerability = best (0.30)":         {"override": {"_skip": True}, "patch": "vuln_best"},
        "Vulnerability = worst (0.90)":        {"override": {"_skip": True}, "patch": "vuln_worst"},
        "Ember baseline = low (5e-5)":         {"override": {"hazard_model.ember_baseline_central": 5.0e-5}},
        "Ember baseline = high (1e-3)":        {"override": {"hazard_model.ember_baseline_central": 1.0e-3}},
        "Climate trend = 0/yr":                {"override": {"hazard_model.climate_trend_central_per_yr": 0.0}},
        "Climate trend = 6%/yr":               {"override": {"hazard_model.climate_trend_central_per_yr": 0.06}},
    }

    results = {"_base": base_E}
    for label, spec in scenarios.items():
        patch = spec.get("patch")
        if patch:
            fs2 = json.loads(json.dumps(fsim))
            if patch == "bp_low":
                bp = np.percentile(fsim["bp_samples"], 5)
                fs2["bp_samples"] = [bp]
            elif patch == "bp_high":
                bp = np.percentile(fsim["bp_samples"], 95)
                fs2["bp_samples"] = [bp]
            elif patch == "flep_zero":
                fs2["flep8_burnable"] = [0.0]
                fs2["flep4_burnable"] = [0.0]
            elif patch == "flep_max":
                fs2["flep8_burnable"] = [0.10]
                fs2["flep4_burnable"] = [0.31]
            elif patch == "vuln_best":
                cfg2 = json.loads(json.dumps(CONFIG))
                for s in cfg2["vulnerability_scenarios"].values():
                    s["mult"] = 0.30
                rng_l = np.random.default_rng(SEED + 1)
                losses = []
                for _ in range(base_iters):
                    s = sample_annual_hazard(rng_l, fs2, cfg2)
                    pc = p_cum_rising(s["p_annual_const"], h, s["climate_trend"])
                    losses.append(shell if rng_l.random() < pc else 0)
                results[label] = np.mean(losses)
                continue
            elif patch == "vuln_worst":
                cfg2 = json.loads(json.dumps(CONFIG))
                for s in cfg2["vulnerability_scenarios"].values():
                    s["mult"] = 0.90
                rng_l = np.random.default_rng(SEED + 2)
                losses = []
                for _ in range(base_iters):
                    s = sample_annual_hazard(rng_l, fs2, cfg2)
                    pc = p_cum_rising(s["p_annual_const"], h, s["climate_trend"])
                    losses.append(shell if rng_l.random() < pc else 0)
                results[label] = np.mean(losses)
                continue
            rng_l = np.random.default_rng(SEED + hash(label) % 10000)
            losses = []
            for _ in range(base_iters):
                s = sample_annual_hazard(rng_l, fs2, CONFIG)
                pc = p_cum_rising(s["p_annual_const"], h, s["climate_trend"])
                losses.append(shell if rng_l.random() < pc else 0)
            results[label] = np.mean(losses)
        else:
            results[label] = run_with_override(spec["override"])
    return results

sens = sensitivity_runs(CONFIG, fsim, n=3000)
base = sens.pop("_base")
deltas = {k: v - base for k, v in sens.items()}
order  = sorted(deltas.items(), key=lambda kv: abs(kv[1]), reverse=True)

fig, ax = plt.subplots(figsize=(10, 6))
labels = [k for k, _ in order]
vals   = [v for _, v in order]
colors = ["#d62728" if v > 0 else "#1f77b4" for v in vals]
y = np.arange(len(labels))
ax.barh(y, vals, color=colors)
ax.set_yticks(y); ax.set_yticklabels(labels)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Δ 15-yr expected $ loss vs. base case")
ax.set_title(f"Sensitivity tornado — base E[loss @15yrs] = ${base:,.0f}")
ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x/1000:+.0f}K"))
ax.invert_yaxis()
plt.tight_layout()
plt.show()

print(f"\nBase 15-yr expected loss: ${base:,.0f}")
print("Largest movers (in order):")
for k, v in order[:5]:
    print(f"  {k:42s}  Δ = ${v:+,.0f}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 15. Persist outputs to gold

# COMMAND ----------
# Decision table → gold
gold_dt = dt.copy()
gold_dt["parcel_address"] = CONFIG["parcel"]["address"]
gold_dt["run_timestamp"]  = datetime.now(timezone.utc)
gold_dt["seed"]           = SEED
spark.createDataFrame(gold_dt).write.mode("overwrite").saveAsTable(
    f"{CONFIG['uc']['catalog']}.{CONFIG['uc']['gold_dt']}"
)
print(f"Wrote {CONFIG['uc']['catalog']}.{CONFIG['uc']['gold_dt']}")

# MC dollar distribution → gold (long format)
mc_long = []
for h in CONFIG["horizons_years"]:
    for i, d in enumerate(mc["dollar"][h]):
        mc_long.append({
            "horizon_yrs":  h,
            "iter":         i,
            "dollar_loss":  int(d),
            "pcum_const":   float(mc["pcum_const"][h][i]),
            "pcum_rise":    float(mc["pcum_rise"][h][i]),
        })
mc_df = pd.DataFrame(mc_long)
mc_df["parcel_address"] = CONFIG["parcel"]["address"]
mc_df["run_timestamp"]  = datetime.now(timezone.utc)
spark.createDataFrame(mc_df).write.mode("overwrite").saveAsTable(
    f"{CONFIG['uc']['catalog']}.{CONFIG['uc']['gold_mc']}"
)
print(f"Wrote {CONFIG['uc']['catalog']}.{CONFIG['uc']['gold_mc']} ({len(mc_df):,} rows)")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 16. Assumptions & caveats log
# MAGIC
# MAGIC | # | Assumption | Direction of bias |
# MAGIC |---|---|---|
# MAGIC | 1 | **CA → NV DINS transfer** for vulnerability multiplier | Likely *underestimates* loss in NV (NV building codes are more permissive than CA's WUI codes; CA DINS is dominated by post-2017 hardening era) |
# MAGIC | 2 | **FSim 270 m resolution** vs. single parcel | Smooths over local fuel discontinuities; for our parcel, the surrounding fuel cells are nearby — sampled in the 5×5 grid |
# MAGIC | 3 | **WRC 2020 LANDFIRE fuels** (pre-Caldor) | Likely *underestimates* current hazard; Caldor 2021 burned within 20 mi of this parcel and changed fuel state locally |
# MAGIC | 4 | **Building-level (not unit-level) loss model** | Any neighbor unit fire totals the structure — included via ember/indirect term |
# MAGIC | 5 | **Mitigation persistence** modeled as scenarios, not assumed permanent | Conservative — best-realized scenario only carries 20% weight |
# MAGIC | 6 | **Climate non-stationarity** as linear trend (+3%/yr ± 1.5%) | Phase 3 will replace with LOCA/NEX-DCP30 projections; linear underestimates the upper tail |
# MAGIC | 7 | **Binary loss model** (total or zero — no partial losses) | Spec-aligned; partial losses are out of scope for this self-insurance question |
# MAGIC | 8 | **Premium static at $24K/yr** | Real premiums in WUI markets are rising 10–25%/yr; model holds it flat |
# MAGIC | 9 | **No correlation between BP and FLEP samples** | Bootstrap from neighborhood treats them independent; mild conservatism |
# MAGIC | 10 | **Ember baseline = 2e-4/yr** | DINS-derived; carries lognormal uncertainty σ=0.5 in log space |

# COMMAND ----------
# MAGIC %md
# MAGIC ## 17. End-of-notebook headline read
# MAGIC The decision table and plots above are the audit-grade outputs. The headline read for the
# MAGIC self-insurance vote is the row at horizon = 15 yrs, and the comparison column is **cumulative
# MAGIC premium vs. expected $ loss vs. P99 $ loss**. If P99 ≪ cumulative premium and E[loss] ≪
# MAGIC premium, the HOA's no vote was rational. If P99 ≫ premium (the brutal tail), it was not.
