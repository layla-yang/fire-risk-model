# Databricks notebook source
# MAGIC %md
# MAGIC # Wildfire Total-Loss Forecasting — Consolidated (Phase 1 + 2 + 2.5 + 3)
# MAGIC
# MAGIC **Property:** 759 Boulder Ct, Stateline, NV 89449 (Tahoe / Douglas County)
# MAGIC **Building:** 1980 wood-frame, multi-unit STR condo, dual HOA
# MAGIC **Question:** Probability of total structural loss over 5/10/15 yrs and the distribution of dollar exposure, to support the self-insurance decision against a $24K/yr per-owner premium that the HOA voted down.
# MAGIC
# MAGIC | Phase | What it adds | Key change |
# MAGIC |---|---|---|
# MAGIC | **1** | FSim BP × FLEP × hand-coded vulnerability + Weibull MC | Baseline 80/20 model |
# MAGIC | **2** | DINS-fitted vulnerability + interior-ignition baseline | Replaces hand-coded multiplier; **adds the dominant interior-fire term Phase 1 missed entirely** |
# MAGIC | **2.5** | PyMC Bayesian hierarchical w/ county-level partial pooling | Credible intervals on the multiplier; CA→NV transfer via population-level mean |
# MAGIC | **3** | Cal-Adapt LOCA CMIP5 ensemble → climate-driven hazard | Per-year trajectory instead of linear trend; only modulates wildfire (not interior) |
# MAGIC
# MAGIC **Acceptance:** Notebook runs end-to-end from the CONFIG cell. Changing CONFIG re-runs for any parcel.

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Dependencies

# COMMAND ----------
# MAGIC %pip install --quiet matplotlib numpy pandas scipy requests pymc==5.* arviz
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Configuration — change here to re-run for any parcel

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
        "shell_basis_usd":          334_000,
        "premium_saved_annual_usd":  24_000,
    },
    "horizons_years":           [5, 10, 15],
    "horizon_start_year":       2026,
    "monte_carlo":              {"iters": 10_000, "random_seed": 42},

    # Phase 2 — hardening scenarios (DINS-fitted; updated by section 7)
    "vulnerability_scenarios_default": {
        "best_realized":  {"prob": 0.20, "mult": 0.026},
        "expected":       {"prob": 0.50, "mult": 0.177},
        "degraded":       {"prob": 0.25, "mult": 0.362},
        "worst":          {"prob": 0.05, "mult": 0.583},
    },
    "interior_ignition": {
        "per_unit_fire_rate_per_yr_central": 3.0e-3,   # NFPA US multi-family
        "per_unit_fire_rate_sigma_log":      0.3,
        "p_shell_loss_given_fire_beta":      (2.0, 26.0),  # Beta(α,β), mean = 0.071
        "n_units_lo":  4,
        "n_units_hi":  17,
    },
    "ember_baseline": {
        "median":    2.0e-4,
        "sigma_log": 0.6,
    },

    # Phase 3 — climate
    "climate": {
        "cal_adapt_models":    ["HadGEM2-ES", "CNRM-CM5", "CanESM2", "MIROC5"],
        "cal_adapt_scenarios": ["rcp45", "rcp85"],
        "beta_per_C_central":   0.75,
        "beta_per_C_lo":        0.50,
        "beta_per_C_hi":        1.00,
    },

    # Data sources
    "fsim": {
        "bp_endpoint":    "https://imagery.geoplatform.gov/iipp/rest/services/Fire_Aviation/USFS_EDW_RMRS_WRC_BurnProbability/ImageServer",
        "flep4_endpoint": "https://imagery.geoplatform.gov/iipp/rest/services/Fire_Aviation/USFS_EDW_RMRS_WRC_FlameLengthExceedProb4ft/ImageServer",
        "flep8_endpoint": "https://imagery.geoplatform.gov/iipp/rest/services/Fire_Aviation/USFS_EDW_RMRS_WRC_FlameLengthExceedProb8ft/ImageServer",
        "scaling":        {"bp": 1/100_000, "flep4": 1/1_000, "flep8": 1/1_000},
        "grid_radius_cells": 2,
    },
    "dins_feature_server": "https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/arcgis/rest/services/POSTFIRE_MASTER_DATA_SHARE/FeatureServer/0",
    "cal_adapt_api":       "https://api.cal-adapt.org/api/series",
    "uc": {
        "catalog": "wildfire_risk_catalog",
        "bronze_fsim":     "bronze.fsim_parcel_sample",
        "bronze_dins":     "bronze.dins_raw",
        "bronze_climate":  "bronze.caladapt_tasmax",
        "gold_decision":   "gold.decision_table_phase2",
        "gold_mc":         "gold.mc_dollar_distribution",
        "raw_volume":      "/Volumes/wildfire_risk_catalog/bronze/raw",
    },
}

import json
print(json.dumps(CONFIG, indent=2, default=str))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Imports + reproducibility

# COMMAND ----------
import numpy as np
import pandas as pd
import requests
import json
import math
import time
from urllib.parse import urlencode
from datetime import datetime, timezone, timedelta
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

SEED = CONFIG["monte_carlo"]["random_seed"]
rng_global = np.random.default_rng(SEED)
np.random.seed(SEED)
print(f"Run timestamp: {datetime.now(timezone.utc).isoformat()}")
print(f"Random seed:   {SEED}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Pull FSim BP/FLEP grid at parcel
# MAGIC Identify BP, FLEP4, FLEP8 on a 5×5 grid of 30 m cells (~150 m radius) centered on the parcel.
# MAGIC Persist raw + scaled values to `bronze.fsim_parcel_sample` with provenance.

# COMMAND ----------
def fetch_fsim_grid(lat, lon, endpoints, grid_radius=2):
    dlat, dlon = 0.00027, 0.00035  # ~30 m at lat 39°
    headers = {"User-Agent": "wildfire-risk-model/0.1"}
    out = {}
    for layer, url in endpoints.items():
        mat = []
        for i in range(-grid_radius, grid_radius + 1):
            row = []
            for j in range(-grid_radius, grid_radius + 1):
                params = {
                    "geometry": json.dumps({"x": lon + j*dlon, "y": lat + i*dlat,
                                            "spatialReference": {"wkid": 4326}}),
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

LAT, LON = CONFIG["parcel"]["lat"], CONFIG["parcel"]["lon"]
endpoints = {
    "BP":    CONFIG["fsim"]["bp_endpoint"],
    "FLEP4": CONFIG["fsim"]["flep4_endpoint"],
    "FLEP8": CONFIG["fsim"]["flep8_endpoint"],
}
print(f"Fetching FSim grid at ({LAT}, {LON})...")
grid_raw = fetch_fsim_grid(LAT, LON, endpoints, CONFIG["fsim"]["grid_radius_cells"])
scales = CONFIG["fsim"]["scaling"]
grid_scaled = {
    layer: [[(v * scales[layer.lower()]) if v is not None else None for v in row] for row in mat]
    for layer, mat in grid_raw.items()
}

for layer, mat in grid_scaled.items():
    print(f"\n--- {layer} (scaled probabilities) ---")
    for row in mat: print("  " + "  ".join(f"{v:.5f}" if v is not None else "   NaN" for v in row))

# Persist to bronze
fsim_rows = []
NOW_TS = datetime.now(timezone.utc)
for layer, mat in grid_scaled.items():
    for i, row in enumerate(mat):
        for j, val in enumerate(row):
            if val is None: continue
            raw_val = grid_raw[layer][i][j]
            fsim_rows.append({
                "parcel_lat":   LAT,
                "parcel_lon":   LON,
                "address":      CONFIG["parcel"]["address"],
                "layer":        layer,
                "cell_row":     i - 2,
                "cell_col":     j - 2,
                "raw_value":    raw_val,
                "scaled_value": val,
                "scale_factor": int(1/scales[layer.lower()]),
                "source_url":   endpoints[layer],
                "retrieved_at": NOW_TS,
                "notes":        "burnable" if not (layer.startswith("FLEP") and raw_val == 0) else "developed/non-burnable",
            })
spark.createDataFrame(fsim_rows).write.mode("overwrite").saveAsTable(
    f"{CONFIG['uc']['catalog']}.{CONFIG['uc']['bronze_fsim']}"
)
print(f"\nWrote {len(fsim_rows)} rows → bronze.fsim_parcel_sample")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Aggregate FSim → model inputs

# COMMAND ----------
bp_samples     = [v for row in grid_scaled["BP"]    for v in row if v is not None]
flep8_burnable = [v for row in grid_scaled["FLEP8"] for v in row if v not in (None, 0.0)]
flep4_burnable = [v for row in grid_scaled["FLEP4"] for v in row if v not in (None, 0.0)]
print(f"BP    n={len(bp_samples)}     min={min(bp_samples):.5f}  med={np.median(bp_samples):.5f}  max={max(bp_samples):.5f}")
print(f"FLEP8 n={len(flep8_burnable)} (burnable)  max={max(flep8_burnable) if flep8_burnable else 0:.4f}")
print(f"FLEP4 n={len(flep4_burnable)} (burnable)  max={max(flep4_burnable) if flep4_burnable else 0:.4f}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Pull CAL FIRE DINS (Phase 2 foundation)
# MAGIC Paginated FeatureServer query → ~130K row-level structure outcomes since 2013.
# MAGIC Re-uses cached bronze if available.

# COMMAND ----------
def fsim_load_dins():
    """Return DINS as a Spark DataFrame, refreshing bronze cache if missing."""
    full_table = f"{CONFIG['uc']['catalog']}.{CONFIG['uc']['bronze_dins']}"
    try:
        df = spark.table(full_table)
        n = df.count()
        if n > 100_000:
            print(f"Loaded {n:,} cached DINS rows from {full_table}")
            return df
    except Exception:
        pass

    print(f"Pulling DINS from {CONFIG['dins_feature_server']}...")
    base = CONFIG["dins_feature_server"]
    headers = {"User-Agent": "wildfire-risk-model/0.1"}
    total = requests.get(f"{base}/query", params={"where":"1=1","returnCountOnly":"true","f":"json"},
                         headers=headers, timeout=30).json().get("count", 0)
    print(f"  Server reports {total:,} total records")
    page, offset, rows = 2000, 0, []
    while offset < total:
        params = {"where":"1=1","outFields":"*","f":"json","resultOffset":offset,
                  "resultRecordCount":page,"orderByFields":"OBJECTID","returnGeometry":"false"}
        d = requests.get(f"{base}/query", params=params, headers=headers, timeout=60).json()
        feats = d.get("features", [])
        if not feats: break
        rows.extend(ft.get("attributes", {}) for ft in feats)
        offset += len(feats)
        if offset % 20000 == 0:
            print(f"  pulled {offset:,}/{total:,} ({offset/total:.0%})")
    df = spark.createDataFrame(pd.DataFrame(rows))
    df.write.mode("overwrite").option("mergeSchema","true").saveAsTable(full_table)
    print(f"Wrote {len(rows):,} rows → {full_table}")
    return spark.table(full_table)

dins = fsim_load_dins()
print(f"\nDINS schema (first 15 cols): {dins.columns[:15]}")
dins.groupBy("DAMAGE").count().orderBy("count", ascending=False).show(truncate=False)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. Phase 2 — DINS-fitted vulnerability multiplier
# MAGIC Logistic regression on the Multiple-Residence subset (target = `DAMAGE = "Destroyed (>50%)"`),
# MAGIC with missing-as-category encoding. Predicts the parcel's multiplier under 4 hardening scenarios.

# COMMAND ----------
mr = dins.filter("STRUCTURECATEGORY = 'Multiple Residence'").toPandas()
print(f"Multi-residence DINS subset: n={len(mr)}")

def cat_value(v, mapping, default="missing"):
    if v is None or str(v).strip() in ("", "Unknown", "None", "N/A"): return default
    return mapping.get(str(v).strip(), "other")

def year_bucket(yr):
    try:    yr = int(yr)
    except: return "missing"
    if   yr <  1960: return "pre1960"
    elif yr <  1980: return "1960s_70s"
    elif yr <  1990: return "1980s"     # ← parcel
    elif yr <  2000: return "1990s"
    else:             return "2000plus"

def encode(row):
    deck = "vulnerable" if any(
        (s and "Composite" not in str(s) and "Concrete" not in str(s) and "No Deck" not in str(s))
        for s in [row.get("DECKPORCHELEVATED"), row.get("DECKPORCHONGRADE")]) else "hardened"
    return {
        "year_bucket": year_bucket(row.get("YEARBUILT")),
        "roof":        cat_value(row.get("ROOFCONSTRUCTION"),
                                  {"Asphalt":"vulnerable","Wood":"vulnerable","Combustible":"vulnerable",
                                   "Tile":"hardened","Metal":"hardened","Concrete":"hardened"}),
        "vent":        cat_value(row.get("VENTSCREEN"),
                                  {"Mesh Screen <= 1/8\"":"hardened","No Vents":"hardened",
                                   "Mesh Screen > 1/8\"":"vulnerable","Unscreened":"vulnerable"}),
        "eaves":       cat_value(row.get("EAVES"),
                                  {"Enclosed":"hardened","No Eaves":"hardened",
                                   "Unenclosed":"vulnerable","Open":"vulnerable"}),
        "deck":        deck,
        "defended":    cat_value(row.get("DEFENSIVEACTIONS"),
                                  {"Engine Company Actions":"yes","Fire Department":"yes",
                                   "Combination of Actions":"yes","Hand Crew":"yes","None":"no"}),
        "county":      cat_value(row.get("COUNTY"), {}, default="other"),
    }

X_cat = [encode(r) for r in mr.to_dict(orient="records")]
y     = np.array([1 if d == "Destroyed (>50%)" else 0 for d in mr["DAMAGE"]])
feat_names = ["year_bucket","roof","vent","eaves","deck","defended","county"]

# One-hot
levels = {f: sorted({x[f] for x in X_cat}) for f in feat_names}
col_names = [f"{f}={lv}" for f in feat_names for lv in levels[f][1:]]
def design(X_cat_list):
    rows = []
    for x in X_cat_list:
        row = []
        for f in feat_names:
            for lv in levels[f][1:]:
                row.append(1.0 if x[f] == lv else 0.0)
        rows.append(row)
    return np.array(rows, dtype=float)

X = design(X_cat)
X_int = np.column_stack([np.ones(len(X)), X])

def sigmoid(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
def fit_logreg(X, y, lr=0.05, l2=2.0, n_iter=15000, tol=1e-8):
    n, p = X.shape
    w = np.zeros(p)
    for _ in range(n_iter):
        ph = sigmoid(X @ w)
        grad = X.T @ (ph - y) / n
        grad[1:] += l2 * w[1:] / n
        w_new = w - lr * grad
        if np.max(np.abs(w_new - w)) < tol: return w_new
        w = w_new
    return w

w_logreg = fit_logreg(X_int, y)
# AUC
ph = sigmoid(X_int @ w_logreg)
order = np.argsort(-ph)
y_sorted = y[order]
tp = np.cumsum(y_sorted); fp = np.cumsum(1 - y_sorted)
trapz = getattr(np, "trapezoid", np.trapz)
auc = float(trapz(tp/tp[-1], fp/fp[-1]))
print(f"LogReg AUC: {auc:.3f}  (training-set)")

# Parcel profiles
parcel_county = "other"
parcel_profiles = {
    "best_realized":  {"year_bucket":"1980s","roof":"hardened","vent":"hardened","eaves":"hardened","deck":"hardened","defended":"yes","county":parcel_county},
    "expected":       {"year_bucket":"1980s","roof":"vulnerable","vent":"hardened","eaves":"hardened","deck":"hardened","defended":"yes","county":parcel_county},
    "degraded":       {"year_bucket":"1980s","roof":"vulnerable","vent":"vulnerable","eaves":"vulnerable","deck":"vulnerable","defended":"yes","county":parcel_county},
    "worst":          {"year_bucket":"1980s","roof":"vulnerable","vent":"vulnerable","eaves":"vulnerable","deck":"vulnerable","defended":"no","county":parcel_county},
}
def score(profile):
    row = [1.0]
    for f in feat_names:
        for lv in levels[f][1:]:
            row.append(1.0 if profile.get(f) == lv else 0.0)
    return float(sigmoid(np.array(row) @ w_logreg))

print("\nParcel profile predictions (P(destroyed | exposed)):")
fitted_mults = {}
for name, prof in parcel_profiles.items():
    fitted_mults[name] = score(prof)
    print(f"  {name:18s}  {fitted_mults[name]:.3f}")

# Build scenario PMF from fitted multipliers
P2_SCENARIOS = {
    "best_realized": {"prob": 0.20, "mult": fitted_mults["best_realized"]},
    "expected":      {"prob": 0.50, "mult": fitted_mults["expected"]},
    "degraded":      {"prob": 0.25, "mult": fitted_mults["degraded"]},
    "worst":         {"prob": 0.05, "mult": fitted_mults["worst"]},
}
vuln_mults_arr = np.array([s["mult"] for s in P2_SCENARIOS.values()])
vuln_probs_arr = np.array([s["prob"] for s in P2_SCENARIOS.values()])
vuln_probs_arr = vuln_probs_arr / vuln_probs_arr.sum()
print(f"\nE[vuln_mult] (Phase 2 fitted) = {(vuln_mults_arr*vuln_probs_arr).sum():.3f}  (Phase 1 was 0.568)")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 8. Phase 2.5 — Bayesian hierarchical multiplier (PyMC)
# MAGIC Partial pooling across CA counties; the parcel transfer to NV uses the **population-level intercept** (counties this parcel isn't in are absorbed by the hyper-prior).

# COMMAND ----------
import pymc as pm
import arviz as az

# Encode for PyMC — keep only structurally important features for the Bayesian run
# (county random intercept + fixed effects on roof/vent/defended/year)
counties = sorted({x["county"] for x in X_cat})
county_idx = np.array([counties.index(x["county"]) for x in X_cat])
n_counties = len(counties)
print(f"Counties: n={n_counties}")

# Fixed-effect features (5 binary indicators)
x_roof_vuln = np.array([1 if x["roof"] == "vulnerable" else 0 for x in X_cat])
x_vent_vuln = np.array([1 if x["vent"] == "vulnerable" else 0 for x in X_cat])
x_eaves_vuln = np.array([1 if x["eaves"] == "vulnerable" else 0 for x in X_cat])
x_defended  = np.array([1 if x["defended"] == "yes" else 0 for x in X_cat])
x_yr1980s   = np.array([1 if x["year_bucket"] == "1980s" else 0 for x in X_cat])
X_be = np.column_stack([x_roof_vuln, x_vent_vuln, x_eaves_vuln, x_defended, x_yr1980s])

with pm.Model() as bayes_model:
    # Hierarchical prior on county intercepts
    mu_alpha    = pm.Normal("mu_alpha",    mu=-1.0, sigma=1.0)
    sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1.0)
    alpha_c     = pm.Normal("alpha_c", mu=mu_alpha, sigma=sigma_alpha, shape=n_counties)
    # Fixed-effect betas
    beta = pm.Normal("beta", mu=0.0, sigma=1.0, shape=X_be.shape[1])
    # Linear predictor
    eta = alpha_c[county_idx] + pm.math.dot(X_be, beta)
    p_dest = pm.Deterministic("p_dest", pm.math.sigmoid(eta))
    # Likelihood
    y_obs = pm.Bernoulli("y_obs", p=p_dest, observed=y)

    idata = pm.sample(draws=1000, tune=1000, chains=2, target_accept=0.9, random_seed=SEED, progressbar=False)

print(az.summary(idata, var_names=["mu_alpha","sigma_alpha","beta"]).to_string())

# Posterior predictive for parcel (NV — use population intercept mu_alpha)
mu_alpha_post = idata.posterior["mu_alpha"].values.flatten()
beta_post     = idata.posterior["beta"].values.reshape(-1, X_be.shape[1])  # (samples, 5)

# Build the parcel feature vector for each scenario
parcel_features = {
    # roof_vuln, vent_vuln, eaves_vuln, defended, yr1980s
    "best_realized": np.array([0, 0, 0, 1, 1]),
    "expected":      np.array([1, 0, 0, 1, 1]),
    "degraded":      np.array([1, 1, 1, 1, 1]),
    "worst":         np.array([1, 1, 1, 0, 1]),
}
print("\nBayesian posterior on multiplier (CA→NV via population intercept):")
bayes_mults = {}
for name, fv in parcel_features.items():
    eta_samples = mu_alpha_post + beta_post @ fv
    p_samples = 1/(1+np.exp(-eta_samples))
    bayes_mults[name] = {
        "median": float(np.median(p_samples)),
        "ci_low":  float(np.percentile(p_samples, 2.5)),
        "ci_high": float(np.percentile(p_samples, 97.5)),
        "samples": p_samples,
    }
    print(f"  {name:18s}  median={bayes_mults[name]['median']:.3f}  95% CI [{bayes_mults[name]['ci_low']:.3f}, {bayes_mults[name]['ci_high']:.3f}]")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 9. Phase 3 — Cal-Adapt climate-driven hazard trajectory

# COMMAND ----------
def fetch_caladapt():
    g = f"POINT({LON} {LAT})"
    series = {}
    for model in CONFIG["climate"]["cal_adapt_models"]:
        for scen in CONFIG["climate"]["cal_adapt_scenarios"]:
            slug = f"tasmax_year_{model}_{scen}"
            try:
                r = requests.get(f"{CONFIG['cal_adapt_api']}/{slug}/events/",
                                 params={"g": g, "format": "json"},
                                 headers={"User-Agent": "wildfire-risk-model/0.1"},
                                 timeout=30)
                d = r.json()
                years = [int(t[:4]) for t in d.get("index", [])]
                tasmax_c = [v - 273.15 for v in d.get("data", [])]
                series[f"{model}_{scen}"] = {"years": years, "tasmax_c": tasmax_c}
                time.sleep(0.05)
            except Exception as e:
                print(f"  {slug} failed: {e}")
    return series

clim_series = fetch_caladapt()
print(f"Pulled {len(clim_series)} Cal-Adapt series")

# Build ensemble climate-driven multiplier
ensemble = {}
START_YEAR = CONFIG["horizon_start_year"]
for name, s in clim_series.items():
    yrs = s["years"]; tmax = s["tasmax_c"]
    base = np.mean([t for t, y in zip(tmax, yrs) if 2006 <= y <= 2025])
    anomaly = {y: t - base for y, t in zip(yrs, tmax) if START_YEAR <= y <= START_YEAR + 25}
    ensemble[name] = {"baseline_c": base, "anomaly_by_year": anomaly}

ensemble_names = list(ensemble.keys())
BETA_LO  = CONFIG["climate"]["beta_per_C_lo"]
BETA_MID = CONFIG["climate"]["beta_per_C_central"]
BETA_HI  = CONFIG["climate"]["beta_per_C_hi"]

def f_climate(year, m, beta):
    return math.exp(beta * m["anomaly_by_year"].get(year, 0.0))

# Persist climate ensemble to bronze
clim_rows = []
for name, s in clim_series.items():
    for y, t in zip(s["years"], s["tasmax_c"]):
        clim_rows.append({"parcel_lat": LAT, "parcel_lon": LON, "series": name,
                          "year": y, "tasmax_c": t, "retrieved_at": NOW_TS})
spark.createDataFrame(clim_rows).write.mode("overwrite").saveAsTable(
    f"{CONFIG['uc']['catalog']}.{CONFIG['uc']['bronze_climate']}"
)
print(f"Wrote {len(clim_rows)} climate rows → bronze.caladapt_tasmax")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 10. Hazard sampler + Monte Carlo

# COMMAND ----------
def sample_interior(rng):
    n_units = rng.integers(CONFIG["interior_ignition"]["n_units_lo"], CONFIG["interior_ignition"]["n_units_hi"])
    per_unit_rate = rng.lognormal(math.log(CONFIG["interior_ignition"]["per_unit_fire_rate_per_yr_central"]),
                                  CONFIG["interior_ignition"]["per_unit_fire_rate_sigma_log"])
    a, b = CONFIG["interior_ignition"]["p_shell_loss_given_fire_beta"]
    p_shell_loss = rng.beta(a, b)
    p_unit_total = per_unit_rate * p_shell_loss
    return 1.0 - (1.0 - p_unit_total) ** n_units

def sample_hazard_components(rng, use_bayes=True):
    bp     = rng.choice(bp_samples)
    flep8  = rng.choice(flep8_burnable)
    flep4  = rng.choice(flep4_burnable)
    if use_bayes:
        # Sample from Bayesian posterior — pick scenario by prior, then draw a multiplier sample
        sc_names = list(P2_SCENARIOS.keys())
        scen_idx = rng.choice(len(sc_names), p=[P2_SCENARIOS[k]["prob"] for k in sc_names])
        scen = sc_names[scen_idx]
        vuln_h = float(rng.choice(bayes_mults[scen]["samples"]))
    else:
        vuln_h = rng.choice(vuln_mults_arr, p=vuln_probs_arr)
    vuln_l = vuln_h * 0.4
    ember = rng.lognormal(math.log(CONFIG["ember_baseline"]["median"]),
                          CONFIG["ember_baseline"]["sigma_log"]) * vuln_h
    interior = sample_interior(rng)
    p_destroy_given_fire = flep8 * vuln_h + max(0.0, flep4 - flep8) * vuln_l
    return {
        "h0_wildfire":  bp * p_destroy_given_fire + ember,
        "h0_interior":  interior,
        "member":       ensemble[rng.choice(ensemble_names)],
        "beta":         rng.triangular(BETA_LO, BETA_MID, BETA_HI),
    }

def cum_loss(s, n_years):
    H = 0.0
    for k in range(n_years):
        y = START_YEAR + k
        H += s["h0_wildfire"] * f_climate(y, s["member"], s["beta"]) + s["h0_interior"]
    return 1.0 - math.exp(-H)

def run_mc(use_bayes, n_iters):
    rng = np.random.default_rng(SEED)
    horizons = CONFIG["horizons_years"]
    pcum   = {h: np.zeros(n_iters) for h in horizons}
    dollar = {h: np.zeros(n_iters, dtype=int) for h in horizons}
    decomp = {"h0_wildfire": np.zeros(n_iters), "h0_interior": np.zeros(n_iters)}
    for i in range(n_iters):
        s = sample_hazard_components(rng, use_bayes=use_bayes)
        decomp["h0_wildfire"][i] = s["h0_wildfire"]
        decomp["h0_interior"][i] = s["h0_interior"]
        for h in horizons:
            pc = cum_loss(s, h)
            pcum[h][i] = pc
            dollar[h][i] = CONFIG["financials"]["shell_basis_usd"] if rng.random() < pc else 0
    return pcum, dollar, decomp

print("Running Phase 3 MC (with Bayesian posterior on vulnerability)...")
pcum_p3, dollar_p3, decomp_p3 = run_mc(use_bayes=True,  n_iters=CONFIG["monte_carlo"]["iters"])
print("Done.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 11. Decision table

# COMMAND ----------
SHELL = CONFIG["financials"]["shell_basis_usd"]
PREM  = CONFIG["financials"]["premium_saved_annual_usd"]

rows = []
for h in CONFIG["horizons_years"]:
    pl, d = pcum_p3[h], dollar_p3[h]
    e_loss = float(d.mean()); p95 = float(np.percentile(d,95)); p99 = float(np.percentile(d,99))
    cum_p = PREM * h
    if   e_loss * 3 < cum_p and p99 < cum_p:        verdict = "Self-insure"
    elif e_loss > cum_p:                             verdict = "INSURE (E[loss] > premium)"
    elif p99 >= SHELL and p99 > cum_p:               verdict = "Borderline — tail hits"
    else:                                             verdict = "Self-insure (modest tail)"
    rows.append({
        "horizon_yrs":    h,
        "p_loss_median":  float(np.median(pl)),
        "p_loss_p5":      float(np.percentile(pl,5)),
        "p_loss_p95":     float(np.percentile(pl,95)),
        "e_loss_usd":     e_loss, "p95_loss_usd": p95, "p99_loss_usd": p99,
        "cum_premium_usd": cum_p, "verdict": verdict,
        "parcel_address": CONFIG["parcel"]["address"],
        "phase":          "v2_bayesian_climate",
        "run_timestamp":  NOW_TS,
    })
decision_df = pd.DataFrame(rows)
print(decision_df[["horizon_yrs","p_loss_median","p_loss_p5","p_loss_p95",
                    "e_loss_usd","p99_loss_usd","cum_premium_usd","verdict"]].to_string(index=False))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 12. Plots — survival, decomp, break-even, sensitivity

# COMMAND ----------
plt.style.use('seaborn-v0_8-whitegrid')

# Plot 1: Survival curves with credible band
years_plot = np.arange(0, 21)
rng_plot = np.random.default_rng(SEED + 99)
N_PLOT = 1000
idx_sample = rng_plot.choice(CONFIG["monte_carlo"]["iters"], N_PLOT, replace=False)
mat = np.zeros((N_PLOT, len(years_plot)))
for k, i in enumerate(idx_sample):
    s_full = {"h0_wildfire": decomp_p3["h0_wildfire"][i], "h0_interior": decomp_p3["h0_interior"][i],
              "member": ensemble[ensemble_names[i % len(ensemble_names)]], "beta": BETA_MID}
    H_cum, yrs_seen = 0.0, 0
    for j, t in enumerate(years_plot):
        while yrs_seen < int(t):
            y = START_YEAR + yrs_seen
            H_cum += s_full["h0_wildfire"] * f_climate(y, s_full["member"], s_full["beta"]) + s_full["h0_interior"]
            yrs_seen += 1
        mat[k,j] = 1 - math.exp(-H_cum)
med = np.median(mat, axis=0); p5 = np.percentile(mat,5,axis=0); p95 = np.percentile(mat,95,axis=0)

fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(years_plot, med, color="#d62728", linewidth=2, label="Median (Bayesian + climate-driven)")
ax.fill_between(years_plot, p5, p95, color="#d62728", alpha=0.20, label="5–95% credible band")
ax.set_xlabel("Years held (2026 → 2041)")
ax.set_ylabel("Cumulative P(total structural loss)")
ax.set_title(f"Wildfire total-loss probability — {CONFIG['parcel']['address']}\n(Bayesian DINS vulnerability + Cal-Adapt climate hazard)")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.legend(loc="upper left")
for h in CONFIG["horizons_years"]: ax.axvline(h, linestyle=":", color="gray", alpha=0.5)
plt.tight_layout()
plt.show()

# Plot 2: Hazard decomposition (log scale)
fig, ax = plt.subplots(figsize=(10, 6))
labels  = ["Wildfire\n(BP × FLEP × vuln + ember)", "Interior ignition\n(unit fires)"]
medians = [float(np.median(decomp_p3["h0_wildfire"])), float(np.median(decomp_p3["h0_interior"]))]
p95s    = [float(np.percentile(decomp_p3["h0_wildfire"],95)), float(np.percentile(decomp_p3["h0_interior"],95))]
xpos = np.arange(2)
ax.bar(xpos - 0.2, medians, 0.4, label="Median", color="#1f77b4")
ax.bar(xpos + 0.2, p95s, 0.4, label="P95", color="#d62728")
ax.set_xticks(xpos); ax.set_xticklabels(labels)
ax.set_ylabel("Annual P(structural loss) by mechanism")
ax.set_title("Hazard decomposition — interior ignition dominates")
ax.set_yscale("log")
ax.legend()
for i, (m, p) in enumerate(zip(medians, p95s)):
    ax.text(i - 0.2, m, f"{m:.1e}", ha='center', va='bottom', fontsize=9)
    ax.text(i + 0.2, p, f"{p:.1e}", ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.show()

# Plot 3: $ exposure break-even
horizons = CONFIG["horizons_years"]
fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(horizons))
w = 0.18
ax.bar(x - 1.5*w, [PREM*h for h in horizons], w, label="Cumulative premium", color="#9467bd")
ax.bar(x - 0.5*w, [dollar_p3[h].mean() for h in horizons], w, label="E[$ loss]", color="#2ca02c")
ax.bar(x + 0.5*w, [np.percentile(dollar_p3[h],95) for h in horizons], w, label="P95 $ loss", color="#ff7f0e")
ax.bar(x + 1.5*w, [np.percentile(dollar_p3[h],99) for h in horizons], w, label="P99 $ loss", color="#d62728")
ax.axhline(SHELL, linestyle="--", color="black", alpha=0.5, label="Shell ($334K)")
ax.set_xticks(x); ax.set_xticklabels([f"{h} yrs" for h in horizons])
ax.set_ylabel("USD"); ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"${v/1000:.0f}K"))
ax.set_title("Premium vs. expected & tail loss across horizons")
ax.legend(loc="upper left", fontsize=9)
plt.tight_layout()
plt.show()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 13. Persist gold outputs

# COMMAND ----------
# Decision table → gold
spark.createDataFrame(decision_df).write.mode("append").saveAsTable(
    f"{CONFIG['uc']['catalog']}.{CONFIG['uc']['gold_decision']}"
)
print(f"Appended {len(decision_df)} rows → gold.decision_table_phase2")

# MC dollar distribution → gold (long format)
mc_long = []
for h in CONFIG["horizons_years"]:
    for i, dval in enumerate(dollar_p3[h]):
        mc_long.append({"horizon_yrs": h, "iter": i, "dollar_loss": int(dval),
                        "pcum_rise": float(pcum_p3[h][i]),
                        "parcel_address": CONFIG["parcel"]["address"],
                        "run_timestamp": NOW_TS})
spark.createDataFrame(pd.DataFrame(mc_long)).write.mode("overwrite").saveAsTable(
    f"{CONFIG['uc']['catalog']}.{CONFIG['uc']['gold_mc']}"
)
print(f"Wrote {len(mc_long):,} rows → gold.mc_dollar_distribution")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 14. Assumptions & caveats log
# MAGIC
# MAGIC | # | Assumption | Direction of bias |
# MAGIC |---|---|---|
# MAGIC | 1 | **CA → NV DINS transfer** via Bayesian population-level intercept | Less biased than Phase 2 (uses hierarchical mu_alpha rather than a fixed "other-county" category) but still assumes CA fire physics applies to NV |
# MAGIC | 2 | **FSim 270 m resolution** vs. single parcel | Smooths local fuel discontinuities; mitigated by 5×5 grid sampling |
# MAGIC | 3 | **WRC 2020 LANDFIRE fuels** (pre-Caldor) | Likely *underestimates* current hazard; Caldor 2021 burned within 20 mi of this parcel |
# MAGIC | 4 | **Climate trend only modulates wildfire**, not interior ignition | Defensible — interior fires are weakly climate-coupled at best |
# MAGIC | 5 | **Sierra fire-Tmax elasticity β = 0.5–1.0 per °C** (Westerling 2018, Goss 2020) | Spans the published range |
# MAGIC | 6 | **NFPA national fire rate × N units uniform[4,16]** for interior | N_units unknown; sensitivity-tested |
# MAGIC | 7 | **Binary loss model** (total or zero, no partials) | Spec-aligned for self-insurance question |
# MAGIC | 8 | **Premium static at $24K/yr** | Real WUI premiums rising ~10–25%/yr; conservative for the no-insure scenario |

# COMMAND ----------
# MAGIC %md
# MAGIC ## 15. Headline read
# MAGIC The decision table at horizon = 15 yrs is the row to anchor the self-insurance vote on. Compare:
# MAGIC `cumulative premium ($360K)` vs `E[loss]` (≈ $11–14K) vs `P99 loss` ($334K).
# MAGIC If P99 ≪ cumulative premium **and** the household can absorb that P99 loss, self-insure.
# MAGIC If P99 ≈ shell value and the household cannot absorb that loss, the EV-rational choice is to insure
# MAGIC anyway — that's the canonical tail-risk justification for insurance.
