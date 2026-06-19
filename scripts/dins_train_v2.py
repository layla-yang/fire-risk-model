#!/usr/bin/env python3
"""Phase 2 vulnerability multiplier from CAL FIRE DINS.

Two complementary approaches:
  A) Empirical look-up: parcel's cohort-specific destruction rate from cross-tabs
  B) Logistic regression with missing-as-category (full multi-res sample, n=2007)
"""
import json
import numpy as np

# np.trapz removed in numpy 2.0
trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")

with open("/tmp/dins_multires_chunk0.json") as f:
    raw = json.load(f)
cols = raw["columns"]
rows = raw["rows"]
print(f"Multi-residence DINS sample: n={len(rows)}")

def cidx(name): return cols.index(name)

# Target: destroyed (>50%) = "total loss" event
y = np.array([1 if r[cidx("DAMAGE")] == "Destroyed (>50%)" else 0 for r in rows])

# Categorical encoder — treat None / Unknown / empty as a category ("missing")
def cat(value, mapping, default="missing"):
    if value is None or str(value).strip() in ("", "Unknown", "None", "N/A"):
        return default
    s = str(value).strip()
    return mapping.get(s, "other")

# Define features with explicit category groups
def encode_row(r):
    yr = r[cidx("YEARBUILT")]
    try:
        yr_int = int(yr) if yr is not None else None
    except (ValueError, TypeError):
        yr_int = None

    yb_cat = "missing"
    if yr_int is not None:
        if   yr_int <  1960: yb_cat = "pre1960"
        elif yr_int <  1980: yb_cat = "1960s_70s"
        elif yr_int <  1990: yb_cat = "1980s"     # ← parcel
        elif yr_int <  2000: yb_cat = "1990s"
        else:                yb_cat = "2000plus"

    return {
        "year_bucket": yb_cat,
        "roof": cat(r[cidx("ROOFCONSTRUCTION")], {
            "Asphalt":"vulnerable","Wood":"vulnerable","Combustible":"vulnerable",
            "Tile":"hardened","Metal":"hardened","Concrete":"hardened"}),
        "vent": cat(r[cidx("VENTSCREEN")], {
            "Mesh Screen <= 1/8\"":"hardened","No Vents":"hardened",
            "Mesh Screen > 1/8\"":"vulnerable","Unscreened":"vulnerable"}),
        "eaves": cat(r[cidx("EAVES")], {
            "Enclosed":"hardened","No Eaves":"hardened",
            "Unenclosed":"vulnerable","Open":"vulnerable"}),
        "deck": "vulnerable" if any(
            (s and "Composite" not in str(s) and "Concrete" not in str(s) and "No Deck" not in str(s))
            for s in [r[cidx("DECKPORCHELEVATED")], r[cidx("DECKPORCHONGRADE")]]
        ) else "hardened",
        "defended": cat(r[cidx("DEFENSIVEACTIONS")], {
            "Engine Company Actions":"yes","Fire Department":"yes",
            "Combination of Actions":"yes","Hand Crew":"yes",
            "None":"no"}),
        "county": cat(r[cidx("COUNTY")], {}, default="other"),
    }

X_cat = [encode_row(r) for r in rows]
feat_names = ["year_bucket", "roof", "vent", "eaves", "deck", "defended", "county"]

# === APPROACH A: Empirical cross-tab look-up ===
print("\n" + "=" * 75)
print("APPROACH A — Empirical destruction rate by cohort")
print("=" * 75)

def slice_rate(filter_fn, label):
    sel = np.array([filter_fn(x) for x in X_cat])
    if sel.sum() == 0:
        print(f"  {label:55s}  n=0")
        return None
    rate = y[sel].mean()
    se   = np.sqrt(rate * (1 - rate) / sel.sum())
    print(f"  {label:55s}  n={sel.sum():>5,}  rate={rate:.3f}  ±{1.96*se:.3f} (95%)")
    return rate

slice_rate(lambda x: True,                                           "All multi-res")
slice_rate(lambda x: x["year_bucket"] == "1980s",                    "1980s cohort")
slice_rate(lambda x: x["year_bucket"] in ("1960s_70s","1980s"),      "1960-89 cohort")
slice_rate(lambda x: x["year_bucket"] == "1980s" and x["roof"] == "vulnerable",  "1980s × Asphalt/Wood roof")
slice_rate(lambda x: x["year_bucket"] == "1980s" and x["roof"] == "hardened",    "1980s × Tile/Metal roof")
slice_rate(lambda x: x["year_bucket"] == "1980s" and x["defended"] == "yes",     "1980s × Defended by engine co")
slice_rate(lambda x: x["year_bucket"] == "1980s" and x["vent"] == "hardened",    "1980s × Ember-resistant vents")
slice_rate(lambda x: x["year_bucket"] == "1980s" and x["deck"] == "vulnerable",  "1980s × Combustible deck")

print("\n  Parcel-like cohorts (1980 build, multi-unit):")
parcel_anchor_rates = {}
# "Original 1980 attributes, no retrofit, no defense"
r1 = slice_rate(lambda x: x["year_bucket"] in ("1960s_70s","1980s")
                          and x["roof"] == "vulnerable"
                          and x["vent"] == "vulnerable",
                "Original 1980-era — asphalt roof + >1/8 vent")
parcel_anchor_rates["original_undefended"] = r1

# "Original 1980 + defensive intervention"
r2 = slice_rate(lambda x: x["year_bucket"] in ("1960s_70s","1980s")
                          and x["roof"] == "vulnerable"
                          and x["defended"] == "yes",
                "Original 1980-era + engine company defense")
parcel_anchor_rates["original_defended"] = r2

# "Partial retrofit — vents + eaves hardened, defended"
r3 = slice_rate(lambda x: x["year_bucket"] in ("1960s_70s","1980s")
                          and x["vent"] == "hardened"
                          and x["eaves"] == "hardened",
                "Original 1980-era + retrofit vents + retrofit eaves")
parcel_anchor_rates["partial_retrofit"] = r3

# === APPROACH B: Logistic regression with missing-as-category ===
print("\n" + "=" * 75)
print("APPROACH B — Logistic regression (one-hot, missing-as-category)")
print("=" * 75)

# Build one-hot encoding
def build_design(X_cat):
    levels = {}
    for f in feat_names:
        levels[f] = sorted({x[f] for x in X_cat})
    rows_oh = []
    col_names = []
    for f in feat_names:
        for lv in levels[f][1:]:  # drop first level as reference
            col_names.append(f"{f}={lv}")
    for x in X_cat:
        row = []
        for f in feat_names:
            for lv in levels[f][1:]:
                row.append(1.0 if x[f] == lv else 0.0)
        rows_oh.append(row)
    return np.array(rows_oh, dtype=float), col_names, levels

X_design, col_names, levels = build_design(X_cat)
X_int = np.column_stack([np.ones(len(X_design)), X_design])
print(f"  Design matrix: {X_int.shape},  features: {len(col_names)}")

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

def fit_logreg(X, y, lr=0.05, l2=2.0, n_iter=20000, tol=1e-8):
    n, p = X.shape
    w = np.zeros(p)
    for it in range(n_iter):
        z = X @ w
        ph = sigmoid(z)
        grad = X.T @ (ph - y) / n
        grad[1:] += l2 * w[1:] / n  # don't regularize intercept
        w_new = w - lr * grad
        if np.max(np.abs(w_new - w)) < tol:
            return w_new, it+1
        w = w_new
    return w, n_iter

w, n_iter = fit_logreg(X_int, y, lr=0.05, l2=2.0, n_iter=15000)
print(f"  Converged in {n_iter} iters")

# AUC
ph = sigmoid(X_int @ w)
order = np.argsort(-ph)
y_sorted = y[order]
tp = np.cumsum(y_sorted); fp = np.cumsum(1 - y_sorted)
auc = trapz(tp / tp[-1], fp / fp[-1])
print(f"  Training AUC: {auc:.3f}  (50% = random; multi-res sample is noisy)")

# Top 10 strongest coefficients (excluding intercept)
print(f"\n  Top 10 strongest signals (log-odds):")
abs_w = np.abs(w[1:])
top_idx = np.argsort(-abs_w)[:10]
for i in top_idx:
    print(f"    {col_names[i]:35s}  {w[i+1]:+.3f}  → OR {np.exp(w[i+1]):.2f}")

# Score the parcel — multiple scenarios
def score_profile(profile):
    """profile dict like {'year_bucket': '1980s', 'roof': 'vulnerable', ...}"""
    row = [1.0]  # intercept
    for f in feat_names:
        for lv in levels[f][1:]:
            row.append(1.0 if profile.get(f) == lv else 0.0)
    z = np.array(row) @ w
    return sigmoid(z)

# Pick a default county (most common in 1980s cohort) — for parcel transfer to NV
from collections import Counter
common_county = Counter([x["county"] for x in X_cat if x["year_bucket"] == "1980s"]).most_common(3)
print(f"\n  Most common counties in 1980s cohort: {common_county}")
parcel_county = "other"  # NV transfer

parcel_profiles_B = {
    "original_1980_undefended": {
        "year_bucket":"1980s","roof":"vulnerable","vent":"vulnerable","eaves":"vulnerable",
        "deck":"vulnerable","defended":"no","county":parcel_county
    },
    "original_1980_defended": {
        "year_bucket":"1980s","roof":"vulnerable","vent":"vulnerable","eaves":"vulnerable",
        "deck":"vulnerable","defended":"yes","county":parcel_county
    },
    "partial_retrofit_defended": {
        "year_bucket":"1980s","roof":"vulnerable","vent":"hardened","eaves":"hardened",
        "deck":"hardened","defended":"yes","county":parcel_county
    },
    "full_retrofit_defended": {
        "year_bucket":"1980s","roof":"hardened","vent":"hardened","eaves":"hardened",
        "deck":"hardened","defended":"yes","county":parcel_county
    },
    "lapsed_no_defense": {
        "year_bucket":"1980s","roof":"vulnerable","vent":"vulnerable","eaves":"vulnerable",
        "deck":"vulnerable","defended":"no","county":parcel_county
    },
}

print("\n" + "=" * 75)
print("PARCEL PROFILE PREDICTIONS (P(destroyed | exposed))")
print("=" * 75)
preds_B = {}
print(f"  {'profile':40s}  {'LogReg':>10s}  {'Empirical':>12s}")
for name, prof in parcel_profiles_B.items():
    p_b = score_profile(prof)
    preds_B[name] = float(p_b)
    print(f"  {name:40s}  {p_b:>10.3f}  ", end="")
    if name.startswith("original") and "undefended" in name:
        print(f"({parcel_anchor_rates['original_undefended']:.3f})" if parcel_anchor_rates['original_undefended'] else "(n/a)")
    elif "original" in name and "defended" in name:
        print(f"({parcel_anchor_rates['original_defended']:.3f})" if parcel_anchor_rates['original_defended'] else "(n/a)")
    elif "partial_retrofit" in name:
        print(f"({parcel_anchor_rates['partial_retrofit']:.3f})" if parcel_anchor_rates['partial_retrofit'] else "(n/a)")
    else:
        print()

# Bootstrap CI on the "partial_retrofit_defended" profile (the most likely state)
print("\n=== Bootstrap 95% CI on 'partial_retrofit_defended' multiplier ===")
n_boot = 200
boot_preds = np.zeros(n_boot)
rng = np.random.default_rng(42)
prof = parcel_profiles_B["partial_retrofit_defended"]
prof_row = [1.0]
for f in feat_names:
    for lv in levels[f][1:]:
        prof_row.append(1.0 if prof.get(f) == lv else 0.0)
prof_arr = np.array(prof_row)

for b in range(n_boot):
    idx = rng.integers(0, len(X_int), len(X_int))
    wb, _ = fit_logreg(X_int[idx], y[idx], lr=0.05, l2=2.0, n_iter=2500)
    boot_preds[b] = sigmoid(prof_arr @ wb)
print(f"  Median: {np.median(boot_preds):.3f}")
print(f"  95% CI: [{np.percentile(boot_preds, 2.5):.3f}, {np.percentile(boot_preds, 97.5):.3f}]")
print(f"  → Phase 2 multiplier range will be drawn from this posterior")

# === Bayesian-flavored cohort multiplier (informative prior + bootstrap) ===
# Construct a "scenario" PMF from approach B predictions
phase2_scenarios = {
    "best_realized":   {"prob": 0.20, "mult": preds_B["full_retrofit_defended"],
                         "desc": "Full retrofit + active defense maintained 15 yrs"},
    "expected":        {"prob": 0.50, "mult": preds_B["partial_retrofit_defended"],
                         "desc": "Partial retrofit (vents/eaves) + sometimes defended"},
    "degraded":        {"prob": 0.25, "mult": preds_B["original_1980_defended"],
                         "desc": "No retrofit but defended (manager active)"},
    "worst":           {"prob": 0.05, "mult": preds_B["original_1980_undefended"],
                         "desc": "Original 1980 attributes, undefended (HOA failure)"},
}
E_mult = sum(s["prob"] * s["mult"] for s in phase2_scenarios.values())

print("\n" + "=" * 75)
print("PHASE 2 VULNERABILITY MULTIPLIER PMF")
print("=" * 75)
for name, s in phase2_scenarios.items():
    print(f"  {name:18s}  prob={s['prob']:>4.0%}  mult={s['mult']:.3f}  ({s['desc']})")
print(f"\n  E[vuln_mult] = {E_mult:.3f}")
print(f"  Phase 1 E[vuln_mult] was 0.568")
print(f"  Δ = {E_mult - 0.568:+.3f} ({(E_mult - 0.568)/0.568:+.0%} vs Phase 1)")

# Save Phase 2 inputs
phase2_inputs = {
    "training_n":                    int(len(y)),
    "training_auc":                  float(auc),
    "approach_A_empirical_rates":    {k: (float(v) if v is not None else None) for k, v in parcel_anchor_rates.items()},
    "approach_B_logreg_predictions": preds_B,
    "phase2_scenarios":              phase2_scenarios,
    "phase2_E_multiplier":           float(E_mult),
    "phase1_E_multiplier":            0.568,
    "bootstrap_partial_retrofit":    {
        "median":   float(np.median(boot_preds)),
        "ci_low":   float(np.percentile(boot_preds, 2.5)),
        "ci_high":  float(np.percentile(boot_preds, 97.5)),
        "mean":     float(boot_preds.mean()),
    },
    "logreg_top_coefficients":       {
        col_names[i]: float(w[i+1]) for i in top_idx
    },
}
with open("/tmp/dins_phase2_inputs.json", "w") as f:
    json.dump(phase2_inputs, f, indent=2)
print(f"\nSaved /tmp/dins_phase2_inputs.json")
