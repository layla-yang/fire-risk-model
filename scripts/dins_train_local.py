#!/usr/bin/env python3
"""Train logistic regression on DINS Multiple-Residence subset (pure numpy — no sklearn).
Produce vulnerability multipliers for the parcel's profile across hardening scenarios."""
import json
import numpy as np

with open("/tmp/dins_multires_chunk0.json") as f:
    raw = json.load(f)
cols = raw["columns"]
rows = raw["rows"]
print(f"Loaded {len(rows)} multi-residence DINS rows, cols={len(cols)}")

def cidx(name): return cols.index(name)

# === Target: destroyed (>50%) — the "total loss" event ===
y = np.array([1 if r[cidx("DAMAGE")] == "Destroyed (>50%)" else 0 for r in rows])
print(f"Destroyed rate (raw): {y.mean():.3f} (n={len(y)})")

# === Feature engineering ===
# 1. Year built buckets — encode as ordinal (older = more vulnerable)
def year_bucket(yr):
    if yr is None: return None
    yr = int(yr) if not isinstance(yr, int) else yr
    if yr <  1960: return 0
    if yr <  1980: return 1
    if yr <  1990: return 2  # OUR PARCEL is here (1980)
    if yr < 2000:  return 3
    return 4
yb_raw = [year_bucket(r[cidx("YEARBUILT")]) for r in rows]

# 2. Roof: Asphalt/Wood = vulnerable (1), Tile/Metal/Concrete = hardened (0)
def roof_hardened(r):
    s = (r or "").strip()
    if s in ("Tile", "Metal", "Concrete"): return 1
    if s in ("Asphalt", "Wood", "Combustible"): return 0
    return None
roof = [roof_hardened(r[cidx("ROOFCONSTRUCTION")]) for r in rows]

# 3. Vent: <=1/8 mesh = hardened (1), >1/8 mesh / unscreened = vulnerable (0)
def vent_hardened(r):
    s = (r or "").strip()
    if s == "Mesh Screen <= 1/8\"": return 1
    if s in ("Mesh Screen > 1/8\"", "Unscreened"): return 0
    if s == "No Vents": return 1  # no vents = no ember entry
    return None
vent = [vent_hardened(r[cidx("VENTSCREEN")]) for r in rows]

# 4. Eaves: enclosed/none = hardened (1), open = vulnerable (0)
def eaves_hardened(r):
    s = (r or "").strip()
    if s in ("Enclosed", "No Eaves"): return 1
    if s in ("Unenclosed", "Open"): return 0
    return None
eaves = [eaves_hardened(r[cidx("EAVES")]) for r in rows]

# 5. Window: multi-pane = hardened, single-pane = vulnerable
def window_hardened(r):
    s = (r or "").strip()
    if "Multi" in s or "Tempered" in s or "Double" in s: return 1
    if s in ("Single Pane",): return 0
    return None
window = [window_hardened(r[cidx("WINDOWPANE")]) for r in rows]

# 6. Defensive actions: defended = hardened (1)
def defended(r):
    s = (r or "").strip()
    if s in ("Engine Company Actions", "Fire Department", "Combination of Actions",
             "Hand Crew", "Bulldozer", "Air"): return 1
    if s == "None":  return 0
    return None
defended_arr = [defended(r[cidx("DEFENSIVEACTIONS")]) for r in rows]

# 7. Combustible deck (1 = has combustible deck, 0 = no deck or non-combustible)
def deck_combustible(elev, og):
    for s in (elev, og):
        s = (s or "").strip()
        if s and s != "No Deck/Porch" and s != "None" and "Composite" not in s and "Concrete" not in s:
            return 1
    return 0
deck = [deck_combustible(r[cidx("DECKPORCHELEVATED")], r[cidx("DECKPORCHONGRADE")]) for r in rows]

# Assemble feature matrix
features_raw = np.column_stack([yb_raw, roof, vent, eaves, window, defended_arr, deck])
feat_names = ["year_bucket", "roof_hardened", "vent_hardened", "eaves_hardened",
              "window_hardened", "defended", "deck_combustible"]

# Drop rows with any None feature (keep them simple)
mask_complete = np.array([all(v is not None for v in row) for row in features_raw])
print(f"\nRows with all features known: {mask_complete.sum()}/{len(rows)} ({mask_complete.mean():.1%})")

# For training, use complete rows
X_full = features_raw[mask_complete].astype(float)
y_full = y[mask_complete]
print(f"Training set: {X_full.shape}, destroyed rate: {y_full.mean():.3f}")

# Center year_bucket
X = X_full.copy()
X[:, 0] = X[:, 0] - 2  # center around the 1980-89 cohort

# Add intercept
X_with_int = np.column_stack([np.ones(len(X)), X])

# === Logistic regression via L2-regularized gradient descent ===
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

def fit_logreg(X, y, lr=0.1, l2=1.0, n_iter=5000, tol=1e-7):
    n, p = X.shape
    w = np.zeros(p)
    for it in range(n_iter):
        z = X @ w
        ph = sigmoid(z)
        grad = X.T @ (ph - y) / n + l2 * w / n
        grad[0] = X[:, 0].T @ (ph - y) / n  # don't regularize intercept
        w_new = w - lr * grad
        if np.max(np.abs(w_new - w)) < tol:
            break
        w = w_new
    return w, it+1

w, n_iter = fit_logreg(X_with_int, y_full, lr=0.1, l2=1.0, n_iter=10000)
print(f"\nLogReg converged in {n_iter} iterations")
print(f"Coefficients (log-odds):")
print(f"  {'intercept':25s}  {w[0]:+.4f}")
for nm, wi in zip(feat_names, w[1:]):
    print(f"  {nm:25s}  {wi:+.4f}  → odds ratio {np.exp(wi):.3f}")

# Training accuracy + AUC
ph_train = sigmoid(X_with_int @ w)
acc = (ph_train.round() == y_full).mean()
# Manual AUC
order = np.argsort(-ph_train)
y_sorted = y_full[order]
tp = np.cumsum(y_sorted)
fp = np.cumsum(1 - y_sorted)
tpr = tp / tp[-1]
fpr = fp / fp[-1]
auc = np.trapz(tpr, fpr)
print(f"\nTraining accuracy: {acc:.3f}")
print(f"AUC: {auc:.3f}")

# Calibration: predicted vs observed in 5 bins
bins = np.percentile(ph_train, [0, 20, 40, 60, 80, 100])
bin_idx = np.digitize(ph_train, bins[1:-1])
print("\nCalibration (predicted vs observed):")
print(f"  {'bin':>4s}  {'n':>4s}  {'mean_pred':>10s}  {'observed':>10s}")
for b in range(5):
    sel = (bin_idx == b)
    if sel.sum() > 0:
        print(f"  {b:>4d}  {sel.sum():>4d}  {ph_train[sel].mean():>10.3f}  {y_full[sel].mean():>10.3f}")

# === Predict on parcel profile ===
# 759 Boulder Ct, 1980 build, multi-unit, user said well-maintained mitigation but original construction
parcel_profiles = {
    "original_1980_undefended":         {"year_bucket": 2, "roof_hardened": 0, "vent_hardened": 0, "eaves_hardened": 0, "window_hardened": 0, "defended": 0, "deck_combustible": 1},
    "original_1980_defended":           {"year_bucket": 2, "roof_hardened": 0, "vent_hardened": 0, "eaves_hardened": 0, "window_hardened": 1, "defended": 1, "deck_combustible": 1},
    "partial_retrofit_defended":        {"year_bucket": 2, "roof_hardened": 0, "vent_hardened": 1, "eaves_hardened": 1, "window_hardened": 1, "defended": 1, "deck_combustible": 0},
    "full_retrofit_defended":           {"year_bucket": 2, "roof_hardened": 1, "vent_hardened": 1, "eaves_hardened": 1, "window_hardened": 1, "defended": 1, "deck_combustible": 0},
    "lapsed_no_defense":                {"year_bucket": 2, "roof_hardened": 0, "vent_hardened": 0, "eaves_hardened": 0, "window_hardened": 1, "defended": 0, "deck_combustible": 1},
}

print("\n" + "=" * 70)
print("PARCEL PROFILE PREDICTIONS (1980-cohort multi-unit):")
print("=" * 70)
preds = {}
for name, profile in parcel_profiles.items():
    x = np.array([1.0] + [profile[f] for f in feat_names])
    x[1] = x[1] - 2  # center year
    z = x @ w
    p = sigmoid(z)
    preds[name] = p
    print(f"  {name:40s}  P(destroyed | exposed) = {p:.3f}  ({p*100:.1f}%)")

# Bootstrap CI for the "expected" profile
print("\n=== Bootstrap 95% CI on the 'expected/partial_retrofit_defended' multiplier ===")
n_boot = 500
boot_preds = np.zeros(n_boot)
rng = np.random.default_rng(42)
for b in range(n_boot):
    idx = rng.integers(0, len(X_with_int), len(X_with_int))
    Xb, yb = X_with_int[idx], y_full[idx]
    wb, _ = fit_logreg(Xb, yb, lr=0.1, l2=1.0, n_iter=2000)
    x = np.array([1.0] + [parcel_profiles["partial_retrofit_defended"][f] for f in feat_names])
    x[1] = x[1] - 2
    boot_preds[b] = sigmoid(x @ wb)
print(f"  Median: {np.median(boot_preds):.3f}  (Phase 1 expected was 0.55)")
print(f"  95% CI: [{np.percentile(boot_preds, 2.5):.3f}, {np.percentile(boot_preds, 97.5):.3f}]")
print(f"  Mean:   {boot_preds.mean():.3f}")

# Save Phase 2 multiplier inputs for downstream MC
phase2_inputs = {
    "vulnerability_predictions": preds,
    "bootstrap_median":   float(np.median(boot_preds)),
    "bootstrap_ci_low":   float(np.percentile(boot_preds, 2.5)),
    "bootstrap_ci_high":  float(np.percentile(boot_preds, 97.5)),
    "training_n":         int(len(y_full)),
    "training_auc":       float(auc),
    "logreg_coefficients": {nm: float(wi) for nm, wi in zip(["intercept"] + feat_names, w)},
}
with open("/tmp/dins_phase2_inputs.json", "w") as f:
    json.dump(phase2_inputs, f, indent=2)
print(f"\nSaved /tmp/dins_phase2_inputs.json")
