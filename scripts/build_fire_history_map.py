#!/usr/bin/env python3
"""Build a regional fire-history map around the parcel.
Shows: parcel location, all fire perimeters since 1984 within 50 miles,
named highlights (Caldor, Angora, Gondola), and empirical fire frequency stats."""

import json
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap

with open("/tmp/nifc_fires_with_caldor.json") as f:
    data = json.load(f)

LAT = data["parcel"]["lat"]; LON = data["parcel"]["lon"]
fires = data["fires"]

print(f"Building map: {len(fires)} fires within 50 miles of parcel")

# === Compute empirical fire frequency by radius ===
radii_mi = [2, 5, 10, 20, 30, 50]
print("\nEmpirical fire frequency by radius (1984-2025, 41 years):")
print(f"  {'Radius':>10s}  {'# fires':>8s}  {'# unique yrs':>13s}  {'≥10K acres':>11s}  {'Annual rate':>12s}")
freq_summary = {}
for r in radii_mi:
    near = [f for f in fires if float(f["min_perimeter_dist_mi"]) <= r]
    years = {int(f["year"]) for f in near}
    big = [f for f in near if float(f["acres"]) >= 10000]
    rate = len(years) / 41
    freq_summary[r] = {"n_fires": len(near), "n_unique_years": len(years),
                        "n_big_fires": len(big), "annual_rate": rate}
    print(f"  {r:>5d} mi  {len(near):>8d}  {len(years):>13d}  {len(big):>11d}  {rate:>10.1%}")

# === The map ===
fig, axes = plt.subplots(1, 2, figsize=(18, 9))

# Define colormap by fire year — older = lighter
years = sorted({int(f["year"]) for f in fires})
ymin, ymax = min(years), max(years)
cmap = plt.colormaps.get_cmap("YlOrRd")

def year_color(yr, alpha=0.55):
    norm = (yr - ymin) / max(1, ymax - ymin)
    c = list(cmap(0.25 + 0.75 * norm))
    c[3] = alpha
    return c

# --- LEFT PANEL: 50-mile context map ---
ax = axes[0]
for f in fires:
    rings = f.get("geometry_rings", [])
    for ring in rings:
        if len(ring) < 3: continue
        ax.add_patch(MplPolygon([(p[0], p[1]) for p in ring],
                                 facecolor=year_color(int(f["year"])),
                                 edgecolor="#660000", linewidth=0.3))

# Label the major fires
big_named = [f for f in fires if float(f["acres"]) >= 5000 and float(f["min_perimeter_dist_mi"]) < 40]
for f in big_named:
    rings = f.get("geometry_rings", [])
    if not rings: continue
    pts = [p for ring in rings for p in ring]
    cx = sum(p[0] for p in pts)/len(pts); cy = sum(p[1] for p in pts)/len(pts)
    label = f"{f['incident'][:18]} {f['year']}\n{int(float(f['acres'])):,} ac"
    ax.annotate(label, (cx, cy), fontsize=8, color="#222", fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"))

# Parcel marker
ax.plot(LON, LAT, '*', color='blue', markersize=24, markeredgecolor='black', markeredgewidth=1.5, zorder=5)
ax.annotate("759 Boulder Ct\n(your parcel)", (LON, LAT), xytext=(LON+0.15, LAT-0.05),
            fontsize=11, fontweight='bold', color='darkblue',
            arrowprops=dict(arrowstyle='->', color='darkblue', lw=1.5))

# Concentric distance rings
from matplotlib.patches import Circle
for r_mi, label in [(5, "5 mi"), (10, "10 mi"), (20, "20 mi"), (30, "30 mi")]:
    r_deg = r_mi / 69  # rough lat-deg (close enough for visual)
    circle = Circle((LON, LAT), r_deg, fill=False, edgecolor='#444', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.add_patch(circle)
    ax.text(LON, LAT + r_deg, label, fontsize=9, color='#666', ha='center', va='bottom')

ax.set_xlim(LON - 0.75, LON + 0.75)
ax.set_ylim(LAT - 0.65, LAT + 0.65)
ax.set_aspect('equal')
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title(f"Fire history within 50 miles of 759 Boulder Ct (1984–2025)\n{len(fires)} fires ≥100 acres • color = year (lighter = older)", fontsize=12)
ax.grid(True, alpha=0.3)

# --- RIGHT PANEL: empirical frequency stats ---
ax2 = axes[1]
radii = sorted(freq_summary.keys())
rates = [freq_summary[r]["annual_rate"]*100 for r in radii]
ax2.barh(range(len(radii)), rates, color="#d62728", alpha=0.75)
ax2.set_yticks(range(len(radii)))
ax2.set_yticklabels([f"Within {r} mi" for r in radii])
ax2.set_xlabel("% of years (1984-2025) with a fire ≥100 acres in this radius")
ax2.set_title("How often has a fire occurred near this parcel?", fontsize=12)
for i, r in enumerate(radii):
    n = freq_summary[r]["n_fires"]; big = freq_summary[r]["n_big_fires"]
    ax2.text(rates[i] + 1, i, f"{rates[i]:.0f}%  ({n} fires, {big} ≥10K ac)",
              va='center', fontsize=10)
ax2.set_xlim(0, max(rates) * 1.45 if rates else 100)

# Add comparison annotation
ax2.axvline(x=0.04, color='blue', linestyle='-', linewidth=2)
ax2.text(0.5, 5.3, "WRC FSim model says\nparcel BP = 0.04%/yr",
          fontsize=11, color='blue', fontweight='bold')

ax2.invert_yaxis()
plt.tight_layout()
plt.savefig("/tmp/wildfire_history_map.png", dpi=140, bbox_inches='tight')
print(f"\nSaved /tmp/wildfire_history_map.png")
plt.close()

# === Closer-up map: 15-mile radius ===
fig2, ax = plt.subplots(figsize=(12, 12))
for f in fires:
    if float(f["min_perimeter_dist_mi"]) > 20: continue
    rings = f.get("geometry_rings", [])
    for ring in rings:
        if len(ring) < 3: continue
        ax.add_patch(MplPolygon([(p[0], p[1]) for p in ring],
                                 facecolor=year_color(int(f["year"]), alpha=0.65),
                                 edgecolor="#660000", linewidth=0.5))
# Label all named fires within 20 mi
close_named = [f for f in fires if float(f["min_perimeter_dist_mi"]) <= 20
                and f["incident"] not in ("Unknown", "unknown", "(unnamed)")]
labeled = set()
for f in sorted(close_named, key=lambda x: float(x["min_perimeter_dist_mi"])):
    name = f["incident"]
    if name in labeled: continue
    labeled.add(name)
    rings = f.get("geometry_rings", [])
    if not rings: continue
    pts = [p for ring in rings for p in ring]
    cx = sum(p[0] for p in pts)/len(pts); cy = sum(p[1] for p in pts)/len(pts)
    label = f"{name}\n{f['year']} • {int(float(f['acres'])):,} ac • {float(f['min_perimeter_dist_mi']):.1f} mi"
    ax.annotate(label, (cx, cy), fontsize=8.5, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="#660000"))

ax.plot(LON, LAT, '*', color='blue', markersize=30, markeredgecolor='black', markeredgewidth=2, zorder=10)
ax.annotate("759 Boulder Ct", (LON, LAT), xytext=(LON+0.05, LAT-0.04),
            fontsize=14, fontweight='bold', color='darkblue', zorder=10,
            arrowprops=dict(arrowstyle='->', color='darkblue', lw=2))

# Distance rings
for r_mi, label in [(1, "1 mi"), (3, "3 mi"), (5, "5 mi"), (10, "10 mi"), (15, "15 mi")]:
    r_deg = r_mi / 69
    circle = Circle((LON, LAT), r_deg, fill=False, edgecolor='#333', linestyle='--', linewidth=1, alpha=0.7)
    ax.add_patch(circle)
    ax.text(LON, LAT + r_deg + 0.005, label, fontsize=10, color='#444', ha='center', va='bottom', fontweight='bold')

ax.set_xlim(LON - 0.32, LON + 0.32)
ax.set_ylim(LAT - 0.27, LAT + 0.27)
ax.set_aspect('equal')
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title(f"Close-up: fires within 20 miles of the parcel (1984–2025)\nThe Gondola fire (2002) burned within 0.6 miles • Caldor (2021) within 6 miles", fontsize=13)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/tmp/wildfire_history_closeup.png", dpi=140, bbox_inches='tight')
print(f"Saved /tmp/wildfire_history_closeup.png")
plt.close()

# Save empirical stats
with open("/tmp/empirical_fire_stats.json","w") as f:
    json.dump({
        "parcel": {"lat": LAT, "lon": LON},
        "period": "1984-2025 (41 years)",
        "frequency_by_radius": freq_summary,
        "closest_5_fires": sorted(fires, key=lambda x: float(x["min_perimeter_dist_mi"]))[:5],
    }, f, default=str, indent=2)
print("Saved /tmp/empirical_fire_stats.json")
