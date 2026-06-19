# Wildfire Total-Loss Forecasting Model

A data-driven model for evaluating wildfire risk to a specific real-estate purchase, built around a self-insurance / buy-or-not decision for a 1980 multi-unit condo at Lake Tahoe basin.

This repository contains the full reproducible pipeline:
- Databricks notebooks (Phase 1 → Phase 3 + Bayesian)
- Standalone analysis scripts
- An interactive Streamlit + Folium Databricks App for exploring fire history on a Tahoe map and walking through the buying decision
- Generated HTML decision memos (plain-English + technical versions)
- Derived data JSONs and plot outputs

> Decision-support only — consult a licensed financial / real-estate advisor before acting on any conclusions here.

## What the model does

Given a parcel (lat / lon), the model estimates the cumulative probability of total structural loss from wildfire and interior fire over **5 / 10 / 15-year holding horizons**, then translates that into dollar exposure and an investment-decision framework.

### Decision question

> *Should I buy a 24-unit, $420K condo where wildfire insurance is unavailable, given that the building structure is uninsured and a 1-in-25 chance of total loss exists over 15 years?*

### The model pipeline (four phases)

| Phase | What it adds | Headline output |
|---|---|---|
| **Phase 1** | FSim BP / FLEP + hand-coded vulnerability + Weibull MC | 0.25% 15-yr P(loss) — too low |
| **Phase 2** | DINS-fitted vulnerability (132K row-level fire-damage records) + NFPA interior-fire term | 3.04% 15-yr P(loss) |
| **Phase 2.5** | Bayesian hierarchical multiplier with CA county partial pooling (PyMC) | Posterior credible intervals on vulnerability |
| **Phase 3** | Cal-Adapt LOCA CMIP5 climate ensemble (8 GCMs × 2 RCPs) | 2.59% 15-yr median |
| **Phase 4** | NIFC empirical fire-frequency anchor (replaces FSim under-prediction) + corrected NFPA decomposition | **4.07% 15-yr median, range 0.8–14.9%** |

### Key empirical finding

The official US WRC FSim model said this parcel has only **0.04% annual fire chance**. NIFC fire-perimeter history (394 fires within 50 miles since 1984) shows **fires in 40 of 41 years**, including:
- **Gondola 2002** (643 ac) within **0.6 miles**
- **Autumn Hills 1996** (3,805 ac) within 1.4 miles
- **Caldor 2021** (221,786 ac) within 6 miles — destroyed ~1,000 structures
- **Angora 2007** (3,070 ac) within 7.9 miles — destroyed 254 homes

CAL FIRE DINS (lat/lon-filtered) confirms **~640 structures destroyed within 20 miles of the parcel in the last 41 years**, dominated by Caldor (318), Angora (309), Tamarack (13).

## Live deployment

The full model runs on a Databricks workspace with Unity Catalog tables backing an interactive app.

### Workspace

| Resource | URL / identifier |
|---|---|
| **Databricks workspace** | [`https://fevm-wildfire-risk.cloud.databricks.com`](https://fevm-wildfire-risk.cloud.databricks.com) |
| **CLI profile** | `fe-vm-wildfire-risk` |
| **Region** | `us-west-2` (AWS, serverless) |
| **SQL warehouse** | `Serverless Starter Warehouse` (ID `309c89fad003bef2`, Small) |
| **Workspace lifetime** | Provisioned 2026-06-19, expires 2026-07-19 (30-day FE-VM TTL) |

### Notebooks (in workspace)

- [`/Workspace/Users/layla.yang@databricks.com/wildfire_risk_phase1`](https://fevm-wildfire-risk.cloud.databricks.com/#workspace/Users/layla.yang@databricks.com/wildfire_risk_phase1) — Phase 1 (80/20 model)
- [`/Workspace/Users/layla.yang@databricks.com/wildfire_risk_v2`](https://fevm-wildfire-risk.cloud.databricks.com/#workspace/Users/layla.yang@databricks.com/wildfire_risk_v2) — Consolidated Phase 1+2+2.5+3 with PyMC Bayesian + Cal-Adapt climate

### Unity Catalog tables (`wildfire_risk_catalog`)

**Bronze** (raw / lightly-processed data from public APIs):

| Table | Rows | Description |
|---|---|---|
| `bronze.fsim_parcel_sample` | 75 | USFS WRC FSim BP/FLEP4/FLEP8 values on a 5×5 grid (30m cells) around the parcel, with scaling factor + provenance |
| `bronze.dins_raw` | 132,522 | Full CAL FIRE Damage Inspection (DINS) database with 45 fields per inspected structure |
| `bronze.fire_history` | 394 | NIFC Interagency Fire Perimeter History (fires ≥100 acres within 50 mi of parcel since 1984, including full polygon geometry as GeoJSON) |
| `bronze.caladapt_tasmax` | 8 series × ~95 yrs | Cal-Adapt LOCA CMIP5 annual maximum-temperature projections at the parcel point (4 GCMs × 2 RCPs) |

Plus a volume `bronze.raw` containing the original API responses for full audit trail:
- `parcel_geocode_provenance_v2026-06-19.json`
- `fsim_grid_parcel_v2026-06-19.json`
- `nifc_fires_v2026-06-19.json`
- `dins_raw_v2026-06-19.ndjson` (175 MB)

**Gold** (decision-ready outputs):

| Table | Rows | Description |
|---|---|---|
| `gold.decision_table_phase2` | 12 | Phase 1/2/3/4 × 5/10/15-year horizons. Columns: `p_loss_median`, `p_loss_p5`, `p_loss_p95`, `e_loss_usd`, `p95_loss_usd`, `p99_loss_usd`, `cum_premium_usd`, `verdict` |
| `gold.mc_dollar_distribution` | 30,000 | Long-format Monte Carlo iteration results (10K iters × 3 horizons) — used to back the dashboards and recompute distributions on demand |

### Interactive Databricks App

🔗 **[https://wildfire-fire-history-map-7474658710602767.aws.databricksapps.com](https://wildfire-fire-history-map-7474658710602767.aws.databricksapps.com)** (requires workspace SSO)

Four tabs:
1. **🗺️ Fire history map** — zoomable OpenStreetMap with 394 fire perimeters (or hover-only marker mode), distance rings, parcel marker, decade-layer toggles, click-to-zoom from the fire table
2. **📋 Historical destruction near here** — 41-year structure-loss summary (~640 destructions within 20 mi, dominated by Caldor 2021, Angora 2007, Tamarack 2021)
3. **🎯 What does 1-in-25 feel like?** — calibration tab comparing 4%-over-15-years to other life-event probabilities, with flood-zone real-estate as the closest parallel
4. **🏠 Should you buy?** — financial walk-through with STR income, 30-yr mortgage at 15% down, Schedule E tax write-off, and per-scenario payment-timing cards (settle vs. default path)

App identifiers:
- App name: `wildfire-fire-history-map`
- Source path: `/Workspace/Users/layla.yang@databricks.com/apps/wildfire-fire-history-map`
- Service principal: `3dcea42d-ae65-4d21-83d4-2de7adb8d2d5` (granted `CAN_USE` on the SQL warehouse and `SELECT` on `bronze.fire_history` + `gold.decision_table_phase2`)

### Lakeview dashboard

🔗 **[Wildfire Self-Insurance Decision Dashboard](https://fevm-wildfire-risk.cloud.databricks.com/dashboardsv3/01f16b9d7c30152da619cd90f6dc7dff/published)** (workspace SSO)

Headline counters + the gold-table-backed decision matrix + MC outcome distribution. Built programmatically via the Dashboard API (see `scripts/build_lakeview_dashboard.py`).

## Repository structure

```
fire-risk-model/
├── README.md                          # You are here
├── notebooks/                         # Databricks notebooks (.py source format)
│   ├── wildfire_risk_phase1.py        # Phase 1 — initial 80/20 model
│   └── wildfire_risk_v2.py            # Consolidated Phase 1+2+2.5+3 (716 lines)
├── app/                               # Streamlit + Folium Databricks App
│   ├── app.py                         # Main app (4 tabs: map / history / context / decision)
│   ├── app.yaml                       # Databricks Apps config
│   ├── requirements.txt
│   └── images/                        # Fire-history maps used in the decision tab
├── scripts/                           # Local analysis scripts (run-anywhere Python)
│   ├── wildfire_risk_local_run.py     # Phase 1 local runner (numpy-only)
│   ├── wildfire_phase2_run.py         # Phase 2 MC with DINS vulnerability
│   ├── wildfire_phase3_run.py         # Phase 3 with Cal-Adapt climate
│   ├── wildfire_phase4_run.py         # Final corrected model
│   ├── dins_train_local.py            # DINS logistic regression v1
│   ├── dins_train_v2.py               # DINS logistic regression v2 (missing-as-category)
│   ├── build_fire_history_map.py      # Generate the fire-perimeter maps
│   ├── build_decision_memo.py         # Generate HTML decision memo (technical)
│   ├── build_memo_plain.py            # Generate HTML decision memo (plain English)
│   ├── build_buy_or_not_memo.py       # Reframed buy/no-buy memo (light blue palette)
│   └── build_lakeview_dashboard.py    # Programmatic Lakeview dashboard creation
├── memos/                             # Generated HTML decision memos
│   ├── wildfire_buy_or_not_memo.html
│   ├── wildfire_decision_memo.html
│   ├── wildfire_decision_memo_plain.html
│   └── wildfire_str_calculator.html   # Standalone client-side STR ROI calculator
├── outputs/                           # Plot PNGs from the analysis scripts
└── data/                              # Derived data JSONs (small, reproducible)
    ├── fsim_grid_parcel.json          # USFS FSim 5x5 grid at parcel
    ├── caladapt_tasmax.json           # 8-GCM × 2-RCP Cal-Adapt tasmax time series
    ├── empirical_fire_stats.json      # NIFC-derived empirical frequency stats
    ├── parcel_geocode_provenance.json # Census Geocoder result for the parcel
    ├── wildfire_mc_results.json       # Phase 1 MC output
    ├── wildfire_p2_results.json       # Phase 2 MC output
    ├── wildfire_p3_results.json       # Phase 3 MC output
    ├── wildfire_p4_results.json       # Phase 4 MC output
    └── dins_phase2_inputs.json        # DINS-fitted vulnerability scenarios
```

## Data sources (all public)

| Dataset | Use |
|---|---|
| [USFS Wildfire Risk to Communities (FSim)](https://imagery.geoplatform.gov/iipp/rest/services/Fire_Aviation/USFS_EDW_RMRS_WRC_BurnProbability/ImageServer) | Modeled BP / FLEP at parcel |
| [NIFC Interagency Fire Perimeter History](https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/InterAgencyFirePerimeterHistory_All_Years_View/FeatureServer/0) | 394 fires ≥100 ac within 50 mi since 1984 |
| [CAL FIRE DINS (Damage Inspection Database)](https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/arcgis/rest/services/POSTFIRE_MASTER_DATA_SHARE/FeatureServer/0) | 132,522 structure-level damage records |
| [Cal-Adapt LOCA CMIP5](https://api.cal-adapt.org/api/) | 8-GCM × 2-RCP climate projections |
| [US Census Geocoder](https://geocoding.geo.census.gov/) | Parcel lat/lon |
| NFPA US residential fire statistics | Interior fire baseline rate |
| AirROI / Stateline NV STR market data | STR revenue assumption for the buying calculator |

## How to reproduce

### Run the analysis pipeline locally

```bash
# Requires: numpy, matplotlib (no scipy/pandas needed for phase 1)
cd scripts/
python3 wildfire_risk_local_run.py   # Phase 1 (uses pre-fetched FSim data in data/)
python3 wildfire_phase2_run.py        # Phase 2
python3 wildfire_phase3_run.py        # Phase 3 with climate
python3 wildfire_phase4_run.py        # Final corrected model
```

### Run the consolidated model on Databricks

1. Provision a serverless workspace (FE-VM or equivalent)
2. Set up Unity Catalog: `wildfire_risk_catalog` with `bronze` / `silver` / `gold` schemas + a `bronze.raw` volume
3. Import `notebooks/wildfire_risk_v2.py` as a Databricks notebook
4. Edit the `CONFIG` dict at the top (parcel address, lat/lon, financials)
5. Run all cells — pulls FSim from GeoPlatform, NIFC, Cal-Adapt; trains DINS Bayesian model in PyMC; runs the MC; writes gold tables

### Deploy the interactive app

```bash
cd app/
# Upload to workspace and deploy via Databricks CLI
databricks workspace import-dir . /Workspace/Users/<you>@databricks.com/apps/wildfire-fire-history-map --overwrite --profile <your-profile>
databricks apps deploy wildfire-fire-history-map \
    --source-code-path /Workspace/Users/<you>@databricks.com/apps/wildfire-fire-history-map \
    --profile <your-profile>
```

The app reads from `wildfire_risk_catalog.bronze.fire_history` and `gold.decision_table_phase2` and renders a 4-tab interface:

1. **🗺️ Fire history map** — zoomable OpenStreetMap with 394 fire perimeters, distance rings, filters, and a click-to-zoom fire table
2. **📋 Historical destruction near here** — 41-year structure-destruction summary
3. **🎯 What does 1-in-25 feel like?** — calibration tab comparing 4% to other life-event probabilities (with flood-zone analogy)
4. **🏠 Should you buy?** — financial walk-through with STR income, 30-yr mortgage, tax write-off, and explicit payment-timing for fire scenarios

### Open the standalone decision memos

`memos/wildfire_buy_or_not_memo.html` is the canonical decision memo — fully self-contained (embedded charts), opens in any browser. `memos/wildfire_str_calculator.html` is a standalone client-side STR purchase calculator with sliders.

## Key model assumptions

| Assumption | Value | Source |
|---|---|---|
| Wildfire annual P(loss) anchor | Lognormal(med=0.0035/yr, σ=0.6) | NIFC empirical (7%/yr fire within 5 mi × ~5% conditional) |
| Building vulnerability multiplier | DINS-fitted scenario PMF (best 0.026 → worst 0.583) | CAL FIRE DINS logistic regression, n=2007 multi-residence rows |
| Interior fire per-unit annual rate | 6.6e-3/yr | NFPA apartment fire statistics |
| P(shell loss \| unit fire) | Beta(5.4, 994.6), mean 0.5% | NFPA decomposed: spread × structural × total-loss |
| Building units | 24 | HOA disclosure |
| Climate trend | β = 0.5–1.0/°C, Cal-Adapt LOCA CMIP5 ensemble | Westerling 2018, Goss 2020 |
| Marginal tax rate | 32% federal (NV no state) | User's tax bracket |
| Schedule E depreciation | $336K basis / 27.5 yrs = $12,218/yr | IRS residential rental |
| Property appreciation | 3%/yr | Long-run Tahoe-area average |
| Uninsurability discount at sale | 10–25% | Emerging WUI market evidence |

## Honest limits

1. **CA → NV DINS transfer.** Most building-vulnerability data is California-only. Nevada has similar fire physics; codes and wind patterns differ at the margin.
2. **FSim 270 m resolution** vs. single parcel — model averages a wide neighborhood.
3. **Pre-Caldor LANDFIRE 2020 fuels** — FSim's modeled BP doesn't reflect Sierra fire escalation since 2021.
4. **NFPA national stats applied to a 1980 building** — older construction likely has higher rates than average.
5. **Insurance loading interpretation.** Insurance pricing implies 14–27× higher risk than the model. Likely a mix of carrier WUI-exit pricing (overcharge) and partial-loss / cat-correlation effects the model doesn't capture.
6. **Default-path tax mechanics** are simplified. Real-world CPA modeling needed for the 1099-C / casualty-loss interaction.

## License

This work is provided as-is for decision support and educational purposes. Public data sources retain their original licenses (USFS, CAL FIRE, NIFC, Cal-Adapt, NFPA, Census).

## Acknowledgments

Built with [Claude Code](https://claude.com/claude-code) as the modeling assistant. All data sources cited inline. Decision is the user's; the model is a tool.
