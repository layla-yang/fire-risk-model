#!/usr/bin/env python3
"""Build the wildfire self-insurance Lakeview dashboard programmatically."""
import sys, json, subprocess
sys.path.insert(0, "/Users/layla.yang/.vibe/marketplace/plugins/fe-databricks-tools/skills/databricks-lakeview-dashboard/resources")
from lakeview_builder import LakeviewDashboard

PROFILE = "fe-vm-wildfire-risk"
WAREHOUSE = "309c89fad003bef2"
PARENT = "/Users/layla.yang@databricks.com"
NAME = "wildfire_risk_decision_dashboard"

CATALOG = "wildfire_risk_catalog"
DT      = f"{CATALOG}.gold.decision_table_phase2"
MC      = f"{CATALOG}.gold.mc_dollar_distribution"

dash = LakeviewDashboard("Wildfire Self-Insurance Decision Dashboard — 759 Boulder Ct")

# ============================================================================
# DATASETS
# ============================================================================
dash.add_dataset(
    "dt_p3_15yr",
    "Phase 3 — 15-year horizon (headline)",
    f"SELECT * FROM {DT} WHERE phase = 3 AND horizon_yrs = 15"
)
dash.add_dataset(
    "dt_all",
    "All decision-table rows (P1 / P2 / P3 × 5/10/15yr)",
    f"""SELECT
      horizon_yrs,
      CASE WHEN phase = 1 THEN 'Phase 1: wildfire only'
           WHEN phase = 2 THEN 'Phase 2: + interior ignition'
           WHEN phase = 3 THEN 'Phase 3: + climate' END AS model_version,
      phase,
      p_loss_median,
      p_loss_p5,
      p_loss_p95,
      e_loss_usd,
      p95_loss_usd,
      p99_loss_usd,
      cum_premium_usd,
      verdict
    FROM {DT}"""
)
dash.add_dataset(
    "dt_dollar_compare",
    "Dollar comparison: premium vs E[loss] vs P99 by horizon",
    f"""SELECT
      horizon_yrs,
      CASE WHEN phase = 1 THEN 'Phase 1' WHEN phase = 2 THEN 'Phase 2' WHEN phase = 3 THEN 'Phase 3' END AS model_version,
      'Cumulative premium' AS metric, cum_premium_usd AS amount, 1 AS sort_order
    FROM {DT}
    UNION ALL
    SELECT horizon_yrs,
      CASE WHEN phase = 1 THEN 'Phase 1' WHEN phase = 2 THEN 'Phase 2' WHEN phase = 3 THEN 'Phase 3' END,
      'Expected loss (E[loss])', e_loss_usd, 2
    FROM {DT}
    UNION ALL
    SELECT horizon_yrs,
      CASE WHEN phase = 1 THEN 'Phase 1' WHEN phase = 2 THEN 'Phase 2' WHEN phase = 3 THEN 'Phase 3' END,
      'Tail loss (P99)', p99_loss_usd, 3
    FROM {DT}"""
)
dash.add_dataset(
    "mc_dist_15yr",
    "Monte Carlo dollar-loss distribution at 15 yrs (Phase 3)",
    f"""SELECT
      CASE WHEN dollar_loss = 0 THEN 'No loss ($0)'
           ELSE 'Total loss ($334K)' END AS outcome,
      COUNT(*) AS n_iters
    FROM {MC}
    WHERE horizon_yrs = 15
    GROUP BY 1"""
)
dash.add_dataset(
    "pcum_dist",
    "Distribution of simulated cumulative-loss probabilities by horizon",
    f"""SELECT
      horizon_yrs,
      ROUND(pcum_rise * 100, 1) AS pcum_pct,
      COUNT(*) AS n_iters
    FROM {MC}
    GROUP BY horizon_yrs, ROUND(pcum_rise * 100, 1)"""
)
dash.add_dataset(
    "horizon_summary",
    "Per-horizon Phase-3 summary for line + counter views",
    f"""SELECT
      horizon_yrs,
      p_loss_median * 100 AS p_loss_median_pct,
      p_loss_p5 * 100 AS p_loss_p5_pct,
      p_loss_p95 * 100 AS p_loss_p95_pct,
      e_loss_usd,
      p99_loss_usd,
      cum_premium_usd
    FROM {DT}
    WHERE phase = 3
    ORDER BY horizon_yrs"""
)

# ============================================================================
# LAYOUT
# 6-column grid. Rows go top-to-bottom.
# ============================================================================

# === TOP ROW (y=0): Banner counters for headline 15-yr numbers ===
dash.add_counter(
    dataset_name="dt_p3_15yr",
    value_field="p_loss_median",
    value_agg="MAX",
    title="Chance of total loss over 15 years (model median)",
    position={"x": 0, "y": 0, "width": 2, "height": 3},
)
dash.add_counter(
    dataset_name="dt_p3_15yr",
    value_field="e_loss_usd",
    value_agg="MAX",
    title="Expected loss over 15 yrs (avg across simulations)",
    position={"x": 2, "y": 0, "width": 2, "height": 3},
)
dash.add_counter(
    dataset_name="dt_p3_15yr",
    value_field="cum_premium_usd",
    value_agg="MAX",
    title="What 15 years of insurance premium would cost",
    position={"x": 4, "y": 0, "width": 2, "height": 3},
)

# === ROW 2 (y=3): The two key comparison charts ===
# Bar: $ comparison by horizon and metric
dash.add_bar_chart(
    dataset_name="dt_dollar_compare",
    x_field="horizon_yrs",
    y_field="amount",
    y_agg="SUM",
    title="Dollars: premium vs expected loss vs worst-case loss (by horizon × phase)",
    color_field="metric",
    show_labels=False,
    position={"x": 0, "y": 3, "width": 4, "height": 6},
    colors=["#9467bd", "#2ca02c", "#d62728"],
)

# MC outcome distribution at 15 yrs
dash.add_bar_chart(
    dataset_name="mc_dist_15yr",
    x_field="outcome",
    y_field="n_iters",
    y_agg="SUM",
    title="10,000 simulated 15-yr futures: how many end in total loss?",
    position={"x": 4, "y": 3, "width": 2, "height": 6},
    colors=["#2ca02c", "#d62728"],
    show_labels=True,
)

# === ROW 3 (y=9): Decision table + horizon trend ===
dash.add_bar_chart(
    dataset_name="horizon_summary",
    x_field="horizon_yrs",
    y_field="p_loss_median_pct",
    y_agg="MAX",
    title="Chance of total loss by horizon (Phase 3 model median, %)",
    position={"x": 0, "y": 9, "width": 3, "height": 5},
    colors=["#d62728"],
    show_labels=True,
)

# Decision table grid — full rows
dash.add_table(
    dataset_name="dt_all",
    title="Full decision matrix — every phase × horizon, every metric",
    columns=[
        {"field": "model_version",  "title": "Model version",    "type": "string"},
        {"field": "horizon_yrs",    "title": "Horizon (yrs)",    "type": "integer"},
        {"field": "p_loss_median",  "title": "P(loss) median",   "type": "float", "format": "0.00%"},
        {"field": "p_loss_p5",      "title": "5% lower",         "type": "float", "format": "0.00%"},
        {"field": "p_loss_p95",     "title": "95% upper",        "type": "float", "format": "0.00%"},
        {"field": "e_loss_usd",     "title": "Avg $ loss",       "type": "float", "format": "$#,##0"},
        {"field": "p99_loss_usd",   "title": "Worst-1%-case $",  "type": "float", "format": "$#,##0"},
        {"field": "cum_premium_usd","title": "Cum. premium $",   "type": "float", "format": "$#,##0"},
        {"field": "verdict",        "title": "Verdict",          "type": "string"},
    ],
    position={"x": 3, "y": 9, "width": 3, "height": 5},
)

# === ROW 4 (y=14): Filter widget ===
dash.add_filter_dropdown(
    dataset_name="dt_all",
    field="horizon_yrs",
    title="Filter: select horizon",
    position={"x": 0, "y": 14, "width": 2, "height": 2},
    multi_select=True,
)

# ============================================================================
# SERIALIZE + CREATE
# ============================================================================
dash_json = json.dumps({
    "datasets": dash.datasets,
    "pages":    dash.pages,
    "uiSettings": {
        "theme": {"widgetHeaderAlignment": "ALIGNMENT_UNSPECIFIED"},
        "applyModeEnabled": False,
    },
})

payload = {
    "display_name":         "Wildfire Self-Insurance Decision — 759 Boulder Ct",
    "warehouse_id":         WAREHOUSE,
    "parent_path":          PARENT,
    "serialized_dashboard": dash_json,
}
with open("/tmp/dashboard_payload.json","w") as f:
    json.dump(payload, f)
print(f"Payload built: {len(dash_json):,} bytes of dashboard JSON.")
print(f"Datasets: {len(dash.datasets)}, Widgets: {len(dash.pages[0]['layout'])}")

# Create the dashboard via API
r = subprocess.run(
    ["databricks","api","post","/api/2.0/lakeview/dashboards","--profile",PROFILE,
     "--json","@/tmp/dashboard_payload.json"],
    capture_output=True, text=True,
)
print("\n--- API response ---")
print("STDOUT:", r.stdout[:1500])
if r.stderr: print("STDERR:", r.stderr[:500])

# Parse + persist the ID
try:
    resp = json.loads(r.stdout)
    dashboard_id = resp.get("dashboard_id") or resp.get("id")
    if dashboard_id:
        print(f"\n✓ Dashboard created: dashboard_id = {dashboard_id}")
        print(f"  Edit URL: https://fevm-wildfire-risk.cloud.databricks.com/dashboardsv3/{dashboard_id}")
        print(f"  View URL: https://fevm-wildfire-risk.cloud.databricks.com/dashboardsv3/{dashboard_id}/published")

        # Publish it
        pub = subprocess.run(
            ["databricks","api","post",f"/api/2.0/lakeview/dashboards/{dashboard_id}/published","--profile",PROFILE,"--json","{}"],
            capture_output=True, text=True,
        )
        print(f"\n--- Publish response ---")
        print("STDOUT:", pub.stdout[:500])
        if pub.stderr: print("STDERR:", pub.stderr[:500])

        with open("/tmp/dashboard_info.json","w") as f:
            json.dump({"dashboard_id": dashboard_id, "edit_url": f"https://fevm-wildfire-risk.cloud.databricks.com/dashboardsv3/{dashboard_id}",
                       "view_url": f"https://fevm-wildfire-risk.cloud.databricks.com/dashboardsv3/{dashboard_id}/published"}, f)
except Exception as e:
    print(f"\nERROR parsing response: {e}")
