"""
Wildfire risk decision app for 759 Boulder Ct, Stateline NV.

Tab 1: Interactive fire-history map (Folium + OpenStreetMap, zoomable)
       with a toggle between full perimeter view and marker-only-on-hover view.
Tab 2: Plain-English buy / don't-buy decision readout backed by the model.

Reads `wildfire_risk_catalog.bronze.fire_history` and
`wildfire_risk_catalog.gold.decision_table_phase2` from a SQL warehouse.
"""

from __future__ import annotations

import json
import os
import sys
import traceback

import branca.colormap as cm
import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PARCEL_LAT   = 38.967876
PARCEL_LON   = -119.887425
PARCEL_LABEL = "759 Boulder Ct (your parcel)"

FIRE_TABLE     = "wildfire_risk_catalog.bronze.fire_history"
DECISION_TABLE = "wildfire_risk_catalog.gold.decision_table_phase2"

RING_RADII_MI    = [1, 3, 5, 10, 15, 20, 30]
METERS_PER_MILE  = 1609.34

SHELL_BASIS = 334_000
PREMIUM     = 24_000


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="759 Boulder Ct — Wildfire Decision",
    page_icon="🔥",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _get_workspace_client():
    from databricks.sdk import WorkspaceClient
    return WorkspaceClient()


def _run_sql(query: str, row_limit: int = 10_000) -> pd.DataFrame:
    """Execute SQL via Statement Execution REST API (INLINE disposition).

    The SQL connector's CloudFetch path is unreachable from the Databricks
    Apps sandbox; this avoids S3 entirely.
    """
    from databricks.sdk.service.sql import Disposition, Format, StatementState

    warehouse_id = os.environ.get("DATABRICKS_WAREHOUSE_ID", "")
    ws = _get_workspace_client()
    resp = ws.statement_execution.execute_statement(
        statement=query,
        warehouse_id=warehouse_id,
        disposition=Disposition.INLINE,
        format=Format.JSON_ARRAY,
        wait_timeout="50s",
        row_limit=row_limit,
    )
    while resp.status and resp.status.state in (StatementState.PENDING, StatementState.RUNNING):
        resp = ws.statement_execution.get_statement(resp.statement_id)
    if not resp.status or resp.status.state != StatementState.SUCCEEDED:
        err = resp.status.error if resp.status else "unknown error"
        raise RuntimeError(f"Statement failed: {err}")
    cols = [c.name for c in resp.manifest.schema.columns]
    rows = resp.result.data_array or []
    return pd.DataFrame(rows, columns=cols)


@st.cache_data(ttl=3600, show_spinner="Loading fire attributes…")
def load_fire_attrs() -> pd.DataFrame:
    """Load light-weight attributes (no geometry) for ALL fires — small payload.

    Geometries are huge (Caldor alone ~5MB of JSON) and can blow past the
    Statement Execution INLINE response cap, causing silent truncation that
    drops the largest fires from the result. Splitting attrs from geometry
    is the robust pattern.
    """
    df = _run_sql(f"""
        SELECT incident, year, acres, centroid_lat, centroid_lon,
               dist_mi, min_perimeter_dist_mi
        FROM {FIRE_TABLE}
    """, row_limit=10_000)
    for col in ("year", "acres", "centroid_lat", "centroid_lon",
                "dist_mi", "min_perimeter_dist_mi"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    print(
        f"[load_fire_attrs] rows={len(df)}  acres_max={df['acres'].max():.0f}  "
        f"dist_min={df['dist_mi'].min():.2f}",
        file=sys.stderr, flush=True,
    )
    return df


@st.cache_data(ttl=3600, show_spinner="Loading fire geometries…")
def load_fire_geoms(min_acres: float, max_dist_mi: float) -> pd.DataFrame:
    """Load geometries for the filtered subset only.

    Keyed on (min_acres, max_dist_mi) so Streamlit cache returns previously-
    fetched subsets without going back to the warehouse.
    """
    df = _run_sql(f"""
        SELECT incident, year, geometry_geojson
        FROM {FIRE_TABLE}
        WHERE acres >= {float(min_acres)} AND dist_mi <= {float(max_dist_mi)}
    """, row_limit=10_000)
    print(
        f"[load_fire_geoms] min_acres={min_acres} max_dist={max_dist_mi} "
        f"→ {len(df)} rows",
        file=sys.stderr, flush=True,
    )
    return df


@st.cache_data(ttl=3600, show_spinner="Loading decision table…")
def load_decision_table() -> pd.DataFrame:
    df = _run_sql(f"""
        SELECT phase, horizon_yrs, p_loss_median, p_loss_p5, p_loss_p95,
               e_loss_usd, p99_loss_usd, cum_premium_usd, verdict
        FROM {DECISION_TABLE}
        WHERE phase = 4
        ORDER BY horizon_yrs
    """)
    for col in ("phase", "horizon_yrs", "p_loss_median", "p_loss_p5", "p_loss_p95",
                 "e_loss_usd", "p99_loss_usd", "cum_premium_usd"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_rings(geom_json: str) -> list[list[tuple[float, float]]]:
    if not geom_json:
        return []
    try:
        rings = json.loads(geom_json)
    except (ValueError, TypeError):
        return []
    parsed: list[list[tuple[float, float]]] = []
    for ring in rings:
        if not isinstance(ring, list):
            continue
        latlon = [(pt[1], pt[0]) for pt in ring if isinstance(pt, list) and len(pt) >= 2]
        if len(latlon) >= 3:
            parsed.append(latlon)
    return parsed


def _size_color(acres: float) -> str:
    if acres >= 100_000: return "#7a0177"
    if acres >=  25_000: return "#c51b8a"
    if acres >=   5_000: return "#f768a1"
    if acres >=   1_000: return "#fa9fb5"
    return "#fcc5c0"


def _centroid_of(rings: list[list[tuple[float, float]]]) -> tuple[float, float]:
    pts = [p for ring in rings for p in ring]
    if not pts:
        return (0.0, 0.0)
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


# ---------------------------------------------------------------------------
# Load data once
# ---------------------------------------------------------------------------
try:
    fires_df    = load_fire_attrs()
    decision_df = load_decision_table()
except Exception as exc:  # noqa: BLE001
    st.error(f"Failed to load data from Databricks: {exc}")
    st.stop()


# ===========================================================================
# TABS
# ===========================================================================
tab_map, tab_history, tab_context, tab_decision = st.tabs([
    "🗺️  Fire history map",
    "📋  Historical destruction near here",
    "🎯  What does 1-in-25 feel like?",
    "🏠  Should you buy?",
])


# ---------------------------------------------------------------------------
# TAB 1 — Interactive fire-history map
# ---------------------------------------------------------------------------
with tab_map:
    st.title("Fire history near 759 Boulder Ct — 1984 to 2025")
    st.caption(
        "Zoom in and out to see real fires that have occurred around this property. "
        "Use the sidebar to filter."
    )

    if fires_df.empty:
        st.warning("No fire history rows returned.")
    else:
        # ---------- Sidebar filters ----------
        with st.sidebar:
            st.header("Map filters")
            year_min_data = int(fires_df["year"].min())
            year_max_data = max(int(fires_df["year"].max()), 2025)
            year_lo, year_hi = st.slider(
                "Year range",
                min_value=year_min_data, max_value=year_max_data,
                value=(1984, 2025),
            )
            acres_max = int(fires_df["acres"].max())
            min_acres = st.slider(
                "Minimum fire size (acres)",
                min_value=0, max_value=max(1000, acres_max),
                value=100, step=50,
            )
            max_dist = st.slider(
                "Max distance from parcel (mi)",
                min_value=1, max_value=50, value=30,
            )

            st.divider()
            st.subheader("Display")
            show_perims = st.toggle(
                "Show full fire perimeters",
                value=True,
                help="ON: every fire polygon is drawn on the map.  "
                     "OFF: each fire is shown as a small dot; hover over a dot to see its info.",
            )
            show_rings = st.toggle(
                "Show distance rings (1, 3, 5, 10, 15, 20, 30 mi)",
                value=True,
            )
            color_mode = st.radio(
                "Color by",
                options=["Year (older = lighter)", "Size"],
                index=0,
            )

        # ---------- Apply filters ----------
        # All numeric columns are float64 from the loader. Build mask piece by piece
        # and fill any boolean NA → False (so rows with missing values get excluded
        # rather than propagating NA through the AND chain).
        y = fires_df["year"]
        a = fires_df["acres"]
        d = fires_df["dist_mi"]
        mask = (
            (y >= year_lo).fillna(False)
            & (y <= year_hi).fillna(False)
            & (a >= min_acres).fillna(False)
            & (d <= max_dist).fillna(False)
        )
        filtered = fires_df.loc[mask].copy()
        # Debug
        print(
            f"[filter] year_lo={year_lo} year_hi={year_hi} min_acres={min_acres} "
            f"max_dist={max_dist} → {mask.sum()} fires match",
            file=sys.stderr, flush=True,
        )

        # Fetch geometries for the filtered subset and merge in.
        # Cache key = (min_acres, max_dist); the year/show-rings filters are
        # client-side only so the geometry payload stays minimal.
        try:
            geoms_df = load_fire_geoms(float(min_acres), float(max_dist))
            # Both keys must have matching dtypes for merge
            geoms_df["year"] = pd.to_numeric(geoms_df["year"], errors="coerce")
            filtered["year"] = pd.to_numeric(filtered["year"], errors="coerce")
            filtered = filtered.merge(
                geoms_df[["incident", "year", "geometry_geojson"]],
                on=["incident", "year"], how="left",
            )
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Could not load geometries: {exc}")
            filtered["geometry_geojson"] = ""

        # ---------- Pre-build the fire table (used for display + selection lookup) ----------
        fire_table = (
            filtered.copy()
            .sort_values(["year", "acres"], ascending=[False, False])
            .reset_index(drop=True)
        )

        # ---------- Read any prior table-row selection from session_state ----------
        # When the user clicks a row, Streamlit reruns with the new selection
        # already populated in session_state under the dataframe's key.
        selected_fire = None
        prior_state = st.session_state.get("fire_table_select")
        if prior_state is not None:
            try:
                sel_rows = (
                    prior_state.selection.rows
                    if hasattr(prior_state, "selection")
                    else prior_state.get("selection", {}).get("rows", [])
                )
            except Exception:
                sel_rows = []
            if sel_rows:
                idx = sel_rows[0]
                if 0 <= idx < len(fire_table):
                    row = fire_table.iloc[idx]
                    selected_fire = {
                        "incident": str(row["incident"]) if pd.notna(row["incident"]) else "",
                        "year":     int(row["year"]) if pd.notna(row["year"]) else None,
                    }

        # ---------- Build folium map ----------
        year_cmap = cm.linear.YlOrRd_09.scale(1984, 2025)
        year_cmap.caption = "Fire year (lighter = older)"

        # Default zoom 9 (was 10) so the Caldor 2021 footprint (~24 mi west of
        # parcel, 221K acres) is fully visible without the user having to pan.
        fmap = folium.Map(
            location=[PARCEL_LAT, PARCEL_LON],
            zoom_start=9,
            tiles="OpenStreetMap",
            control_scale=True,
        )

        # Parcel marker
        folium.Marker(
            location=[PARCEL_LAT, PARCEL_LON],
            popup=folium.Popup(PARCEL_LABEL, max_width=250),
            tooltip=PARCEL_LABEL,
            icon=folium.Icon(color="blue", icon="home", prefix="fa"),
        ).add_to(fmap)

        # Distance rings
        if show_rings:
            rings_fg = folium.FeatureGroup(name="Distance rings", show=True)
            for radius_mi in RING_RADII_MI:
                folium.Circle(
                    location=[PARCEL_LAT, PARCEL_LON],
                    radius=radius_mi * METERS_PER_MILE,
                    color="#333333", weight=1, dash_array="6,6", fill=False,
                    tooltip=f"{radius_mi} mi",
                ).add_to(rings_fg)
            rings_fg.add_to(fmap)

        # Fires — grouped by decade
        decade_groups: dict[int, folium.FeatureGroup] = {}

        selected_bounds: list[tuple[float, float]] = []  # collect lat/lon for fit_bounds

        for _, row in filtered.iterrows():
            rings = _parse_rings(row["geometry_geojson"])
            if not rings:
                continue

            year = int(row["year"]) if pd.notna(row["year"]) else 0
            acres = float(row["acres"]) if pd.notna(row["acres"]) else 0.0
            dist_mi = float(row["dist_mi"]) if pd.notna(row["dist_mi"]) else 0.0
            incident = str(row["incident"]) if pd.notna(row["incident"]) else "Unknown"

            # Highlight the selected fire from the table click
            is_selected = (
                selected_fire is not None
                and incident == selected_fire["incident"]
                and year == selected_fire["year"]
            )

            base_color = (year_cmap(year) if 1984 <= year <= 2025 else "#888888") \
                          if color_mode.startswith("Year") else _size_color(acres)

            decade = (year // 10) * 10 if year else 0
            if decade not in decade_groups:
                decade_groups[decade] = folium.FeatureGroup(name=f"{decade}s", show=True)

            tooltip = (
                f"<b>{incident}</b><br>{year} • {acres:,.0f} acres • "
                f"{dist_mi:.1f} mi from parcel"
                + (" • <b>SELECTED</b>" if is_selected else "")
            )

            if show_perims:
                # Full polygon view — render ALL rings (Caldor has 60 disconnected
                # rings, only ring[0] was visible before, which was a 100m sliver).
                # folium.Polygon's `locations` accepts a list-of-rings for multi-polygon.
                folium.Polygon(
                    locations=rings,
                    color="#FFB300" if is_selected else base_color,
                    weight=5 if is_selected else 1.5,
                    fill=True,
                    fill_color=base_color,
                    fill_opacity=0.85 if is_selected else 0.45,
                    tooltip=tooltip,
                ).add_to(decade_groups[decade])
            else:
                clat, clon = _centroid_of(rings)
                radius_px = 4 + min(12, (acres ** 0.5) / 50)
                folium.CircleMarker(
                    location=[clat, clon],
                    radius=radius_px * (1.6 if is_selected else 1.0),
                    color="#FFB300" if is_selected else base_color,
                    weight=4 if is_selected else 1.5,
                    fill=True, fill_color=base_color,
                    fill_opacity=0.95 if is_selected else 0.85,
                    tooltip=tooltip,
                    popup=folium.Popup(
                        f"<b>{incident}</b><br>Year: {year}<br>"
                        f"{acres:,.0f} acres<br>{dist_mi:.1f} mi away<br>"
                        f"<i>Click \"Show full fire perimeters\" in the sidebar to see the polygon.</i>",
                        max_width=260,
                    ),
                ).add_to(decade_groups[decade])

            # Collect bounds for the selected fire so we can fit-zoom to it
            if is_selected:
                for ring in rings:
                    selected_bounds.extend(ring)

        for decade in sorted(decade_groups.keys()):
            decade_groups[decade].add_to(fmap)

        if color_mode.startswith("Year"):
            year_cmap.add_to(fmap)

        folium.LayerControl(collapsed=False).add_to(fmap)

        # If the user clicked a row, fit map to that fire's bounds
        if selected_bounds:
            lats = [p[0] for p in selected_bounds]
            lons = [p[1] for p in selected_bounds]
            # Pad bounds slightly so the polygon isn't right at the viewport edges
            pad = 0.02
            fmap.fit_bounds(
                [[min(lats) - pad, min(lons) - pad],
                 [max(lats) + pad, max(lons) + pad]]
            )
            st.info(
                f"📍 Showing **{selected_fire['incident']}** ({selected_fire['year']}). "
                "Use the year/size filters above to clear the focus, or click the same row again to deselect."
            )

        # Use a stable-but-selection-aware key so the map remounts when the
        # selected fire changes (otherwise folium caches the old fit_bounds).
        map_key = f"firemap-{selected_fire['incident']}-{selected_fire['year']}" if selected_fire else "firemap"
        st_folium(fmap, width=1400, height=700, returned_objects=[], key=map_key)

        # ---------- Summary metrics ----------
        st.subheader("Summary of fires shown")

        total_fires = len(filtered)
        biggest = filtered.loc[filtered["acres"].idxmax()] if total_fires else None
        closest = filtered.loc[filtered["dist_mi"].idxmin()] if total_fires else None
        mega_years = sorted(filtered.loc[filtered["acres"] >= 10_000, "year"].dropna().unique().tolist())

        # Use 4 short-value metrics so labels don't truncate. Names + details go in the
        # table below where there's full width.
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Fires shown", f"{total_fires:,}")
        if biggest is not None:
            c2.metric(
                "Biggest fire (acres)",
                f"{biggest['acres']:,.0f}",
                f"{biggest['incident']} • {int(biggest['year'])}",
            )
        if closest is not None:
            c3.metric(
                "Closest fire (mi)",
                f"{closest['dist_mi']:.1f}",
                f"{closest['incident']} • {int(closest['year'])}",
            )
        c4.metric(
            "Years w/ ≥10K-acre fires",
            f"{len(mega_years)}",
            ", ".join(str(int(y)) for y in mega_years[-4:]) if mega_years else "none",
        )

        if total_fires:
            decade_df = (
                filtered.assign(decade=(filtered["year"] // 10 * 10).astype(int).astype(str) + "s")
                .groupby("decade").size().rename("fires").reset_index().sort_values("decade")
            )
            st.subheader("Fires by decade")
            st.bar_chart(decade_df, x="decade", y="fires", height=240)

            # All fires table — sorted by most recent first (built earlier as `fire_table`)
            st.subheader(f"All {total_fires} fires shown (most recent first)")
            st.caption("💡  Click a row to zoom the map to that fire and highlight it. Click the same row again to deselect.")
            display_table = pd.DataFrame({
                "Year":                       fire_table["year"].astype(int),
                "Incident":                   fire_table["incident"].fillna("(unnamed)"),
                "Size (acres)":               fire_table["acres"].round(0).astype(int),
                "Distance from parcel (mi)":  fire_table["dist_mi"].round(1),
            })
            st.dataframe(
                display_table,
                hide_index=True,
                height=400,
                on_select="rerun",
                selection_mode="single-row",
                key="fire_table_select",
            )

    st.caption(
        "Data: NIFC Interagency Fire Perimeter History (1984 – present). "
        "Distances are great-circle from the parcel centroid."
    )


# ---------------------------------------------------------------------------
# TAB 2 (new) — Historical destruction near the parcel
# ---------------------------------------------------------------------------
with tab_history:
    st.title("What 41 years of wildfire actually destroyed within 20 miles of this parcel")
    st.caption(
        "Verified from CAL FIRE DINS (structure-level inspections, 2013+) and published incident "
        "reports for older fires. This is the empirical reality the model is calibrated against."
    )

    # ---- Headline numbers ----
    h1, h2, h3 = st.columns(3)
    h1.metric("Structures destroyed within 20 mi", "≈ 640", "1984–2025 (41 years)")
    h2.metric("Driven by just 3 fires", "98%", "Caldor + Angora + Tamarack")
    h3.metric("Avg annual destruction rate", "≈ 0.026% / yr",
              "of all ~60K housing units in the 20-mi circle")

    st.divider()

    st.markdown("##### The three fires that destroyed almost everything")
    incidents = pd.DataFrame([
        {
            "Fire": "Caldor",
            "Year": "2021",
            "Distance from parcel": "9.9–20 mi (Grizzly Flats area, south edge of 20-mi circle)",
            "Structures destroyed within 20 mi": "318",
            "Source": "CAL FIRE DINS (lat/lon-filtered, 4,353 inspections in El Dorado County)",
        },
        {
            "Fire": "Angora",
            "Year": "2007",
            "Distance from parcel": "7–10 mi (Meyers / Tahoe Mountain Rd, South Lake Tahoe)",
            "Structures destroyed within 20 mi": "309",
            "Source": "CAL FIRE published / Wikipedia (242 homes + 67 commercial)",
        },
        {
            "Fire": "Tamarack",
            "Year": "2021",
            "Distance from parcel": "14–20 mi (Markleeville, Alpine County)",
            "Structures destroyed within 20 mi": "13",
            "Source": "CAL FIRE DINS (lat/lon-filtered; 10 more destroyed beyond 20 mi)",
        },
    ])
    st.dataframe(incidents, hide_index=True)

    st.markdown("##### Other named fires within 20 mi (1984–2025)")
    st.markdown(
        """
        ~55 other named fires ≥100 acres burned within 20 mi of the parcel during this period.
        Almost none destroyed primary residences. Notable smaller losses:

        - **Voltaire 2018** (Douglas NV, 11.9 mi): a few outbuildings only
        - **Numbers 2020** (Douglas NV, 14 mi): no residences destroyed
        - **Gondola 2002** (0.6 mi from parcel!): burned in Heavenly ski area — no structures
        - **Autumn Hills 1996** (1.4 mi): burned rangeland east of Genoa — no residences

        Net total from these: estimated **5–10 small structures (mostly outbuildings)**.
        """
    )

    st.divider()

    st.markdown("##### Normalized per-structure rates — what does this mean per home?")

    rate_table = pd.DataFrame([
        {
            "Population": "All housing in 20-mi circle (Stateline NV + South Lake Tahoe + Carson Valley + Markleeville + Grizzly Flats etc.)",
            "Approx housing units": "~60,000",
            "Total destroyed (41 yrs)": "~640",
            "Annual rate per structure": "0.026% / yr",
            "Cumulative over 15 yrs": "~0.4%",
        },
        {
            "Population": "WUI-edge structures only (the buildings actually next to burnable fuel)",
            "Approx housing units": "~12,000",
            "Total destroyed (41 yrs)": "~640",
            "Annual rate per structure": "0.130% / yr",
            "Cumulative over 15 yrs": "~1.9%",
        },
        {
            "Population": "**This specific parcel** (developed-cell condo, ½ mi from Heavenly, low-end WUI)",
            "Approx housing units": "1",
            "Total destroyed (41 yrs)": "0 so far",
            "Annual rate per structure": "—",
            "Cumulative over 15 yrs": "**Model says 4.1%** (95% range 0.8–15%)",
        },
    ])
    st.dataframe(rate_table, hide_index=True)

    st.info(
        """
        **Why the model number (4.1%) is higher than the WUI-edge empirical average (1.9%):**

        1. **Climate trend forward projection** — the model assumes +3%/yr hazard growth through 2041 from
           Cal-Adapt CMIP5. The 41-yr empirical average doesn't extrapolate; future risk is plausibly higher.
        2. **Interior unit-fire risk** — the empirical wildfire-only data misses interior ignition (kitchen,
           electrical) that destroys multi-unit buildings independent of wildfire. This is its own ~0.09% / yr
           term for a 24-unit building.
        3. **Reality-check adjustment for partial-loss / cat-event correlation** — model bumped up vs raw
           historical to account for what insurance loss data implies (5–12× higher than wildfire alone).
        4. **The empirical average dilutes WUI-edge risk** by including all 12,000 WUI structures equally,
           when in reality the highest-risk subset (think Grizzly Flats) had ~90% destruction in Caldor.
        """
    )

    st.divider()
    st.markdown("##### Key takeaway")
    st.markdown(
        """
        Real wildfire destruction near this parcel over 41 years is a **3-event story**:
        Caldor 2021, Angora 2007, Tamarack 2021. Each was a once-in-a-generation regional disaster.
        The 640 destroyed structures clustered in specific WUI-edge neighborhoods (Grizzly Flats, Meyers,
        Markleeville outskirts) — none of which is exactly where 759 Boulder Ct sits.

        **This parcel's specific location (developed cell, ½ mile from Heavenly, surrounded by paved
        areas) is on the *less-vulnerable* edge of the WUI corridor.** But it remains within the broader
        fire-prone region. The model's 4.1% is meant to capture that mix.
        """
    )


# ---------------------------------------------------------------------------
# TAB 3 — What does 1-in-25 feel like? (calibration for a 35-year-old)
# ---------------------------------------------------------------------------
with tab_context:
    st.title("What does a 1-in-25 chance over 15 years actually feel like?")
    st.caption(
        "A 4% probability over 15 years (≈ 0.27% per year, ≈ 1 in 375 annually) is abstract. "
        "Here it is alongside the kinds of life events you already navigate as a 35-year-old."
    )

    st.info(
        """
        **The headline:** in 24 out of 25 simulated futures, none of this happens and the property
        pays off normally. In 1 out of 25, the building burns down. That ratio is comparable to many
        real-world risks people already accept — and several risks people accept are **substantially
        worse than this one.**
        """
    )

    st.divider()

    # ---------- HIGHER (more likely than 4%) ----------
    st.markdown("### 📈  Events MORE likely to happen to you over the next 15 years")
    st.caption("Risks you're already running without thinking about it. All > 4% probability for a 35-year-old.")

    higher = pd.DataFrame([
        {"Life event":"Major depression episode at least once",
         "Probability over 15 yrs":"~25%",
         "Domain":"Health",
         "Source":"NIMH / Lancet Psychiatry meta-analyses"},
        {"Life event":"Getting divorced (if you're married or marry soon)",
         "Probability over 15 yrs":"~25–30%",
         "Domain":"Relationships",
         "Source":"US Census / CDC NCHS"},
        {"Life event":"Significant financial loss to identity theft",
         "Probability over 15 yrs":"~25–35%",
         "Domain":"Crime/finance",
         "Source":"FTC / Javelin Strategy"},
        {"Life event":"A major recession affects your finances",
         "Probability over 15 yrs":"~25%",
         "Domain":"Economy",
         "Source":"NBER recession dating + S&P historical"},
        {"Life event":"Lose your job at least once",
         "Probability over 15 yrs":"~30–50%",
         "Domain":"Career",
         "Source":"BLS displaced workers survey"},
        {"Life event":"Lose a parent",
         "Probability over 15 yrs":"~40–60% (depends on parents' current age)",
         "Domain":"Family",
         "Source":"SSA life tables"},
        {"Life event":"Be hospitalized at least once",
         "Probability over 15 yrs":"~50%",
         "Domain":"Health",
         "Source":"HCUP / AHRQ"},
        {"Life event":"Experience a multi-day power outage",
         "Probability over 15 yrs":"~80%",
         "Domain":"Infrastructure",
         "Source":"EIA reliability statistics"},
        {"Life event":"Be in a reportable car accident",
         "Probability over 15 yrs":"~35–40%",
         "Domain":"Driving",
         "Source":"NHTSA Crash Stats"},
        {"Life event":"**Owning a home in a FEMA 100-year flood zone** (Special Flood Hazard Area, ~13M US homes)",
         "Probability over 15 yrs":"~14%",
         "Domain":"Property / weather",
         "Source":"FEMA NFIP — by definition 1%/yr flood risk; FEMA cites 26% over a 30-yr mortgage"},
        {"Life event":"**Owning a home in any moderate-to-high flood-risk zone** (per First Street Foundation)",
         "Probability over 15 yrs":"~10–25%",
         "Domain":"Property / weather",
         "Source":"First Street Foundation — climate-adjusted risk (broader than FEMA)"},
        {"Life event":"Develop Type 2 diabetes",
         "Probability over 15 yrs":"~8–12% (higher with weight/family history)",
         "Domain":"Health",
         "Source":"ADA / CDC diabetes statistics"},
        {"Life event":"Get scammed online for >$1K",
         "Probability over 15 yrs":"~15–20%",
         "Domain":"Crime/finance",
         "Source":"FTC Consumer Sentinel"},
        {"Life event":"Experience a major mental-health event in a close friend / family member",
         "Probability over 15 yrs":"~60%+",
         "Domain":"Social",
         "Source":"NAMI prevalence stats"},
    ])
    st.dataframe(higher, hide_index=True)

    st.divider()

    # ---------- SIMILAR (roughly comparable to 4%) ----------
    st.markdown("### 🎯  Events ROUGHLY AS likely as a fire here (3–6% over 15 years)")
    st.caption("These are the closest gut-feel comparisons to your specific 4% risk.")

    similar = pd.DataFrame([
        {"Life event":"**Your own death between age 35 and 50** (US average, all causes)",
         "Probability over 15 yrs":"~2.5–3%",
         "Domain":"Mortality",
         "Notes":"CDC life tables. Slightly lower than this fire risk — and you don't reorganize your life around the 2.5% chance you don't make it to 50."},
        {"Life event":"Be diagnosed with cancer between age 35 and 50",
         "Probability over 15 yrs":"~3–5%",
         "Domain":"Health",
         "Notes":"NCI SEER incidence rate. Cancer becomes much more common past 50; this is screening-window territory."},
        {"Life event":"Total your car (or have it totaled) in a crash",
         "Probability over 15 yrs":"~3–5%",
         "Domain":"Driving",
         "Notes":"Per NHTSA + Insurance Institute. You drive every day knowing this; you have car insurance for it."},
        {"Life event":"Have a house fire of any size",
         "Probability over 15 yrs":"~4–5%",
         "Domain":"Housing",
         "Notes":"NFPA US national average across all homes. The exact calibration — your parcel's wildfire-total-loss probability ≈ the average US home's chance of ANY fire."},
        {"Life event":"**A 500-year flood hits a specific address** (anywhere in the US)",
         "Probability over 15 yrs":"~3%",
         "Domain":"Property / weather",
         "Notes":"By definition: 0.2%/yr → 1−(1−0.002)^15. About the same magnitude as your wildfire risk. People living in 500-year flood zones don't typically buy flood insurance — they accept the risk."},
        {"Life event":"Get into a top-50 grad school you applied to",
         "Probability over 15 yrs":"~5–15%",
         "Domain":"Education",
         "Notes":"Top program admit rates. Comparable order of magnitude — if you've ever applied to a competitive program, you've already 'bought' a 4%-class outcome."},
        {"Life event":"Have a serious bicycle accident requiring ER visit (if regular cyclist)",
         "Probability over 15 yrs":"~5–8%",
         "Domain":"Recreation",
         "Notes":"CDC bike-injury data for adult cyclists."},
        {"Life event":"Need major surgery (anything beyond minor outpatient)",
         "Probability over 15 yrs":"~5–10%",
         "Domain":"Health",
         "Notes":"AHRQ surgical-procedure rates. Hernia, appendix, ACL, gallbladder — common."},
        {"Life event":"Have a Tahoe ski accident needing medical evacuation (if regular skier)",
         "Probability over 15 yrs":"~5–10%",
         "Domain":"Recreation",
         "Notes":"NSAA / Heavenly local data. If you're buying near Heavenly, you might be skiing here too."},
    ])
    st.dataframe(similar, hide_index=True)

    st.divider()

    # ---------- LOWER (less likely than 4%) ----------
    st.markdown("### 📉  Events LESS likely to happen to you over the next 15 years")
    st.caption("Risks people often worry about that are actually less probable than this fire risk.")

    lower = pd.DataFrame([
        {"Life event":"Die in a car crash",
         "Probability over 15 yrs":"~0.17%",
         "Notes":"NHTSA US average. You drive every day knowing this — it's 23× less likely than your fire risk."},
        {"Life event":"Die in a plane crash on commercial flights (regular traveler)",
         "Probability over 15 yrs":"~0.0001%",
         "Notes":"You probably fly a few times a year. ~40,000× less likely than your fire risk."},
        {"Life event":"Specific home destroyed by tornado (US average, even in Tornado Alley)",
         "Probability over 15 yrs":"~0.3%",
         "Notes":"Despite tornado damage being culturally salient, total destruction at a specific address is rare."},
        {"Life event":"Get struck by lightning in your lifetime",
         "Probability over 15 yrs":"~0.007%",
         "Notes":"~600× less likely than your fire risk."},
        {"Life event":"Murdered (US average for 35-yr-old)",
         "Probability over 15 yrs":"~0.07%",
         "Notes":"~55× less likely than your fire risk, despite cultural prominence."},
        {"Life event":"Have a passport stolen abroad needing emergency replacement (frequent traveler)",
         "Probability over 15 yrs":"~1–2%",
         "Notes":"You probably travel internationally; you don't fret about this most of the time."},
        {"Life event":"Win at least $10K in a single lottery purchase",
         "Probability over 15 yrs":"~0.0001%",
         "Notes":"~40,000× less likely. Yet many people 'invest' in this regularly."},
        {"Life event":"Get attacked by a shark while ocean-swimming",
         "Probability over 15 yrs":"~0.00001%",
         "Notes":"You swim in oceans without a second thought."},
    ])
    st.dataframe(lower, hide_index=True)

    st.divider()

    # ---------- The intuition pump ----------
    st.markdown("### 🧠  The framing that probably helps most")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            **You're 35. Your own 15-year mortality risk is about 2.5–3%.**

            That's *lower* than this fire risk. You don't reorganize your life around the chance
            you don't make it to 50 — you accept it and plan for the other 97% of futures.

            **You drive every day knowing you have a 35–40% chance of being in a crash in the next 15 years.**

            That's *10× more likely* than this fire risk. You manage it with insurance + reasonable
            driving habits, and the decision to keep driving isn't even a question.
            """
        )

    with col2:
        st.markdown(
            """
            **The cancer screening you'd get at 50 is for a ~5% probability over the prior 15 years.**

            That's basically the same probability we're discussing here. You wouldn't say "the chance
            is too high to live" — you'd say "let's monitor, plan, and act if it happens."

            **You apply for jobs / grad school / promotions with worse base rates and don't blink.**

            A 4% probability of total fire loss isn't a *rare extreme event* — it's a normal-magnitude
            life risk that deserves planning, not panic.
            """
        )

    st.success(
        """
        ✅  **The point isn't that 4% is small.** It's that 4% is a probability you're already
        comfortable navigating in many domains of your life. The decision for this purchase is whether
        you can navigate it **here**, given the financial structure shown in the next tab — not whether
        4% is too scary in the abstract.
        """
    )

    st.divider()

    # ---------- Flood vs wildfire — the closest property-risk parallel ----------
    st.markdown("### 🌊  The closest direct parallel: flood-zone real estate")
    st.markdown(
        """
        If you want a property-risk analogy that maps almost exactly onto your situation,
        **owning a home in a coastal or river flood zone** is the cleanest one. The mechanics are
        nearly identical to wildfire:
        """
    )

    flood_compare = pd.DataFrame([
        {
            "Dimension": "Catastrophic-event probability",
            "Coastal / river flood-zone home": "~14% over 15 yrs (100-yr SFHA) to ~25% (high-risk per First Street)",
            "Your wildfire situation": "~4% over 15 yrs (this parcel)",
        },
        {
            "Dimension": "Loss mechanism",
            "Coastal / river flood-zone home": "Storm surge / atmospheric river / river overflow destroys structure",
            "Your wildfire situation": "Embers / direct flame / interior fire destroys structure",
        },
        {
            "Dimension": "Typical loss magnitude",
            "Coastal / river flood-zone home": "$30K–$200K (most claims); rare total loss",
            "Your wildfire situation": "$0 (no event) or $334K (total loss); binary",
        },
        {
            "Dimension": "Insurance availability",
            "Coastal / river flood-zone home": "Available via NFIP at federally-subsidized rates; private market patchy",
            "Your wildfire situation": "Essentially unavailable; HOA quote was $24K/yr — voted down",
        },
        {
            "Dimension": "Resale-value impact (current trend)",
            "Coastal / river flood-zone home": "Climate-adjusted pricing emerging; coastal premiums shrinking 5–15%",
            "Your wildfire situation": "WUI properties starting to trade 10–30% below comparable insured properties",
        },
        {
            "Dimension": "Why people buy anyway",
            "Coastal / river flood-zone home": "Lifestyle (beach, river access, view), STR income, appreciation history, accept the tail",
            "Your wildfire situation": "Lifestyle (Tahoe / Heavenly), STR income, appreciation history, accept the tail",
        },
    ])
    st.dataframe(flood_compare, hide_index=True)

    st.markdown(
        """
        **The takeaway from the flood comparison:**

        - **~13 million US homes sit in FEMA-designated flood zones.** People buy them every day,
          knowing their probability of catastrophic damage over a 15-year hold is actually **higher**
          than your wildfire risk here.
        - **Flood-zone buyers manage the risk through insurance (NFIP) + financial discipline.** You can't
          fully replicate that here because the wildfire insurance market for this building is closed —
          but the financial discipline part (cash reserves, mortgage structure, tax-deductible business
          loss path) still applies.
        - **The decision framework is the same one millions of coastal homeowners run every year.**
          You're not in unprecedented decision territory; you're in a well-studied category of risk.
        """
    )

    st.caption(
        "Sources: CDC NCHS life tables, NCI SEER cancer statistics, NHTSA crash data, NFPA fire statistics, "
        "ADA/CDC diabetes statistics, FTC Consumer Sentinel, BLS displaced workers survey, Social Security "
        "Administration life tables, NIMH mental health prevalence, EIA reliability statistics."
    )


# ---------------------------------------------------------------------------
# TAB 4 — Should you buy?
# ---------------------------------------------------------------------------
with tab_decision:
    st.title("Should You Buy This Property?")
    st.caption(
        "**Property:** 759 Boulder Ct, Stateline NV  •  **Building type:** "
        "1980 wood-frame multi-unit condo  •  **Your share of structural value:** \$334,000"
    )

    # Pull the headline numbers from the decision table
    if not decision_df.empty:
        d5  = decision_df.loc[decision_df["horizon_yrs"] == 5].iloc[0]  if (decision_df["horizon_yrs"] == 5).any()  else None
        d10 = decision_df.loc[decision_df["horizon_yrs"] == 10].iloc[0] if (decision_df["horizon_yrs"] == 10).any() else None
        d15 = decision_df.loc[decision_df["horizon_yrs"] == 15].iloc[0] if (decision_df["horizon_yrs"] == 15).any() else None
    else:
        d5 = d10 = d15 = None

    # =========================================================================
    # SECTION 1 — FIRE RISK FOR THIS PARCEL
    # =========================================================================
    st.header("1.  Fire risk for this parcel")

    if d15 is not None:
        p15      = float(d15["p_loss_median"])
        p15_low  = float(d15["p_loss_p5"])
        p15_high = float(d15["p_loss_p95"])
        one_in   = int(round(1 / max(p15, 1e-6)))
        st.info(
            f"""
            **The empirical read.** Within 50 miles of the parcel since 1984, **40 of 41 years
            had a fire ≥100 acres** — including fires within 0.6 miles (Gondola 2002), 1.4 miles
            (Autumn Hills 1996), 6 miles (Caldor 2021, ~1,000 structures destroyed), and 7.9 miles
            (Angora 2007, 254 homes destroyed). This is a high-WUI parcel, full stop.

            Combining real fire history with building-vulnerability data, the chance the building
            is destroyed over **15 years is about {p15*100:.1f}% (1 in {one_in})**, with a defensible
            range of {p15_low*100:.1f}%–{p15_high*100:.1f}%.
            """
        )

    if d5 is not None and d10 is not None and d15 is not None:
        st.markdown("##### Chance the building is destroyed at each horizon")
        c1, c2, c3 = st.columns(3)
        c1.metric("Within 5 years",  f"{float(d5['p_loss_median'])*100:.1f}%",
                  f"1 in {int(round(1/max(float(d5['p_loss_median']), 1e-6)))}")
        c2.metric("Within 10 years", f"{float(d10['p_loss_median'])*100:.1f}%",
                  f"1 in {int(round(1/max(float(d10['p_loss_median']), 1e-6)))}")
        c3.metric("Within 15 years", f"{float(d15['p_loss_median'])*100:.1f}%",
                  f"1 in {int(round(1/max(float(d15['p_loss_median']), 1e-6)))}")

    st.markdown("##### The fire-history map")
    st.image("images/wildfire_history_closeup.png",
             caption="Every shape is a fire ≥100 acres within 20 miles of the parcel since 1984. Caldor 2021 (6 mi), Angora 2007 (7.9 mi, destroyed 254 homes), and Gondola 2002 (0.6 mi) are labeled.")
    st.image("images/wildfire_history_map.png",
             caption="Empirical fire frequency by distance — within 5 mi, fires occurred in 7% of years; within 10 mi, 17%. The blue marker shows what the official US model says (0.04%/yr) — ~150× lower than reality.")

    with st.expander("Why the official US fire model dramatically understates this parcel"):
        st.markdown(
            """
            The USFS "Wildfire Risk to Communities" model says this exact parcel has only **0.04%
            annual fire chance** — dramatically lower than 41 years of empirical history shows.
            Three reasons:

            1. **The parcel sits on a "developed" pixel** in the model's fuel map. The model fills in
               fire risk for developed cells by averaging surrounding burnable cells, diluting the real risk.
            2. **The model uses 2020 fuel data** — before Caldor 2021 and the recent Sierra fire escalation.
            3. **It doesn't model human-caused ignitions** (power lines, vehicles, campfires) — which
               drive most actual fires in the wildland-urban interface.

            We use empirical fire history, not this model, as the wildfire anchor in our analysis.
            """
        )

    st.divider()

    # =========================================================================
    # SECTION 2 — WHAT WE MODELED AND WHAT EACH MODEL SAID
    # =========================================================================
    st.header("2.  What we modeled and what each model said")

    st.markdown(
        "The 15-year total-loss probability above is the combination of four sub-models. "
        "Each captures a different mechanism by which the building could be destroyed. "
        "Reading them separately keeps the assumptions transparent."
    )

    model_table = pd.DataFrame([
        {
            "Model":       "Wildfire reaches building",
            "What it covers": "How often a wildfire physically burns through the parcel's cell",
            "Data source":  "USFS WRC FSim + NIFC fire-perimeter history (1984–2025, 394 fires)",
            "What it said":   "Official model: 0.04%/yr  • Empirical reality: ~0.35%/yr (~150× higher)",
            "Why it matters": "Anchors the chance fire arrives at the building",
        },
        {
            "Model":       "Building destroyed | fire arrives",
            "What it covers": "If fire reaches the parcel, what % of similar buildings are destroyed",
            "Data source":  "CAL FIRE DINS (132K real post-fire structure inspections)",
            "What it said":   "27–35% for 1980-era multi-family in WUI, depending on hardening",
            "Why it matters": "Building's resilience given the building gets fire-tested",
        },
        {
            "Model":       "Fire starts inside a unit",
            "What it covers": "A kitchen/electrical/etc. fire in any unit spreading to shell",
            "Data source":  "NFPA US residential fire statistics, decomposed for multi-family",
            "What it said":   "~0.03%/yr for the whole building (small but not zero)",
            "Why it matters": "Captures non-wildfire loss path — relevant 365 days/yr",
        },
        {
            "Model":       "Climate trend",
            "What it covers": "How wildfire hazard grows year-over-year through 2041",
            "Data source":  "Cal-Adapt LOCA CMIP5 ensemble (8 GCMs × 2 emission scenarios)",
            "What it said":   "+3%/yr relative hazard growth (median across ensemble)",
            "Why it matters": "Year-15 risk is ~50% higher than year-1 risk in this trend",
        },
    ])
    st.dataframe(model_table, hide_index=True)

    st.markdown("##### Combined output — risk-adjusted dollar exposure")
    st.caption("These dollar values are the *building shell* you stand to lose if the fire actually happens. The next section translates this into actual cash impact given your loan structure and tax treatment.")
    if not decision_df.empty:
        display = decision_df.assign(
            **{
                "Holding period":             decision_df["horizon_yrs"].astype(int).astype(str) + " years",
                "Chance of total loss":       (decision_df["p_loss_median"] * 100).round(2).astype(str) + "%",
                "Range (5–95%)":              (decision_df["p_loss_p5"] * 100).round(2).astype(str) + "% – " + (decision_df["p_loss_p95"] * 100).round(2).astype(str) + "%",
                "Average gross $ loss":       decision_df["e_loss_usd"].apply(lambda x: f"\${x:,.0f}"),
                "Worst-case gross $ loss":    decision_df["p99_loss_usd"].apply(lambda x: f"\${x:,.0f}"),
            }
        )[["Holding period", "Chance of total loss", "Range (5–95%)", "Average gross $ loss", "Worst-case gross $ loss"]]
        st.dataframe(display, hide_index=True)

    with st.expander("Honest limits of these models"):
        st.markdown(
            """
            Every model is wrong in some way. Here's what could change the answer:

            1. **How many units are in the building?** We assumed 4–16. If 20+, interior-fire risk
               rises proportionally.
            2. **Will mitigation hold?** We assume the retired-firefighter manager, sprinklers, and
               defensible space stay in place in ~70% of scenarios. If HOA discipline lapses, risk
               goes up.
            3. **California → Nevada transfer.** Our most detailed building-loss data is CA-only.
               Nevada fire physics is similar; codes and wind patterns differ at the margin.
            4. **Climate scenario.** We use CMIP5; newer CMIP6 projects slightly worse warming.
               Our trend is on the conservative side.
            """
        )

    st.divider()

    # =========================================================================
    # SECTION 3 — WHAT THIS MEANS FOR YOUR BUYING DECISION
    # =========================================================================
    st.header("3.  What this means for your buying decision")

    # ===== Property + financing setup =====
    PURCHASE_PRICE   = 420_000
    DOWN_PCT         = 0.15
    CLOSING_PCT      = 0.025
    LOAN_RATE        = 0.07
    LOAN_YEARS_TERM  = 30
    APPR_PER_YEAR    = 0.03
    LAND_FRACTION    = 0.20            # condo land fraction (non-depreciable)
    DEPREC_LIFE      = 27.5            # IRS Schedule E rental
    MARGINAL_TAX     = 0.32            # fed (NV has no state income tax)

    BUILDING_UNITS   = 24              # updated per HOA

    # STR income assumptions (AirROI Stateline market)
    STR_GROSS_REV    = 42_000          # avg annual gross at default occupancy
    STR_REV_GROWTH   = 0.02            # 2%/yr revenue growth
    PROP_MGMT_PCT    = 0.25            # 25% mgmt fee
    HOA_MONTHLY      = 500
    PROP_TAX_PCT     = 0.007           # 0.7% NV
    MAINT_PCT        = 0.01            # 1%/yr of property value

    cash_to_close   = PURCHASE_PRICE * (DOWN_PCT + CLOSING_PCT)
    down_amount     = PURCHASE_PRICE * DOWN_PCT
    closing_amount  = PURCHASE_PRICE * CLOSING_PCT
    loan_amount     = PURCHASE_PRICE * (1 - DOWN_PCT)
    monthly_pmt     = (loan_amount * (LOAN_RATE/12) *
                       (1 + LOAN_RATE/12) ** (LOAN_YEARS_TERM*12)) / \
                       ((1 + LOAN_RATE/12) ** (LOAN_YEARS_TERM*12) - 1)
    annual_PI       = monthly_pmt * 12
    land_value      = PURCHASE_PRICE * LAND_FRACTION
    depreciable     = PURCHASE_PRICE * (1 - LAND_FRACTION)
    annual_deprec   = depreciable / DEPREC_LIFE

    def mortgage_balance(years):
        if years >= LOAN_YEARS_TERM: return 0.0
        r = LOAN_RATE / 12
        k = years * 12
        return loan_amount * (1+r)**k - monthly_pmt * ((1+r)**k - 1) / r

    def mortgage_interest_for_year(year_y):
        """Approximate interest paid in year `year_y` (year 1 = first full year)."""
        bal_start = mortgage_balance(year_y - 1)
        bal_end   = mortgage_balance(year_y)
        principal = bal_start - bal_end
        return annual_PI - principal

    def annual_cash_flow(year_y):
        """Pre-tax operating CF for year y (negative = you put money in, positive = pocket)."""
        rev_gross     = STR_GROSS_REV * (1 + STR_REV_GROWTH) ** (year_y - 1)
        rev_net_mgmt  = rev_gross * (1 - PROP_MGMT_PCT)
        prop_val_y    = PURCHASE_PRICE * (1 + APPR_PER_YEAR) ** (year_y - 1)
        operating     = (HOA_MONTHLY * 12) + prop_val_y * PROP_TAX_PCT + prop_val_y * MAINT_PCT
        return rev_net_mgmt - operating - annual_PI

    def after_tax_cf(year_y):
        """Cash flow with tax savings from rental loss (depreciation + interest deductions)."""
        rev_gross     = STR_GROSS_REV * (1 + STR_REV_GROWTH) ** (year_y - 1)
        prop_val_y    = PURCHASE_PRICE * (1 + APPR_PER_YEAR) ** (year_y - 1)
        mgmt_fee      = rev_gross * PROP_MGMT_PCT
        hoa           = HOA_MONTHLY * 12
        prop_tax      = prop_val_y * PROP_TAX_PCT
        maint         = prop_val_y * MAINT_PCT
        interest      = mortgage_interest_for_year(year_y)
        deductible_expenses = mgmt_fee + hoa + prop_tax + maint + interest + annual_deprec
        rental_taxable = rev_gross - deductible_expenses
        tax_impact    = rental_taxable * MARGINAL_TAX   # negative = tax savings
        # Pre-tax cash flow = rev - cash expenses (NOT including depreciation, NOT including principal pmt)
        principal     = annual_PI - interest
        pretax_cf     = (rev_gross - mgmt_fee) - hoa - prop_tax - maint - annual_PI
        return {
            "pretax_cf":      pretax_cf,
            "tax_impact":     tax_impact,
            "after_tax_cf":   pretax_cf - tax_impact,   # tax_impact negative for loss → adds back
            "rental_taxable": rental_taxable,
        }

    # ===== Show financing setup =====
    st.markdown("#### Your purchase setup (recomputed for the 24-unit building)")

    setup_table = pd.DataFrame([
        {"Item": "Purchase price",                        "Amount": f"\${PURCHASE_PRICE:,.0f}"},
        {"Item": "Down payment (15%)",                     "Amount": f"\${down_amount:,.0f}"},
        {"Item": "Closing costs (~2.5%)",                  "Amount": f"\${closing_amount:,.0f}"},
        {"Item": "**Total cash to close**",                "Amount": f"**\${cash_to_close:,.0f}**"},
        {"Item": "Loan amount (30-yr fixed)",              "Amount": f"\${loan_amount:,.0f}"},
        {"Item": "Mortgage rate",                          "Amount": f"{LOAN_RATE*100:.1f}%"},
        {"Item": "Monthly P&I payment",                    "Amount": f"\${monthly_pmt:,.0f}"},
        {"Item": "Annual P&I",                              "Amount": f"\${annual_PI:,.0f}"},
        {"Item": "Annual depreciation (tax shield)",       "Amount": f"\${annual_deprec:,.0f}"},
        {"Item": "Building units (for interior-fire risk)","Amount": f"{BUILDING_UNITS} units"},
    ])
    st.dataframe(setup_table, hide_index=True)

    # ===== Year-by-year cash flow WITH STR income =====
    st.markdown("#### Year-by-year cash flow — with STR income offsetting carrying costs")
    st.caption(
        "These are the numbers if no fire happens. They show the cash you actually put in (or take out) "
        "each year given STR income, operating costs, mortgage, and the rental tax write-off."
    )

    cf_rows = []
    cum_after_tax = 0
    cum_str_gross = 0
    for yr in [1, 2, 3, 5, 10, 15]:
        cf = after_tax_cf(yr)
        rev_y = STR_GROSS_REV * (1 + STR_REV_GROWTH) ** (yr - 1)
        cum_str_gross += rev_y if yr <= 3 else 0  # rough
        cf_rows.append({
            "Year":                       f"Year {yr}",
            "STR gross revenue":           f"\${rev_y:,.0f}",
            "Operating cash flow (pre-tax)": f"\${cf['pretax_cf']:,.0f}",
            "Tax savings (rental loss × {:.0f}%)".format(MARGINAL_TAX*100): f"\${-cf['tax_impact']:,.0f}",
            "After-tax cash flow":         f"\${cf['after_tax_cf']:,.0f}",
        })
    st.dataframe(pd.DataFrame(cf_rows), hide_index=True)

    # Cumulative cash spent through 5/10/15
    def cum_cash_to_year(Y):
        s = -cash_to_close
        for y in range(1, Y+1):
            s += after_tax_cf(y)["after_tax_cf"]
        return s
    cum5, cum10, cum15 = cum_cash_to_year(5), cum_cash_to_year(10), cum_cash_to_year(15)

    cum_table = pd.DataFrame([
        {"By end of": "Year 5",  "Cumulative cash you've put in": f"\${-cum5:,.0f}"},
        {"By end of": "Year 10", "Cumulative cash you've put in": f"\${-cum10:,.0f}"},
        {"By end of": "Year 15", "Cumulative cash you've put in": f"\${-cum15:,.0f}"},
    ])
    st.markdown("**Cumulative net cash you've contributed (no-fire path):**")
    st.dataframe(cum_table, hide_index=True)
    st.caption(
        f"i.e., on top of the \${cash_to_close:,.0f} cash to close, the STR income roughly covers "
        f"operating costs and mortgage in later years — net cumulative spend is modest. The wealth "
        f"build is in the **equity** (mortgage paydown + appreciation), realized at sale."
    )

    # ===== Fire scenarios — payment timing =====
    st.markdown("#### If a fire totals the building — what you actually pay and when")
    st.caption(
        "The \$334K shell is the worst-case **gross** loss to the structure. What you'd actually pay "
        "(and when) depends on whether you settle the mortgage or default. Both paths shown below."
    )

    def fire_scenario(y):
        prop_value      = PURCHASE_PRICE * (1 + APPR_PER_YEAR) ** y
        mort_bal        = mortgage_balance(y)
        equity          = max(prop_value - mort_bal, 0)
        # Cash already put in cumulatively by year y
        cash_already    = -cum_cash_to_year(y)
        # Adjusted basis remaining (land + remaining building basis)
        adj_basis       = land_value + max(depreciable - annual_deprec * y, 0)
        # Casualty loss for tax (uninsured): basis - residual land value (assume 20% retained)
        casualty_loss   = adj_basis - 0.20 * land_value
        tax_recovery    = casualty_loss * MARGINAL_TAX
        # 1099-C income if default (forgiven debt minus residual land value)
        forgiven_debt_taxable = max(mort_bal - 0.20 * land_value, 0) * MARGINAL_TAX

        return {
            "y": y,
            "prop_value":    prop_value,
            "mort_bal":      mort_bal,
            "equity":        equity,
            "cash_already":  cash_already,
            "tax_recovery":  tax_recovery,
            "forgiven_tax":  forgiven_debt_taxable,
            "settle_path_net_hit":  cash_already + mort_bal - 0.20*land_value - tax_recovery,
            "default_path_net_hit": cash_already + forgiven_debt_taxable - tax_recovery,
        }

    fires = [fire_scenario(y) for y in (5, 10, 15)]

    st.markdown(
        """
        Below is a **walk-through for each fire-year scenario**. For each, you have two realistic paths
        once the building burns down. Read these like stories — each card is one scenario.
        """
    )

    def fmt_k(n): return f"\${abs(n)/1000:,.0f}K"

    def render_fire_scenario_card(fr):
        """Render one fire-at-year-Y scenario with two side-by-side path cards."""
        y                = fr["y"]
        cash_already     = fr["cash_already"]
        mort_bal         = fr["mort_bal"]
        prop_value       = fr["prop_value"]
        equity           = fr["equity"]
        land_residual    = land_value * 0.20
        tax_recovery     = fr["tax_recovery"]
        forgiven_tax     = fr["forgiven_tax"]
        settle_total     = fr["settle_path_net_hit"]
        default_total    = fr["default_path_net_hit"]

        st.markdown(
            f"#### Scenario: Fire happens at year {y}"
        )
        st.markdown(
            f"**By the time the fire happens, you've already put in {fmt_k(cash_already)}.** "
            f"That's your down payment + closing + {y} years of carrying costs (mortgage, HOA, taxes, "
            f"maintenance) **minus** {y} years of STR income and tax savings. "
            f"This money is sunk — gone regardless of what you do next.\n\n"
            f"At year {y}, the building is worth {fmt_k(prop_value)} and you still owe "
            f"{fmt_k(mort_bal)} on the mortgage. Your equity (the wealth you've built up) "
            f"is **{fmt_k(equity)}** — that's what's at risk when the fire destroys the building."
        )

        path_a, path_b = st.columns(2)

        with path_a:
            st.markdown("##### Path A: Settle the mortgage")
            st.markdown(
                f"""
                **What you do:** pay off the remaining mortgage from other assets (savings,
                investments, HELOC) to preserve your credit score.

                **Cash flow in order:**

                1. **Right after the fire** — come up with **{fmt_k(mort_bal)}** cash to pay off the mortgage
                2. **Within a few months** — sell the empty lot for **~{fmt_k(land_residual)}** (residual land value)
                3. **Over the next 1–3 tax years** — receive **~{fmt_k(tax_recovery)}** in tax refunds from
                   the casualty-loss deduction (Schedule E)
                """
            )
            st.error(f"💸 **Total net out of pocket: {fmt_k(settle_total)}**")
            st.caption("✓ Credit score preserved.  Future borrowing unaffected.")

        with path_b:
            st.markdown("##### Path B: Default / walk away")
            st.markdown(
                f"""
                **What you do:** stop paying the mortgage. Bank forecloses on the (now-destroyed)
                property. You walk away — but take a credit hit.

                **Cash flow in order:**

                1. **Right after the fire** — pay **$0 immediately**
                2. **Next tax year** — owe **~{fmt_k(forgiven_tax)}** in 1099-C taxable income on the
                   forgiven mortgage balance (no qualified-residence exclusion for investment property)
                3. **Over the next 1–3 tax years** — receive **~{fmt_k(tax_recovery)}** in tax refunds
                   from the casualty-loss deduction
                """
            )
            st.error(f"💸 **Total net out of pocket: {fmt_k(default_total)}**")
            st.caption("✗ Credit damaged 7 years.  Future borrowing more expensive.")

        st.divider()

    for fr in fires:
        render_fire_scenario_card(fr)

    # Single comparison summary
    st.markdown("##### At a glance — total out-of-pocket across all scenarios")
    summary = pd.DataFrame([
        {
            "Fire at":          "Year 5",
            "Cash you'd have already put in by then":   fmt_k(fires[0]['cash_already']),
            "Mortgage you still owe":                    fmt_k(fires[0]['mort_bal']),
            "Path A (settle) total out-of-pocket":      fmt_k(fires[0]['settle_path_net_hit']),
            "Path B (default) total out-of-pocket":     fmt_k(fires[0]['default_path_net_hit']),
        },
        {
            "Fire at":          "Year 10",
            "Cash you'd have already put in by then":   fmt_k(fires[1]['cash_already']),
            "Mortgage you still owe":                    fmt_k(fires[1]['mort_bal']),
            "Path A (settle) total out-of-pocket":      fmt_k(fires[1]['settle_path_net_hit']),
            "Path B (default) total out-of-pocket":     fmt_k(fires[1]['default_path_net_hit']),
        },
        {
            "Fire at":          "Year 15",
            "Cash you'd have already put in by then":   fmt_k(fires[2]['cash_already']),
            "Mortgage you still owe":                    fmt_k(fires[2]['mort_bal']),
            "Path A (settle) total out-of-pocket":      fmt_k(fires[2]['settle_path_net_hit']),
            "Path B (default) total out-of-pocket":     fmt_k(fires[2]['default_path_net_hit']),
        },
    ])
    st.dataframe(summary, hide_index=True)

    st.info(
        """
        🎯  **The 1-in-25 probability over 15 years means: in 24 out of 25 simulated futures, none of
        this happens** and the property pays off normally with positive returns. The decision is whether
        you're comfortable holding the **~4% chance** of one of these outcomes against the **~96% chance**
        of the upside path.
        """
    )

    # ===== Interior fire risk note for 24-unit building =====
    st.markdown("#### Interior fire risk — updated for 24 units")
    # Per-unit NFPA: 6.6e-3/yr; P(shell loss | unit fire) ≈ 0.0054 (NFPA-decomposed)
    per_unit_rate = 6.6e-3
    p_shell      = 0.0054
    annual_interior_p_loss = 1 - (1 - per_unit_rate * p_shell) ** BUILDING_UNITS
    cum_interior_15        = 1 - (1 - annual_interior_p_loss) ** 15
    st.markdown(
        f"""
        With **24 units** in the building, the chance any one unit's interior fire spreads to total the
        structural shell rises proportionally:

        - Per-unit annual rate (NFPA apartment statistics): **{per_unit_rate*100:.2f}%/yr**
        - P(spreads to total shell loss | unit fire) ≈ **{p_shell*100:.2f}%** (NFPA-decomposed: spread × structural × total-loss)
        - **Building-level annual interior-fire loss probability:**
          1 − (1 − {per_unit_rate * p_shell:.5f})^24 ≈ **{annual_interior_p_loss*100:.3f}%/yr**
        - **15-year cumulative from interior alone:** ~**{cum_interior_15*100:.2f}%**

        This is **{annual_interior_p_loss / 2.9e-4:.1f}× higher** than my prior assumption of 8 units.
        The 4.1% combined 15-yr probability already mostly reflects this — but interior fire is now a
        materially larger share of total risk than wildfire for this building.
        """
    )

    st.markdown("##### Case for / against — updated for your financing")
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown(
            f"""
            **Against buying:**

            - Real wildfire risk is meaningful (~4.1% over 15 yrs)
            - Insurance market essentially closed for this building
            - Resale value at risk if WUI insurance crisis worsens
            - Climate trend is unfavorable
            - You inherit dual-HOA's mitigation discipline
            """
        )
    with col_right:
        st.markdown(
            f"""
            **For buying (especially given your structure):**

            - \${cash_to_close:,.0f} down — meaningful but not catastrophic exposure if things go wrong
            - Business tax write-off recovers ~32% of any casualty loss
            - Net cash hit is recoverable, not a single-year \$334K demand
            - Mitigation is strong (manager, sprinklers, defensible space)
            - Tahoe STR market value is resilient
            """
        )

    st.markdown("##### The pragmatic test — three updated questions")
    # Use year-15 scenarios for the pragmatic test
    y15  = fires[2]
    settle15_k  = y15['settle_path_net_hit'] / 1000
    default15_k = y15['default_path_net_hit'] / 1000
    mort15_k    = y15['mort_bal'] / 1000
    # Expected loss = probability-weighted over the year-15 default path
    exp_loss_k  = max(default15_k, 0) * 0.041
    st.markdown(
        f"""
        1. **In the worst case (fire at year 15), can you handle the financial impact?**
           Your two realistic paths look like this:
           - **(A) Settle the mortgage** to preserve credit: you'd need to come up with **~\${mort15_k:,.0f}K cash**
             in the year of fire. Net cumulative outflow over the recovery period: ~**\${settle15_k:,.0f}K**.
           - **(B) Default / walk away**: \$0 immediate cash; net cumulative outflow: **~\${default15_k:,.0f}K**
             (mostly your cash-to-close, since the casualty deduction roughly offsets the 1099-C tax on the
             forgiven mortgage). Trade-off: credit hit for 7 years.

           **If neither path works for you, walk away from the purchase or negotiate a 10–15% discount.**

        2. **Are you discounting purchase price for the wildfire risk?**
           Compare against comparable Tahoe condos that *can* be insured. If this is selling at parity,
           it's overpriced relative to the risk you're taking.

        3. **Have you stress-tested your STR cash-flow projection without insurance?**
           Standard STR underwriting expects insurance. If yours doesn't, your downside in a fire year
           is bigger and your exit flexibility is narrower. Run that scenario before signing.

        Probability-weighted expected loss (default path × 4.1%) is roughly **\${exp_loss_k:,.0f}K
        over 15 years** — similar in magnitude to other property holding costs spread across the
        holding period.
        """
    )

    with st.expander("If you do buy — what to track"):
        st.markdown(
            """
            - **Annual fire activity** in El Dorado/Douglas County (NIFC public data) — reassess if you see
              a Caldor-class event within 20 miles
            - **HOA budget and defensible-space maintenance** — if these lapse, your risk rises
            - **Insurance market for HOA master policies** — if a real carrier re-opens at <\$15K/year per
              owner, consider buying coverage
            - **Building manager's tenure** — when they retire, confirm who's taking over the fire-readiness
              program
            - **Your STR depreciation schedule** — track yearly so you know your adjusted basis (= the
              casualty loss you could deduct)
            """
        )

    st.divider()
    st.caption(
        "Sources: NIFC Interagency Fire Perimeter History 1984–2025 (394 fires within 50 mi), "
        "CAL FIRE DINS (132,522 records), USFS Wildfire Risk to Communities, NFPA US residential fire "
        "statistics, Cal-Adapt LOCA CMIP5 ensemble, CA FAIR Plan published rates, IRC §165/167 (casualty "
        "loss / depreciation rules). Decision-support only — consult a licensed financial/real-estate "
        "advisor and CPA before acting."
    )
