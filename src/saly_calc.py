#!/usr/bin/env python3
"""
SALY (Sunlight-Adjusted, risk-Adjusted Yield) – human-friendly, commented script.

What this script does (high level):
1) Read a CSV of properties you want to score (address, suburb, price, weekly_rent, etc.).
2) (Optional) Read transactions.parquet to compute LOCAL price growth (CAGR) and market liquidity.
3) (Optional) Read GNAF + Cadastre + Roads to estimate a sunlight score and frontage proxy.
4) Compute component features: NOY (net operating yield), growth, vacancy, liquidity, sunlight, frontage.
5) Blend z-scored features with weights, scale to a 0–100 SALY score, and save a ranked CSV.

Usage examples:
  # minimal (works with sample CSV)
  python src/saly_calc.py --input data/sample_properties.csv --output data/saly_scores.csv

  # full (uses your datasets)
  python src/saly_calc.py \
      --input data/properties_from_transactions.csv \
      --output data/saly_scores.csv \
      --transactions data/transactions.parquet \
      --gnaf data/gnaf_prop.parquet \
      --cadastre data/cadastre.gpkg \
      --roads data/roads.gpkg \
      --growth_years 5 --liq_years 2
"""

from __future__ import annotations

# --- stdlib and third-party imports ---
import argparse          # tidy command-line parsing
import math              # bearings & cosine for sunlight
import sys               # printing warnings to stderr
from typing import Optional, List

import numpy as np       # numeric ops
import pandas as pd      # tabular data

# Geo stack is optional; we keep the script usable without it
try:
    import geopandas as gpd
    from shapely.geometry import LineString
    HAS_GEO = True
except Exception:
    HAS_GEO = False


# ──────────────────────────────────────────────────────────────────────────────
# Small utilities (kept simple & well-commented)
# ──────────────────────────────────────────────────────────────────────────────

def pick_first(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first column name that exists in df from a priority-ordered list."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def zscore(s: pd.Series) -> pd.Series:
    """Standard z-score with safe fallbacks (NaN/zero-variance handled)."""
    s = pd.to_numeric(s, errors="coerce")
    mu, sd = s.mean(skipna=True), s.std(skipna=True)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd


def scale_0_100(s: pd.Series) -> pd.Series:
    """Min-max scale to a friendly 0–100 range (flat input -> all 50)."""
    s = pd.to_numeric(s, errors="coerce")
    mn, mx = s.min(skipna=True), s.max(skipna=True)
    if pd.isna(mn) or pd.isna(mx) or mn == mx:
        return pd.Series(np.full(len(s), 50.0), index=s.index)
    return 100.0 * (s - mn) / (mx - mn)


def bearing_from_linestring(ls: Optional[LineString]) -> Optional[float]:
    """
    Estimate a single bearing (degrees from North, clockwise) for a LineString.
    We simplify by using the line endpoints; good enough as a proxy.
    """
    if ls is None:
        return None
    try:
        (x0, y0), (x1, y1) = list(ls.coords)[0], list(ls.coords)[-1]
        # Note: atan2(dx, dy) gives angle from North when we treat dy as "northing"
        ang = math.degrees(math.atan2(x1 - x0, y1 - y0)) % 360.0
        return ang
    except Exception:
        return None


def sun_from_bearing_deg(ang: Optional[float]) -> float:
    """
    Convert a bearing angle to a sunlight score in [0,1].
    We assume North=best sun in AU; simple proxy using cosine of the angle from North.
    """
    if ang is None:
        return 0.0
    return (1 + math.cos(math.radians(ang))) / 2.0


def sun_from_cardinal(txt: str) -> float:
    """Map N/NE/E/… to a bearing, then to a sun score. Unknown -> 0."""
    mapping = {
        "N": 0, "NE": 45, "E": 90, "SE": 135,
        "S": 180, "SW": 225, "W": 270, "NW": 315
    }
    ang = mapping.get(str(txt).strip().upper(), None)
    return sun_from_bearing_deg(ang)


def ensure_property_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the input CSV has the columns we need.
    If missing, create with reasonable defaults so the pipeline is robust.
    """
    defaults = {
        "address": "",
        "suburb": "",
        "price": np.nan,
        "weekly_rent": np.nan,
        "vacancy_rate": 0.05,
        "council_rates_pa": 1800.0,
        "insurance_pa": 1200.0,
        "strata_fees_pa": 0.0,
        "maintenance_pct": 0.01,
        "days_on_market": np.nan,
        "orientation": "",
    }
    out = df.copy()
    for c, v in defaults.items():
        if c not in out.columns:
            out[c] = v
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Core feature engineering
# ──────────────────────────────────────────────────────────────────────────────

def compute_noy(properties: pd.DataFrame) -> pd.Series:
    """
    Net Operating Yield = (rent*(1 - vacancy) − annual costs) / price
    - rent = weekly_rent * 52
    - vacancy = vacancy_rate (0.05 default)
    - annual costs = council + insurance + strata + maintenance_pct * price
    """
    price = pd.to_numeric(properties["price"], errors="coerce")
    rent_week = pd.to_numeric(properties["weekly_rent"], errors="coerce").fillna(0)
    vacancy = pd.to_numeric(properties.get("vacancy_rate", 0.05), errors="coerce").fillna(0.05)

    council = pd.to_numeric(properties.get("council_rates_pa", 1800.0), errors="coerce").fillna(1800.0)
    insurance = pd.to_numeric(properties.get("insurance_pa", 1200.0), errors="coerce").fillna(1200.0)
    strata = pd.to_numeric(properties.get("strata_fees_pa", 0.0), errors="coerce").fillna(0.0)
    maint_pct = pd.to_numeric(properties.get("maintenance_pct", 0.01), errors="coerce").fillna(0.01)

    annual_rent = rent_week * 52.0 * (1 - vacancy)
    annual_costs = council + insurance + strata + maint_pct * price

    with np.errstate(divide="ignore", invalid="ignore"):
        noy = (annual_rent - annual_costs) / price

    # Replace infinities/NaNs with 0 so the pipeline never crashes
    return noy.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def load_transactions(path: Optional[str]) -> Optional[pd.DataFrame]:
    """
    Read transactions parquet and standardise the 3 essentials:
      price_std  – numeric price
      date_std   – pandas datetime
      area_std   – area grouping (suburb/SA3/LGA/etc) for growth/liquidity
    We support a bunch of common column names to handle schema differences.
    """
    if not path:
        return None

    try:
        raw = pd.read_parquet(path)
    except Exception as e:
        print(f"[warn] failed to read transactions: {e}", file=sys.stderr)
        return None

    # Try to find equivalent columns by priority
    price_col = pick_first(raw, ["price", "PRICE", "sale_price", "contract_price", "amount"])
    date_col  = pick_first(raw, ["sale_date", "contract_date", "settlement_date", "date",
                                 "date_sold", "DATE_SOLD", "dat", "DAT"])
    area_col  = pick_first(raw, ["suburb", "SUBURB", "sa3", "SA3", "lga", "LGA",
                                 "postcode", "POSTCODE", "area", "AREA"])

    if not price_col or not date_col:
        # Graceful fallback: we can still score without growth/liquidity
        print("[warn] transactions missing price/date; skipping growth/liquidity.", file=sys.stderr)
        return None

    df = raw.copy()
    df["price_std"] = pd.to_numeric(df[price_col], errors="coerce")
    df["date_std"]  = pd.to_datetime(df[date_col], errors="coerce")
    df["area_std"]  = df[area_col].astype(str) if area_col else "ALL"

    df = df.dropna(subset=["price_std", "date_std"])
    return df[["price_std", "date_std", "area_std"]]


def compute_growth(trans: pd.DataFrame, years: int = 5) -> pd.Series:
    """
    Compute area-level CAGR from median prices.
    Steps:
      1) Median price per area×year.
      2) CAGR between latest year and (latest - years) or earliest available.
    Returns: Series indexed by area_std with CAGR values in decimal (e.g., 0.05 = 5%)
    """
    if trans is None or trans.empty:
        return pd.Series(dtype=float)

    t = trans.copy()
    t["year"] = t["date_std"].dt.year

    # Table: rows=area, cols=year, values=median price
    med = t.groupby(["area_std", "year"])["price_std"].median().unstack()

    if med.shape[1] < 2:
        return pd.Series(dtype=float)

    latest_year = med.columns.max()
    base_year = max(med.columns.min(), latest_year - years)

    ratio = (med[latest_year] / med[base_year]).replace([np.inf, -np.inf], np.nan)
    cagr = ratio.pow(1.0 / (latest_year - base_year)).fillna(1.0) - 1.0
    return cagr.fillna(0.0)


def compute_liquidity(trans: pd.DataFrame, window_years: int = 2) -> pd.Series:
    """
    Liquidity proxy = count of recent sales per area (last N years).
    Higher = more active market (easier to enter/exit).
    """
    if trans is None or trans.empty:
        return pd.Series(dtype=float)

    cutoff = trans["date_std"].max() - pd.Timedelta(days=365 * window_years)
    recent = trans[trans["date_std"] >= cutoff]
    return recent.groupby("area_std")["price_std"].count().astype(float)


def frontage_and_sunlight_from_gis(
    gnaf_path: Optional[str],
    cad_path: Optional[str],
    roads_path: Optional[str],
) -> Optional[pd.DataFrame]:
    """
    VERY simple frontage/sunlight proxy from GIS:
      - For each cadastre polygon, intersect boundary with nearby road lines.
      - Sum intersection length (frontage proxy) and take bearing of longest segment.
      - Convert that bearing to a [0,1] sunlight score assuming North=best.

    NOTE: This is a lightweight proxy. A production model would join addresses to exact parcels
          and compute per-property values. Here we fall back to dataset means if joins are hard.
    """
    if not (HAS_GEO and gnaf_path and cad_path and roads_path):
        return None

    try:
        gnaf = pd.read_parquet(gnaf_path)
        cad = gpd.read_file(cad_path)
        roads = gpd.read_file(roads_path)
    except Exception as e:
        print(f"[warn] GIS read failed: {e}", file=sys.stderr)
        return None

    # Find lon/lat columns in GNAF
    lon = pick_first(gnaf, ["lon", "longitude", "x", "LONGITUDE", "LONG"])
    lat = pick_first(gnaf, ["lat", "latitude", "y", "LATITUDE", "LAT"])
    if not lon or not lat:
        return None

    # Build GeoDataFrames in a common CRS (use cadastre CRS if present)
    gnaf_gdf = gpd.GeoDataFrame(
        gnaf, geometry=gpd.points_from_xy(gnaf[lon], gnaf[lat]), crs="EPSG:4326"
    )
    cad = cad if cad.crs else cad.set_crs("EPSG:4326")
    gnaf_gdf = gnaf_gdf.to_crs(cad.crs)
    roads = roads.to_crs(cad.crs)

    # Join addresses to cadastre polygons (rough check)
    try:
        joined = gpd.sjoin(gnaf_gdf, cad[["geometry"]], how="inner", predicate="within")
    except Exception:
        # for older geopandas versions
        joined = gpd.sjoin(gnaf_gdf, cad[["geometry"]], how="inner", op="within")

    # Pre-compute polygon boundaries
    cad = cad.copy()
    cad["boundary"] = cad.geometry.boundary

    # Spatial index on roads for speed (if available)
    sidx = getattr(roads, "sindex", None)

    rows = []
    for i, poly in cad.iterrows():
        boundary = poly["boundary"]
        if boundary is None:
            continue

        # Restrict candidate roads to the polygon's bbox to keep it quick
        if sidx is not None:
            cand = roads.iloc[list(sidx.intersection(poly.geometry.bounds))]
        else:
            cand = roads

        # Intersect road lines with polygon boundary and accumulate
        total_len = 0.0
        longest_seg = None
        longest_len = 0.0

        for _, r in cand.iterrows():
            inter = boundary.intersection(r.geometry)
            # handle LineString and MultiLineString uniformly
            if getattr(inter, "geoms", None):
                segs = list(inter.geoms)
            elif getattr(inter, "geom_type", "") == "LineString":
                segs = [inter]
            else:
                segs = []

            for seg in segs:
                L = float(seg.length)
                total_len += L
                if L > longest_len:
                    longest_len = L
                    longest_seg = seg

        # Convert the longest segment to a bearing & then a sun score
        bearing = bearing_from_linestring(longest_seg)
        sun_geo = sun_from_bearing_deg(bearing)

        perimeter = float(boundary.length) if boundary is not None else np.nan
        frontage_ratio = (total_len / perimeter) if (perimeter and perimeter > 0) else 0.0

        rows.append({"cad_index": i, "frontage_ratio": frontage_ratio, "sun_geo": sun_geo})

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Main program
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    # ---- 1) Arguments ----
    ap = argparse.ArgumentParser(description="Compute SALY 0–100 scores for a property CSV.")
    ap.add_argument("--input", required=True, help="Property CSV (address, suburb, price, weekly_rent, ...)")
    ap.add_argument("--output", required=True, help="Output CSV to write with scores")
    ap.add_argument("--transactions", help="transactions.parquet (for growth/liquidity)")
    ap.add_argument("--gnaf", help="gnaf_prop.parquet (addresses)")
    ap.add_argument("--cadastre", help="cadastre.gpkg (property polygons)")
    ap.add_argument("--roads", help="roads.gpkg (road centerlines)")
    ap.add_argument("--growth_years", type=int, default=5, help="Lookback years for CAGR (default 5)")
    ap.add_argument("--liq_years", type=int, default=2, help="Window for recent-sales liquidity (default 2)")
    args = ap.parse_args()

    # ---- 2) Read properties & ensure required columns exist ----
    props_raw = pd.read_csv(args.input)
    props = ensure_property_columns(props_raw)

    # ---- 3) Compute primary features on the properties table ----
    props["NOY"] = compute_noy(props)                                   # yield
    props["VACANCY"] = pd.to_numeric(props["vacancy_rate"], errors="coerce").fillna(0.05)

    # Sunlight from text orientation if present (N/NE/…); GIS can override later
    props["SUN_TXT_SCORE"] = props["orientation"].astype(str).apply(sun_from_cardinal)

    # Area key for growth/liquidity mapping (prefer suburb; fall back to "ALL")
    props["area_std"] = props.get("suburb", "ALL").astype(str)

    # ---- 4) Transactions → growth & liquidity ----
    trans = load_transactions(args.transactions)
    growth_map = compute_growth(trans, years=args.growth_years) if trans is not None else pd.Series(dtype=float)
    liq_map    = compute_liquidity(trans, window_years=args.liq_years)  if trans is not None else pd.Series(dtype=float)

    # Map area-level values to each property (fill with mean if missing)
    props["GROWTH"]    = props["area_std"].map(growth_map).fillna(growth_map.mean() if not growth_map.empty else 0.0)
    props["LIQ_SALES"] = props["area_std"].map(liq_map).fillna(liq_map.mean() if not liq_map.empty else 0.0)

    # Risk placeholder for now (0 = neutral). Plug hazard datasets here later.
    props["RISK"] = 0.0

    # ---- 5) Optional GIS frontage/sunlight proxy (dataset-level fallback) ----
    geo = frontage_and_sunlight_from_gis(args.gnaf, args.cadastre, args.roads)
    if geo is not None and not geo.empty:
        # For this quick pass we use dataset averages; a follow-up could compute per-address joins.
        props["FRONTAGE_RATIO"] = float(geo["frontage_ratio"].mean())
        props["SUN_GEO_SCORE"]  = float(geo["sun_geo"].mean())
    else:
        props["FRONTAGE_RATIO"] = 0.0
        props["SUN_GEO_SCORE"]  = np.nan

    # Prefer explicit text orientation; if it's absent/zero, use GIS score; otherwise 0.
    props["SUN_SCORE"] = pd.Series(np.where(props["SUN_TXT_SCORE"].fillna(0)==0, props["SUN_GEO_SCORE"], props["SUN_TXT_SCORE"]), index=props.index).fillna(0.0)

    # ---- 6) Blend features into a single, readable score ----
    # z-score the numeric components so each weight is comparable
    z_noy     = zscore(props["NOY"])
    z_growth  = zscore(props["GROWTH"])
    z_vac     = zscore(props["VACANCY"])
    z_liq     = zscore(props["LIQ_SALES"])
    z_risk    = zscore(props["RISK"])

    # Transparent weights (sum to ~1; risk negative)
    W = {
        "NOY": 0.40,
        "GROWTH": 0.25,
        "VACANCY": 0.15,   # we flip sign below (1 - z_vac) so higher vacancy reduces score
        "LIQUIDITY": 0.10,
        "RISK": -0.10,
        "SUN": 0.05,
        "FRONTAGE": 0.05,
    }

    # Linear blend of standardised features + raw [0..1] sun/frontage proxies
    blended = (
        W["NOY"]       * z_noy +
        W["GROWTH"]    * z_growth +
        W["VACANCY"]   * (1 - z_vac) +  # lower vacancy is better → (1 - z)
        W["LIQUIDITY"] * z_liq +
        W["RISK"]      * z_risk +
        W["SUN"]       * props["SUN_SCORE"].astype(float) +
        W["FRONTAGE"]  * props["FRONTAGE_RATIO"].astype(float)
    )

    # Friendly 0–100 SALY for easy sorting/comms
    props["SALY"] = scale_0_100(blended)

    # ---- 7) Save a compact, sorted output table ----
    columns_to_keep = [
        "address", "suburb", "price", "weekly_rent",
        "NOY", "GROWTH", "VACANCY", "LIQ_SALES", "RISK",
        "SUN_SCORE", "FRONTAGE_RATIO", "days_on_market",
        "SALY",
    ]
    columns_to_keep = [c for c in columns_to_keep if c in props.columns]

    props[columns_to_keep].sort_values("SALY", ascending=False).to_csv(args.output, index=False)
    print(f"[ok] Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
