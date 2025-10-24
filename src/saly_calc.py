
#!/usr/bin/env python3
"""
SALY scoring CLI (enhanced)
- Reads a property CSV (--input)
- Optionally ingests transactions.parquet, gnaf_prop.parquet, cadastre.gpkg, roads.gpkg
- Computes: NOY, Growth, Vacancy, Liquidity, Risk (placeholder), Sunlight, Frontage
- Outputs a scored CSV (--output) with SALY 0–100 and components

Example:
  python src/saly_calc.py --input data/sample_properties.csv --output data/saly_scores.csv \
    --transactions /path/transactions.parquet --gnaf /path/gnaf_prop.parquet \
    --cadastre /path/cadastre.gpkg --roads /path/roads.gpkg
"""

import argparse
import math
import sys
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd

# Geospatial imports are optional; only used if files provided
try:
    import geopandas as gpd
    from shapely.geometry import Point, LineString, Polygon
    HAS_GEO = True
except Exception:
    HAS_GEO = False

# ----------------------- Helpers -----------------------

def _z(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu, sd = s.mean(skipna=True), s.std(skipna=True)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd

def _minmax01(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mn, mx = s.min(skipna=True), s.max(skipna=True)
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(np.full(len(s), 0.5), index=s.index)
    return (s - mn) / (mx - mn)

def _minmax0100(s: pd.Series) -> pd.Series:
    return 100.0 * _minmax01(s)

def _first_present(d: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in d.columns:
            return c
    return None

def _bearing_from_linestring(ls: LineString) -> Optional[float]:
    try:
        x0, y0, x1, y1 = None, None, None, None
        # Use first and last coord for an overall bearing
        coords = list(ls.coords)
        if len(coords) < 2:
            return None
        (x0, y0), (x1, y1) = coords[0], coords[-1]
        dx, dy = x1 - x0, y1 - y0
        # Bearing from North (0 deg) clockwise: atan2(dx, dy)
        ang = math.degrees(math.atan2(dx, dy)) % 360.0
        return ang
    except Exception:
        return None

def _sun_from_bearing_deg(ang: Optional[float]) -> float:
    """Return sunlight score in [0,1], where 1 is north-facing (0°), 0 is south (180°)."""
    if ang is None:
        return 0.0
    # transform: (1 + cos(theta_rad)) / 2  with theta measured from North
    rad = math.radians(ang)
    return (1.0 + math.cos(rad)) / 2.0

def _sun_from_cardinal(txt: str) -> float:
    orient_deg = {
        "N": 0, "NE": 45, "E": 90, "SE": 135,
        "S": 180, "SW": 225, "W": 270, "NW": 315
    }
    ang = orient_deg.get(str(txt).strip().upper(), None)
    return _sun_from_bearing_deg(ang)

def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
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
        "orientation": ""
    }
    for c, v in defaults.items():
        if c not in df.columns:
            df[c] = v
    return df

# ---------------- Transactions-derived features ----------------

def load_transactions(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f"[warn] failed to read transactions from {path}: {e}", file=sys.stderr)
        return None

    # Standardize likely column names
    price_col = _first_present(df, ["price","PRICE","sale_price","contract_price","amount"])
    date_col  = _first_present(df, ["sale_date","contract_date","settlement_date","date","date_sold","DATE_SOLD","dat","DAT"])
    area_col  = _first_present(df, ["suburb","SUBURB","sa3","SA3","lga","LGA","postcode","POSTCODE","area","AREA"])

    if not price_col or not date_col:
        print("[warn] transactions missing required columns price/date; skipping growth/liquidity.", file=sys.stderr)
        return None

    df = df.copy()
    df["price_std"] = pd.to_numeric(df[price_col], errors="coerce")
    df["date_std"] = pd.to_datetime(df[date_col], errors="coerce")
    if area_col:
        df["area_std"] = df[area_col].astype(str)
    else:
        df["area_std"] = "ALL"

    df = df.dropna(subset=["price_std","date_std"])
    return df[["price_std","date_std","area_std"]]

def compute_growth(trans: pd.DataFrame, years:int=5) -> pd.Series:
    """Compute CAGR by area over `years` lookback, returns mapping area->growth (float)."""
    if trans is None or trans.empty:
        return pd.Series(dtype=float)
    # pivot median price by year
    t = trans.copy()
    t["year"] = t["date_std"].dt.year
    gp = t.groupby(["area_std","year"])["price_std"].median().unstack()
    if gp.shape[1] < 2:
        return pd.Series(dtype=float)
    max_year = gp.columns.max()
    base_year = max_year - years
    if base_year not in gp.columns:
        # pick earliest available as base
        base_year = gp.columns.min()
    try:
        ratio = gp[max_year] / gp[base_year]
        cagr = ratio.pow(1.0 / (max_year - base_year)) - 1.0
        cagr = cagr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return cagr
    except Exception:
        return pd.Series(dtype=float)

def compute_liquidity(trans: pd.DataFrame, window_years:int=2) -> pd.Series:
    """Recent sales count by area (last `window_years` years)."""
    if trans is None or trans.empty:
        return pd.Series(dtype=float)
    t = trans.copy()
    cutoff = t["date_std"].max() - pd.Timedelta(days=365*window_years)
    recent = t[t["date_std"] >= cutoff]
    liq = recent.groupby("area_std")["price_std"].count().astype(float)
    return liq

# ---------------- Frontage & Orientation from GIS ----------------

def compute_frontage_orientation(
    gnaf_path: Optional[str],
    cad_path: Optional[str],
    roads_path: Optional[str]
) -> Optional[pd.DataFrame]:
    if not HAS_GEO:
        print("[info] Geo stack unavailable; skipping frontage/orientation.", file=sys.stderr)
        return None
    if not (gnaf_path and cad_path and roads_path):
        return None
    try:
        gnaf = pd.read_parquet(gnaf_path)
    except Exception as e:
        print(f"[warn] failed to read GNAF: {e}", file=sys.stderr); return None
    try:
        cad = gpd.read_file(cad_path)
        roads = gpd.read_file(roads_path)
    except Exception as e:
        print(f"[warn] failed to read GPKG files: {e}", file=sys.stderr); return None

    # Standardize GNAF coordinates
    lon_col = _first_present(gnaf, ["lon","longitude","x","LONGITUDE","LONG","LONGITUDE_DD"])
    lat_col = _first_present(gnaf, ["lat","latitude","y","LATITUDE","LAT","LATITUDE_DD"])
    if not lon_col or not lat_col:
        print("[warn] GNAF missing lon/lat; cannot spatially join.", file=sys.stderr)
        return None

    # Create GeoDataFrame for addresses in CRS of cadastre
    gnaf_gdf = gpd.GeoDataFrame(
        gnaf,
        geometry=gpd.points_from_xy(gnaf[lon_col], gnaf[lat_col]),
        crs="EPSG:4326"
    )
    if cad.crs is None:
        cad = cad.set_crs("EPSG:4326", allow_override=True)
    gnaf_gdf = gnaf_gdf.to_crs(cad.crs)
    roads = roads.to_crs(cad.crs)

    # Spatial join addresses to cadastre polygons
    try:
        joined = gpd.sjoin(gnaf_gdf, cad[["geometry"]], how="inner", predicate="within")
    except Exception:
        # Some geopandas versions use op=within
        joined = gpd.sjoin(gnaf_gdf, cad[["geometry"]], how="inner", op="within")

    # For each polygon, compute frontage length and bearing of the longest road-intersection segment
    # Build an index of polygon -> boundary
    cad["boundary"] = cad.geometry.boundary
    results = []

    # To speed up, pre-filter roads near each polygon via bounding box (spatial index)
    try:
        roads_sindex = roads.sindex
    except Exception:
        roads_sindex = None

    for idx, poly in cad.iterrows():
        boundary = poly["boundary"]
        if boundary is None:
            continue
        # candidate roads by bbox
        if roads_sindex is not None:
            cand_idx = list(roads_sindex.intersection(poly.geometry.bounds))
            rds = roads.iloc[cand_idx]
        else:
            rds = roads

        # Compute intersections
        max_len = 0.0
        max_ls = None
        total_len = 0.0
        for _, r in rds.iterrows():
            inter = boundary.intersection(r.geometry)
            if inter.is_empty:
                continue
            if inter.geom_type == "MultiLineString":
                segs = list(inter.geoms)
            elif inter.geom_type == "LineString":
                segs = [inter]
            else:
                continue
            for seg in segs:
                seg_len = float(seg.length)
                total_len += seg_len
                if seg_len > max_len:
                    max_len = seg_len
                    max_ls = seg

        bearing = _bearing_from_linestring(max_ls) if max_ls is not None else None
        sun_score = _sun_from_bearing_deg(bearing)

        # Perimeter to normalize frontage ratio
        perim = float(boundary.length) if boundary is not None else np.nan
        frontage_ratio = (total_len / perim) if (perim and perim > 0) else 0.0

        results.append({
            "cad_index": idx,
            "frontage_len": total_len,
            "frontage_ratio": frontage_ratio,
            "bearing_deg": bearing,
            "sun_score_geo": sun_score
        })

    res_df = pd.DataFrame(results)
    if res_df.empty:
        return None

    # Map each address point to its cad_index (from spatial join)
    if "index_right" in joined.columns:
        addr_to_cad = joined.set_index(joined.index)["index_right"].to_frame(name="cad_index")
        addr_to_cad["row_id"] = joined.index
        addr_to_cad = addr_to_cad.reset_index(drop=True)
    else:
        return res_df

    # Aggregate address-level mapping (many points may map to same poly)
    return res_df

# ---------------- SALY main ----------------

def compute_noy(df: pd.DataFrame) -> pd.Series:
    price = pd.to_numeric(df["price"], errors="coerce")
    rent_week = pd.to_numeric(df["weekly_rent"], errors="coerce").fillna(0)
    vac = pd.to_numeric(df.get("vacancy_rate", 0.05), errors="coerce").fillna(0.05)
    # Costs
    council = pd.to_numeric(df.get("council_rates_pa", 1800.0), errors="coerce").fillna(1800.0)
    insurance = pd.to_numeric(df.get("insurance_pa", 1200.0), errors="coerce").fillna(1200.0)
    strata = pd.to_numeric(df.get("strata_fees_pa", 0.0), errors="coerce").fillna(0.0)
    maint_pct = pd.to_numeric(df.get("maintenance_pct", 0.01), errors="coerce").fillna(0.01)

    annual_rent = rent_week * 52.0 * (1 - vac)
    annual_costs = council + insurance + strata + maint_pct * price
    with np.errstate(divide="ignore", invalid="ignore"):
        noy = (annual_rent - annual_costs) / price
    noy = noy.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return noy

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Property CSV")
    ap.add_argument("--output", required=True, help="Output CSV with scores")
    ap.add_argument("--transactions", help="transactions.parquet")
    ap.add_argument("--gnaf", help="gnaf_prop.parquet")
    ap.add_argument("--cadastre", help="cadastre.gpkg")
    ap.add_argument("--roads", help="roads.gpkg")
    ap.add_argument("--growth_years", type=int, default=5)
    ap.add_argument("--liq_years", type=int, default=2)
    args = ap.parse_args()

    # Load properties
    props = pd.read_csv(args.input)
    props = _ensure_cols(props)

    # Base features
    props["NOY"] = compute_noy(props)
    props["VACANCY"] = pd.to_numeric(props.get("vacancy_rate", 0.05), errors="coerce").fillna(0.05)
    props["LIQ_DOM_RAW"] = -pd.to_numeric(props.get("days_on_market", np.nan), errors="coerce")  # lower DOM = better

    # Sunlight from provided orientation text (fallback)
    props["SUN_TXT"] = props.get("orientation", "").astype(str)
    props["SUN_TXT_SCORE"] = props["SUN_TXT"].apply(_sun_from_cardinal)

    # Transactions -> Growth & Liquidity by area (use 'suburb' if present)
    trans = load_transactions(args.transactions)
    growth_map = compute_growth(trans, years=args.growth_years) if trans is not None else pd.Series(dtype=float)
    liq_map = compute_liquidity(trans, window_years=args.liq_years) if trans is not None else pd.Series(dtype=float)

    # area key
    area_series = props.get("suburb", pd.Series(["ALL"] * len(props)))
    props["area_std"] = area_series.astype(str)

    props["GROWTH"] = props["area_std"].map(growth_map).fillna(growth_map.mean() if not growth_map.empty else 0.0)
    props["LIQ_SALES"] = props["area_std"].map(liq_map).fillna(liq_map.mean() if not liq_map.empty else 0.0)

    # Risk placeholder (set 0; you can map hazards here later)
    props["RISK"] = 0.0

    # Optional: GIS-derived frontage/orientation (if files exist)
    if args.gnaf and args.cadastre and args.roads:
        res_df = compute_frontage_orientation(args.gnaf, args.cadastre, args.roads)
        if res_df is not None and not res_df.empty:
            # We don't have a direct join key; use averages as global bonuses
            props["FRONTAGE_RATIO"] = res_df["frontage_ratio"].mean()
            props["SUN_GEO_SCORE"] = res_df["sun_score_geo"].mean()
        else:
            props["FRONTAGE_RATIO"] = 0.0
            props["SUN_GEO_SCORE"] = np.nan
    else:
        props["FRONTAGE_RATIO"] = 0.0
        props["SUN_GEO_SCORE"] = np.nan

    # Final Sun Score preference: explicit text orientation > GIS avg > 0
    props["SUN_SCORE"] = props["SUN_TXT_SCORE"]
    props["SUN_SCORE"] = np.where(props["SUN_SCORE"].isna() | (props["SUN_SCORE"] == 0), props["SUN_GEO_SCORE"], props["SUN_SCORE"])
    props["SUN_SCORE"] = props["SUN_SCORE"].fillna(0.0)

    # ---------------- Blend & Scale ----------------
    # Z-score within dataset (simple global z)
    z_noy = _z(props["NOY"])
    z_growth = _z(props["GROWTH"])
    z_vac = _z(props["VACANCY"])
    z_liq = _z(props["LIQ_SALES"] if props["LIQ_SALES"].notna().any() else props["LIQ_DOM_RAW"])
    z_risk = _z(props["RISK"])

    # Weights
    W = {
        "NOY": 0.40,
        "GROWTH": 0.25,
        "VACANCY": 0.15,   # enters as (1 - z_vac)
        "LIQUIDITY": 0.10,
        "RISK": -0.10,     # negative weight
        "SUN": 0.05,
        "FRONTAGE": 0.05
    }

    linear = (
        W["NOY"] * z_noy +
        W["GROWTH"] * z_growth +
        W["VACANCY"] * (1 - z_vac) +
        W["LIQUIDITY"] * z_liq +
        W["RISK"] * z_risk +
        W["SUN"] * props["SUN_SCORE"].astype(float) +
        W["FRONTAGE"] * props["FRONTAGE_RATIO"].astype(float)
    )

    props["SALY"] = _minmax0100(linear)

    # Output
    keep_cols = [
        "address","suburb","price","weekly_rent",
        "NOY","GROWTH","VACANCY","LIQ_SALES","RISK","SUN_SCORE","FRONTAGE_RATIO",
        "days_on_market","SALY"
    ]
    keep_cols = [c for c in keep_cols if c in props.columns]
    props[keep_cols].sort_values("SALY", ascending=False).to_csv(args.output, index=False)

    print(f"[ok] Wrote {args.output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
