# SALY Metric  
**Sunlight-Adjusted, risk-Adjusted Yield — a transparent 0–100 property score**

This repository turns messy sales + GIS data into a **single, explainable score (0–100)** per property so investors can shortlist quickly. It also outputs each component so you can see *why* something ranks high.

---

## TL;DR (Plain English)

- **Input:** a property CSV (address, suburb, price, weekly_rent, vacancy…).  
  **Optional:** `transactions.parquet` (sales history), and GIS files `gnaf_prop.parquet` + `cadastre.gpkg` + `roads.gpkg`.
- **Output:** `data/saly_scores.csv` with each property’s **SALY** plus component columns (Yield, Growth, Liquidity, Vacancy, Sunlight, Frontage, Risk).
- **Use:** Rank properties, then inspect component drivers to guide due-diligence.

---

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running

**Sample data:**
```bash
python src/saly_calc.py --input data/sample_properties.csv --output data/saly_scores.csv
```

**With your real data:**
```bash
python src/saly_calc.py \
  --input data/properties_from_transactions.csv \
  --output data/saly_scores.csv \
  --transactions data/transactions.parquet \
  --gnaf data/gnaf_prop.parquet \
  --cadastre data/cadastre.gpkg \
  --roads data/roads.gpkg \
  --growth_years 5 --liq_years 2
```

Notes:
- Area labels are normalised with `strip().upper()` to reduce mismatches.
- If your CSV has `orientation` (N/NE/…): that is used for sunlight; clear it to prefer the GIS proxy.

---

## Outputs

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv("data/saly_scores.csv")
df.nlargest(100, "SALY").to_csv("data/top_100_properties.csv", index=False)
(df.groupby("suburb")["SALY"].mean().sort_values(ascending=False).head(25)
).to_csv("data/top_suburbs.csv")
print("Wrote data/top_100_properties.csv and data/top_suburbs.csv")
PY
```

Optional chart:
```bash
python - <<'PY'
import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv("data/saly_scores.csv")
top = df.nlargest(20,"SALY")[["address","SALY"]]
plt.figure(figsize=(12,5))
top.plot(x="address", y="SALY", kind="bar")
plt.xticks(rotation=75, ha="right"); plt.tight_layout()
plt.savefig("data/saly_top20.png", dpi=180)
print("Saved data/saly_top20.png")
PY
```

---

## Method (with equations)

### Notation
- $P$ = price, $R_w$ = weekly rent, $v \in [0,1]$ = vacancy  
- Annual costs = council + insurance + strata + maintenance\\_pct $\\times$ $P$

### 1) Net Operating Yield (NOY)
Annual rent after vacancy: $R_a = 52\,R_w(1 - v)$

$$
\\mathrm{NOY} = \\frac{R_a - \\mathrm{annual\\_costs}}{P}
$$

### 2) Growth (CAGR by area)
Compute median sale price per **area × year**; choose latest year $Y_1$ and base $Y_0 \\in \\{Y_1-N,\\ \\text{earliest}\\}$:

$$
\\mathrm{CAGR}=\\left(\\frac{M_{Y_1}}{M_{Y_0}}\\right)^{\\tfrac{1}{Y_1 - Y_0}}-1
$$

### 3) Liquidity (area activity)
$$
\\mathrm{Liquidity}=\\#\\{\\text{sales in last } M \\text{ years}\\}
$$

### 4) Sunlight proxy (0..1)
Given bearing $\\theta$ (North $=0^\\circ$ best):

$$
\\mathrm{Sun}=\\frac{1+\\cos\\theta}{2}
$$

If `orientation` is missing, $\\theta$ is approximated from the longest road-boundary segment touching the parcel.

### 5) Frontage proxy
$$
\\mathrm{FrontageRatio}=\\frac{\\text{road boundary length}}{\\text{parcel perimeter}}
$$

### 6) Standardise & blend (transparent weights)
Z-score numeric components within the dataset:

$$
z(x)=\\frac{x-\\mu_x}{\\sigma_x}
$$

**Weights**

| Feature       | Weight |
|---------------|:------:|
| NOY           |  0.40  |
| Growth        |  0.25  |
| (1 − Vacancy) |  0.15  |
| Liquidity     |  0.10  |
| Risk          | −0.10  |
| Sunlight      |  0.05  |
| Frontage      |  0.05  |

Composite score:

$$
S=0.40\\,z(\\mathrm{NOY})+0.25\\,z(\\mathrm{Growth})+0.15\\,(1-z(\\mathrm{Vacancy}))+0.10\\,z(\\mathrm{Liquidity})-0.10\\,z(\\mathrm{Risk})+0.05\\,\\mathrm{Sun}+0.05\\,\\mathrm{Frontage}
$$

Scale to 0–100:

$$
\\mathrm{SALY}=100\\cdot\\frac{S-S_{\\min}}{S_{\\max}-S_{\\min}}
$$

---

## How to interpret SALY
- **Higher = better** (relative to your dataset).  
- Use SALY to **prioritise** due-diligence, not replace it.  
- Inspect components to understand *why* a property ranks high/low.

---

## Data & assumptions
- If `weekly_rent` is missing, rent may be estimated from price × (~4–4.5% gross) ÷ 52 (approximate).  
- Default vacancy is 5% unless provided.  
- Growth/Liquidity are area-level (typically suburb).  
- GIS sunlight/frontage are proxies; precise parcel sunlight needs a robust address→parcel join.  
- Risk is 0 until hazard layers are added.

---

## Troubleshooting
- **Identical Growth values?** Same area or label mismatches; we normalise labels but check inputs.  
- **No GIS effect?** If `orientation` is filled (e.g., “N”), it overrides GIS.  
- **Ubuntu venv errors?** `sudo apt install python3-venv`, then re-create the venv.

---

## Roadmap
- Add flood/bushfire/heat risk and include in blend  
- Parcel-level sunlight + frontage with stronger address→parcel matching  
- Hedonic rent model  
- Suburb↔SA3 hierarchical smoothing for Growth  
- Backtests, weight calibration, confidence intervals  
- Small interactive dashboard

---

## Repo layout
```
src/saly_calc.py          # main CLI (commented, human-readable)
requirements.txt
data/                     # inputs/outputs (heavy files are usually gitignored)
notebooks/demo.ipynb      # optional EDA / demo
docs/logic.md             # rationale, formulas, weights
```

## License
MIT (or update to your preferred license).
