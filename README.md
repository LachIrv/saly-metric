# SALY Metric  
**Sunlight-Adjusted, risk-Adjusted Yield — a transparent 0–100 property score**

This repository turns messy sales + GIS data into a **single, explainable score (0–100)** per property so investors can shortlist quickly. It also outputs each component so you can see *why* something ranks high.

---

## Table of Contents
- [TL;DR (Plain English)](#tldr-plain-english)
- [Quickstart](#quickstart)
- [Running](#running)
- [Outputs](#outputs)
- [Method (with equations)](#method-with-equations)
- [How to interpret SALY](#how-to-interpret-saly)
- [Data & assumptions](#data--assumptions)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Repo layout](#repo-layout)
- [License](#license)

---

## TL;DR (Plain English)

- **Input:** a property CSV (address, suburb, price, weekly_rent, vacancy…).  
  **Optional:** `transactions.parquet` (sales history), and GIS files `gnaf_prop.parquet` + `cadastre.gpkg` + `roads.gpkg`.
- **Output:** `data/saly_scores.csv` with each property’s **SALY** plus component columns (Yield, Growth, Liquidity, Vacancy, Sunlight, Frontage, Risk).
- **Use:** Rank properties, then inspect component drivers to guide due-diligence.

---

## Quickstart

```bash
# venv (Ubuntu)
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

**With real data:**
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

---

## Outputs

Generate helper exports:
```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv("data/saly_scores.csv")
df.nlargest(100, "SALY").to_csv("data/top_100_properties.csv", index=False)
(df.groupby("suburb")["SALY"].mean()
  .sort_values(ascending=False).head(25)
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

### Components
- **NOY** – Net Operating Yield (post vacancy & typical costs)  
- **Growth** – Area median price **CAGR** from transactions (default 5-year lookback)  
- **Liquidity** – Recent sales count by area (default 2-year window)  
- **Vacancy** – Penalises higher vacancy  
- **Sunlight** – From `orientation` (N/NE/…) or GIS road-bearing proxy (north ≈ better)  
- **Frontage** – Road-touching boundary / parcel perimeter (proxy)  
- **Risk** – Placeholder (0 for now; hook for flood/bushfire/heat)

### Notation
- $P$: price  
- $R_w$: weekly rent  
- $v \in [0,1] \$: vacancy rate  
- Annual costs = council + insurance + strata + maintenance\_pct × \(P\)

### 1) Net Operating Yield (NOY)
Annual rent after vacancy: $R_a = 52\cdot R_w \cdot (1 - v)$

$\text{NOY} = \frac{R_a - \text{annual\_costs}}{P}$

### 2) Growth (CAGR by area)
Compute median sale price per **area × year** from transactions; pick latest year \(Y_1\) and base \(Y_0\in\{Y_1-N,\text{earliest}\}\):

\[
\text{CAGR} = \left(\frac{M_{Y_1}}{M_{Y_0}}\right)^{\frac{1}{Y_1 - Y_0}} - 1
\]

### 3) Liquidity (area activity)
\[
\text{Liquidity} = \#\{\text{sales in last } M \text{ years}\}
\]

### 4) Sunlight proxy (0..1)
Given a bearing \(\theta\) (North = 0° best), use:

\[
\text{Sun} = \frac{1 + \cos(\theta)}{2}
\]

If `orientation` is absent, \(\theta\) is approximated from the longest road-boundary segment touching the parcel.

### 5) Frontage proxy
\[
\text{FrontageRatio} = \frac{\text{road-boundary length}}{\text{parcel perimeter}}
\]

### 6) Standardise & blend (transparent weights)
Z-score numeric components within the dataset:

\[
z(x) = \frac{x - \mu_x}{\sigma_x}
\]

Weights:

| Feature       | Weight |
|---------------|:------:|
| NOY           |  0.40  |
| Growth        |  0.25  |
| (1 − Vacancy) |  0.15  |
| Liquidity     |  0.10  |
| Risk          | −0.10  |
| Sunlight      |  0.05  |
| Frontage      |  0.05  |

Composite (linear) score:

\[
S = 0.40\,z(\text{NOY}) + 0.25\,z(\text{Growth}) + 0.15\,(1 - z(\text{Vacancy})) \\
\quad + 0.10\,z(\text{Liquidity}) - 0.10\,z(\text{Risk}) + 0.05\,\text{Sun} + 0.05\,\text{Frontage}
\]

Scale to 0–100:

\[
\text{SALY} = 100 \cdot \frac{S - S_\min}{S_\max - S_\min}
\quad (\text{flat input} \Rightarrow 50)
\]

---

## How to interpret SALY
- **Higher = better** *relative to your dataset*.  
- SALY is for **prioritising** due-diligence, not replacing it.  
- Inspect components (NOY, Growth, Liquidity, Vacancy, Sun, Frontage, Risk) to understand *why* a property ranks high/low.

---

## Data & assumptions
- If `weekly_rent` is missing, rent may be estimated from price × (~4–4.5% gross) ÷ 52 (approximate).  
- Default vacancy is 5% unless provided.  
- Growth/Liquidity are **area-level** (typically suburb); labels are normalised (`strip().upper()`).  
- GIS sunlight/frontage are **proxies**; precise parcel sunlight needs a robust address→parcel join.  
- Risk is a placeholder (0) until hazard layers are added.

---

## Troubleshooting
- **Growth identical for many rows?**  
  All top properties may share one area, or area labels didn’t match transactions; normalisation reduces this.  
- **No GIS effect?**  
  If `orientation` has values (e.g., “N”), it overrides GIS. Clear it to use GIS proxies.  
- **Ubuntu venv errors?**  
  `sudo apt install python3-venv`, then use the venv.

---

## Roadmap
- Add flood/bushfire/heat risk and include in the blend  
- Parcel-level sunlight + frontage via robust address→parcel matching  
- Hedonic rent model (instead of flat yield heuristic)  
- Suburb↔SA3 hierarchical smoothing for Growth  
- Backtests and weight calibration; confidence intervals  
- Small interactive dashboard with per-property explanations

---

## Repo layout
```
src/saly_calc.py          # main CLI (commented, human-readable)
requirements.txt
data/                     # inputs/outputs (heavy files are usually gitignored)
notebooks/demo.ipynb      # optional EDA / demo
docs/logic.md             # rationale, formulas, weights
```

---

## License
MIT (or update to your preferred license).
