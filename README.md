# SALY (Sunlight-Adjusted, Risk-Adjusted Yield) – Enhanced Real Estate Metric

This repo calculates a transparent 0–100 SALY score per property, using:
- **NOY**: Data-driven Net Operating Yield
- **Growth**: Local capital growth (from historical sales)
- **Vacancy penalty**
- **Liquidity**: Based on sale frequency in the area
- **Risk**: Hazard proxies (e.g. flood or bushfire if available)
- **Sunlight**: Based on orientation and road frontage from GIS data

Files included:
- `src/saly_calc.py`: Core computation module
- `app/app.py`: Flask CSV upload interface
- `notebooks/demo.ipynb`: Full scoring and plotting demo
- `data/sample_properties.csv`: Template for property input
- `tests/test_saly.py`: Smoke test

See `docs/logic.md` for metric details.

To run:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/saly_calc.py --input data/sample_properties.csv --output data/saly_scores.csv
```