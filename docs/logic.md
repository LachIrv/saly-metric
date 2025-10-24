
## SALY Metric Logic (v2)

### Components
- **NOY** = (rent × (1 – vacancy) – annual costs) ÷ price
- **Growth** = Local capital gains CAGR from sales data
- **Risk** = Mean of known hazards (flood, bushfire, etc.)
- **Liquidity** = # of recent transactions per region
- **Sunlight Score** = cos(orientation angle)
- **Frontage Score** = street frontage length from cadastre/roads

### Final Formula
```
SALY = NOY + β1·Growth – β2·Vacancy – β3·Risk + β4·Sunlight + β5·Frontage + β6·Liquidity
```
All values are normalized; weights can be adjusted in `saly_calc.py`.
