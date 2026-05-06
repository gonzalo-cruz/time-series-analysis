# Time Series Analysis & Forecasting

**Kaggle Competition · Universidad Rey Juan Carlos (URJC)**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![statsmodels](https://img.shields.io/badge/statsmodels-SARIMA%2FSARIMAX-4EABE0)
![pandas](https://img.shields.io/badge/pandas-data%20wrangling-150458?logo=pandas&logoColor=white)
![matplotlib](https://img.shields.io/badge/matplotlib-visualization-11557C)
![LaTeX](https://img.shields.io/badge/LaTeX-report-008080?logo=latex&logoColor=white)

Forecasting four time series of different frequency and nature using SARIMA/SARIMAX models. Each series script runs a full grid search to select the best model by validation RMSE, then refits on all available data.

## Series

| Series | Variable | Frequency | Train obs. | Forecast horizon |
|--------|----------|-----------|------------|-----------------|
| A | Stock price | Daily (business days) | 2025 | 10 days |
| B | Daily births | Daily | 3287 | 365 days (year 2003) |
| C | Global temperature (°C) | Monthly | 1740 | 12 months (2025) |
| D | Monthly series | Monthly | 132 | 10 months (Jan–Oct 1991) |

## Results

| Series | Best model | Val. RMSE |
|--------|-----------|-----------|
| A | SARIMA(2,0,1)(1,1,1)<sub>5</sub> | 6.16 |
| B | SARIMA(2,0,1)(1,1,1)<sub>7</sub> + Fourier K=5 | 710.10 |
| C | SARIMA(1,0,0)(0,1,1)<sub>12</sub> + linear trend (last 40y) | 0.287 |
| D | SARIMA(0,0,0)(1,0,0)<sub>12</sub> | 186.24 |

Kaggle score: **0.78** (1st place on the leaderboard).

## Project structure

```
.
├── main.py          # Runs all series and combines predictions into submission.csv
├── serie_a.py       # Series A — grid search SARIMA, m=5
├── serie_b.py       # Series B — two-phase grid search SARIMA + Fourier, m=7
├── serie_c.py       # Series C — grid search SARIMA + optional trend, m=12
├── serie_d.py       # Series D — grid search SARIMA, m=12
├── helpers.py       # Shared utilities: ADF test, plots, RMSE, grid search functions
├── sarimax_v2.py    # Standalone research script (grid search across all series)
├── informe.tex      # Full report (LaTeX)
└── informe.pdf      # Compiled report
```

> Train/test data files are not tracked. Place `train_series_A/B/C/D.csv` and `test_serie_A/B/C/D.csv` in the project root before running.

## Usage

```bash
# Run all series and generate submission.csv
python3 main.py

# Run a single series
python3 main.py --only B   # A | B | C | D

# Combine existing pred_*.csv without rerunning
python3 main.py --combine
```

Each series runs as an independent subprocess to avoid memory buildup. If `pred_X.csv` already exists, that series is skipped automatically.

## Dependencies

```bash
pip install pandas numpy matplotlib statsmodels scikit-learn
```

Python 3.10+. `pdflatex` required to recompile the report.

## Methodology notes

- **Series A**: Grid search over SARIMA with m=5 (business week). D=1 handles seasonal stationarity.
- **Series B**: Two-phase search — Phase 1: full grid with K=3 Fourier terms over a 4-year window; Phase 2: top-5 structures × K∈{3,4,5} × windows {4y, all}. Fourier terms capture annual seasonality without data leakage.
- **Series C**: Grid search on the last 40 years (1985–2024) only — earlier data distorts predictions due to the different warming rate in the 19th century. Top-5 models also tested with a linear trend exogenous regressor.
- **Series D**: Grid search with m=12. The simplest seasonal AR model generalised best with only 132 observations.
