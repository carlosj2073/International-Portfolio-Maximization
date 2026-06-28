# International Portfolio Maximization

Sharpe ratio optimization across four international equity indices. Runs two strategies against the same portfolio: Long Only and With Hedging. Quantifies the performance impact of shorting highly correlated assets.

## What it does

Ingests FRED PC1 (percent change) series for US, UK, Canada, and Japan NASDAQ indices. Computes mean returns and a covariance matrix, then runs `scipy.optimize.minimize` (SLSQP) twice: once with weights bounded [0, 1] and once with [-1, 1]. Outputs optimal allocations, Sharpe ratios for each strategy, an asset allocation comparison chart, and an efficient frontier showing where the hedged strategy expands the opportunity set.

## Result

Removing the Long-Only constraint improved the Sharpe ratio by 4.3%.

## Stack

- Python
- Pandas / NumPy / SciPy
- Matplotlib
- FRED economic data

## Setup

```bash
conda env create -f environment.yml
conda activate portfolio-opt
python project.py
```

## Structure

```
fred_data/
  NASDAQCOM_PC1.csv
  NASDAQNQGBN_PC1.csv
  NASDAQNQCAN_PC1.csv
  NASDAQNQJPN_PC1.csv
project.py
environment.yml
```
