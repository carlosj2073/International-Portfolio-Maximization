import pandas as pd
import numpy as np
from scipy.optimize import minimize
from itertools import product

# Correct paths to your uploaded CSV files
uk = pd.read_csv("fred_data/NASDAQNQGBN_PC1.csv")
jp = pd.read_csv("fred_data/NASDAQNQJPN_PC1.csv")
ca = pd.read_csv("fred_data/NASDAQNQCAN_PC1.csv")
us = pd.read_csv("fred_data/NASDAQCOM_PC1.csv")


# 2. Keep only DATE and value columns, and rename value columns
# Adjust the column names if FRED uses something different (often 'VALUE')
uk = uk[["DATE", "VALUE"]].rename(columns={"VALUE": "UK"})
jp = jp[["DATE", "VALUE"]].rename(columns={"VALUE": "JAPAN"})
ca = ca[["DATE", "VALUE"]].rename(columns={"VALUE": "CANADA"})
us = us[["DATE", "VALUE"]].rename(columns={"VALUE": "US"})

# 3. Merge on DATE
data = uk.merge(jp, on="DATE").merge(ca, on="DATE").merge(us, on="DATE")

# Convert DATE to datetime and sort by date
data["DATE"] = pd.to_datetime(data["DATE"])
data = data.sort_values("DATE")
data = data.set_index("DATE")

# 4. Filter for the required period: Jan 2003 to Oct 2025
data = data.loc["2003-01-01":"2025-10-31"]

# 5. Compute monthly returns (simple returns: (P_t / P_{t-1}) - 1)
returns = data.pct_change().dropna()

# Optional: you could use log returns instead:
# returns = np.log(data / data.shift(1)).dropna()

# 6. Compute mean returns and covariance matrix
# These are *monthly* mean and covariance
mu = returns.mean()          # vector of length 4
cov = returns.cov()          # 4x4 matrix

asset_names = ["UK", "JAPAN", "CANADA", "US"]
n_assets = len(asset_names)

# 7. Define portfolio statistics
def portfolio_performance(weights, mu, cov):
    """
    Given weights, return portfolio mean and volatility.
    """
    weights = np.array(weights)
    port_return = np.dot(weights, mu)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
    return port_return, port_vol

def neg_sharpe_ratio(weights, mu, cov):
    """
    Negative Sharpe ratio for use in minimization.
    Assumes risk-free rate = 0.
    """
    port_return, port_vol = portfolio_performance(weights, mu, cov)
    # Avoid division by zero
    if port_vol == 0:
        return 1e10
    sharpe = port_return / port_vol
    return -sharpe  # negative for minimization

# 8. Constraints and bounds
# Weights must sum to 1
constraints = (
    {"type": "eq", "fun": lambda w: np.sum(w) - 1}
)

# If long-only: each weight between 0 and 1
bounds = tuple((0.0, 1.0) for _ in range(n_assets))

# 9. Initial guess: equal weights
w0 = np.array([1.0 / n_assets] * n_assets)

# 10. Run optimization
result = minimize(
    fun=neg_sharpe_ratio,
    x0=w0,
    args=(mu, cov),
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"disp": False}
)

opt_weights = result.x
opt_return, opt_vol = portfolio_performance(opt_weights, mu, cov)
opt_sharpe = opt_return / opt_vol

# 11. Print results
print("Optimal Weights (monthly data, Sharpe maximization, rf=0):")
for name, w in zip(asset_names, opt_weights):
    print(f"{name}: {w:.4f}")

print("\nPortfolio stats (monthly):")
print(f"Mean return: {opt_return:.4%}")
print(f"Volatility:  {opt_vol:.4%}")
print(f"Sharpe:      {opt_sharpe:.4f}")

# 12. (Optional) Annualize if professor wants annual Sharpe
# Assuming 12 months per year
ann_return = opt_return * 12
ann_vol = opt_vol * np.sqrt(12)
ann_sharpe = ann_return / ann_vol

print("\nAnnualized stats (assuming 12 months/year):")
print(f"Annual return: {ann_return:.4%}")
print(f"Annual vol:    {ann_vol:.4%}")
print(f"Annual Sharpe: {ann_sharpe:.4f}")