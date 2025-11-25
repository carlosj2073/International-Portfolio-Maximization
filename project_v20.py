import pandas as pd
import numpy as np
from scipy.optimize import minimize

# ----------------------------------------------------------------------
# Display settings
# ----------------------------------------------------------------------
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# ----------------------------------------------------------------------
# 0. LOAD FRED DATA (Percent Change from Year Ago, Monthly, Average)
# ----------------------------------------------------------------------
asset_files = {
    'UK'    : 'fred_data/NASDAQNQGBN_PC1.csv',
    'Japan' : 'fred_data/NASDAQNQJPN_PC1.csv',
    'Canada': 'fred_data/NASDAQNQCAN_PC1.csv',
    'US'    : 'fred_data/NASDAQCOM_PC1.csv'
}

df_list = {}

for asset_name, filename in asset_files.items():
    df = pd.read_csv(
        filename,
        index_col=0,
        parse_dates=True
    )
    df.columns = [asset_name]
    df_list[asset_name] = df

# join all series on the date index
prices_df = pd.concat(df_list.values(), axis=1, join='outer')
prices_df = prices_df.dropna(how='all')

print("Joined Return Data (FRED PC1 series):")
print(prices_df.head())
print(prices_df.tail())

# FRED PC1 series are percent changes; convert to decimals
returns_df = prices_df / 100.0

# ----------------------------------------------------------------------
# Compute mean returns and covariance matrix for optimization
# ----------------------------------------------------------------------
mean_returns = returns_df.mean()
cov_matrix = returns_df.cov()

print("\nExpected returns for each asset:\n")
print(mean_returns)
print("\nCovariance matrix:\n")
print(cov_matrix)

# ----------------------------------------------------------------------
# OPTIMIZATION TO MAXIMIZE SHARPE RATIO (risk-free rate = 0)
# ----------------------------------------------------------------------
print("\nRunning Sharpe ratio optimization...\n")

# Convert index to list of asset names
assets = list(mean_returns.index)
num_assets = len(assets)

# ---- Sharpe Ratio Function (NEGATIVE because we minimize) ----
def neg_sharpe_ratio(weights, mean_returns, cov_matrix):
    weights = np.array(weights)
    port_return = np.dot(weights, mean_returns)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return - port_return / port_vol   # negative for minimization

# ---- CONSTRAINTS ----
# weights must sum to 1
constraints = ({
    'type': 'eq',
    'fun': lambda w: np.sum(w) - 1
})

# weights between 0 and 1 (no short selling)
bounds = tuple((0.0, 1.0) for _ in range(num_assets))

# starting point: equal weights
initial_guess = [1.0 / num_assets] * num_assets

# ---- RUN OPTIMIZATION ----
result = minimize(
    neg_sharpe_ratio,
    initial_guess,
    args=(mean_returns, cov_matrix),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

optimal_weights = result.x
max_sharpe = -neg_sharpe_ratio(optimal_weights, mean_returns, cov_matrix)

# ---- PRINT RESULTS ----
print("Optimal Portfolio Weights (no short-selling):")
for name, weight in zip(assets, optimal_weights):
    print(f"{name:10s} : {weight:.4f}")

print(f"\nMaximum Sharpe Ratio (risk-free rate = 0): {max_sharpe:.4f}\n")

