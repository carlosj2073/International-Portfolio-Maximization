# Portfolio Optimization Project
# Group: Carlos Jaramillo, Vivek Patel, Jan Morales
import pandas as pd
import numpy as np
from scipy.optimize import minimize

# Changes the display of dataframes in the output
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Assigns respective indexes to each asset file csv as an array
asset_files = {
    'UK'    : 'fred_data/NASDAQNQGBN_PC1.csv',
    'Japan' : 'fred_data/NASDAQNQJPN_PC1.csv',
    'Canada': 'fred_data/NASDAQNQCAN_PC1.csv',
    'US'    : 'fred_data/NASDAQCOM_PC1.csv'
}

# Reads the csv files into df_list for every item in asset_files.
# This creates a list of each asseet and its data from each csv
df_list = {}
for asset_name, filename in asset_files.items():
    df = pd.read_csv(
        filename,
        index_col=0,
        parse_dates=True
    )
    df.columns = [asset_name]
    df_list[asset_name] = df

# Joins each asset's prices in df_list as columns indexed by observation date
# Drops values that do not have an observation date
prices_df = pd.concat(df_list.values(), axis=1, join='outer')
prices_df = prices_df.dropna(how='all')

# Prints data frame first 5 values
print("Joined Return Data (FRED PC1 series):")
print(prices_df.head())

# Converts prices to decimals since FRED PC1 series are percent changes
returns_df = prices_df / 100.0

# Computes expected return of each asset and covariance matrix.
# The covariance matrix includes all pairwise interactions including the asset against itself(the variance) as diagnals
mean_returns = returns_df.mean()
cov_matrix = returns_df.cov()

# Prints mean returns for each asset and covariance matrix
print("\nExpected returns for each asset:\n")
print(mean_returns)
print("\nCovariance matrix:\n")
print(cov_matrix)


# Convert index to list of asset names
assets = list(mean_returns.index)
num_assets = len(assets)

# Creates a fucntion that returns the negative Sharpe Ratio (NEGATIVE because we are minimizing the function).
# Where E(p) = weights * mean_return
# Where port_vol = transposed weights * cov_matrix * weights
# (These matrix formulas are equivalent to the summation formulas we learned in class)   
def neg_sharpe_ratio(weights, mean_returns, cov_matrix):
    weights = np.array(weights)
    port_return = np.dot(weights, mean_returns)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return - port_return / port_vol   

# Assigns constraints for optimization - weights must sum to 1
constraints = ({
    'type': 'eq',
    'fun': lambda w: np.sum(w) - 1
})

# Assigns bounds to weights between 0 and 1 (no short selling)
bounds = tuple((0.0, 1.0) for _ in range(num_assets))

# Assigns a starting point: equal weights
initial_guess = [1.0 / num_assets] * num_assets

# Runs the optimization aimed at minimizing the the objective function(the negative sharpe ratio).
# The optimization starts at the initial guess and adjusts weights until the sharpe ratio is minimized.
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

# Prints the results of the optimization
print("Optimal Portfolio Weights (no short-selling):")
for name, weight in zip(assets, optimal_weights):
    print(f"{name:10s} : {weight:.4f}")

print(f"\nMaximum Sharpe Ratio (risk-free rate = 0): {max_sharpe:.4f}\n")

